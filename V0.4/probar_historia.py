"""Prueba retrospectiva de historia proveedor/comprador, con corte por etapa.
No hay datos independientes de 2025; las tasas de adjudicación son experimentales.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import sys
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.exceptions import ConvergenceWarning
from sklearn.frozen import FrozenEstimator

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'V0.3'))
import modelo_temporal as prev

KEY=['Codigo','Codigoitem','CodigoProveedor','CodigoSucursalProveedor','Correlativo']
GROUP=['Codigo','CodigoProveedor','CodigoUnidad','CodigoProductoONU']
PAIRS={'proveedor':['CodigoProveedor'],
       'proveedor_comprador':['CodigoProveedor','CodigoUnidad'],
       'proveedor_producto':['CodigoProveedor','CodigoProductoONU']}
VARIANTS=['base','participaciones','resultados_proxy']


def read_data(source,output,fraction=.25):
    chunks,events,manifest=[],[],[]
    cols=list(dict.fromkeys(prev.READ_COLS+['FechaEnvioOferta']))
    files=sorted(Path(source).glob('lic_2024-*.csv'))
    if len(files)!=12:
        raise ValueError('Se necesitan los doce CSV de 2024')
    for p in files:
        with p.open('rb') as stream:
            sha=hashlib.file_digest(stream,'sha256').hexdigest()
        n=0
        for d in pd.read_csv(p,sep=';',encoding='cp1252',encoding_errors='replace',dtype=str,
                keep_default_na=False,usecols=cols,chunksize=25000):
            n+=len(d)
            ids=prev.clean(d.Codigo)
            mask=ids.notna()&(prev.hash_groups(ids.fillna('sin_codigo'))/np.float64(2**64)<fraction)
            q=d.loc[mask].copy()
            q['_archivo']=p.name
            chunks.append(q)
            e=d[GROUP].copy()
            e['envio']=prev.dates(d.FechaEnvioOferta)
            e['publicacion']=prev.dates(d.FechaPublicacion)
            e['adjudicacion']=prev.dates(d.FechaAdjudicacion)
            e['ganada']=d['Oferta seleccionada'].eq('Seleccionada')
            e['resultado_valido']=d.Estado.eq('Adjudicada')&d['Oferta seleccionada'].isin(['Seleccionada','No Seleccionada'])
            e['envio_valido']=e.envio.notna()&e.publicacion.notna()&e.envio.ge(e.publicacion)
            e['fecha_resultado_valida']=e.adjudicacion.notna()&e.adjudicacion.ge(e.envio)&e.adjudicacion.le('2026-11-04')
            events.append(e.groupby(GROUP,dropna=False).agg(
                envio=('envio','max'),publicacion=('publicacion','min'),adjudicacion=('adjudicacion','max'),
                ganada=('ganada','max'),resultado_valido=('resultado_valido','all'),
                envio_valido=('envio_valido','all'),fecha_resultado_valida=('fecha_resultado_valida','all')).reset_index())
        manifest.append({'archivo':p.name,'sha256':sha,'filas':n,'bytes':p.stat().st_size})
        print(p.name,n,flush=True)
    raw=pd.concat(chunks,ignore_index=True)
    e=pd.concat(events,ignore_index=True).groupby(GROUP,dropna=False).agg(
        envio=('envio','max'),publicacion=('publicacion','min'),adjudicacion=('adjudicacion','max'),
        ganada=('ganada','max'),resultado_valido=('resultado_valido','all'),
        envio_valido=('envio_valido','all'),fecha_resultado_valida=('fecha_resultado_valida','all')).reset_index()
    raw.to_parquet(output/'muestra_cruda.parquet',index=False)
    e.to_parquet(output/'eventos_completos.parquet',index=False)
    prev.save_json(output/'fuentes.json',manifest)
    return raw,e


def event_ledger(events,keys,lag):
    """Una participación/resultado por licitación y clave, no por fila o ítem repetido."""
    grouping=['Codigo']+keys
    valid_keys=pd.Series(True,index=events.index)
    for c in keys:
        valid_keys &= prev.clean(events[c]).notna()
    e=events.loc[valid_keys].groupby(grouping,dropna=False).agg(
        envio=('envio','max'),adjudicacion=('adjudicacion','max'),
        ganada=('ganada','max'),resultado_valido=('resultado_valido','all'),
        envio_valido=('envio_valido','all'),fecha_resultado_valida=('fecha_resultado_valida','all')).reset_index()
    participation=e.loc[e.envio_valido].copy()
    participation['fecha']=participation.envio
    participation['n']=1
    result=e.loc[e.envio_valido&e.resultado_valido&e.fecha_resultado_valida].copy()
    result['fecha']=result.adjudicacion+pd.to_timedelta(lag,unit='D')
    result['n']=1
    result['w']=result.ganada.astype(int)
    return participation,result


def lookup(ledger,queries,keys,values):
    """Acumulados estrictamente anteriores a la consulta; no actualiza dentro del día."""
    answer=pd.DataFrame(0.,index=queries.index,columns=values)
    if ledger.empty:
        return answer
    # Unión temporal vectorizada: evita crear cientos de miles de DataFrames
    # separados cuando hay muchas parejas proveedor-producto.
    daily=ledger.groupby(keys+['fecha'],dropna=False)[values].sum().reset_index()
    daily=daily.sort_values(keys+['fecha'])
    daily[values]=daily.groupby(keys,dropna=False)[values].cumsum()
    q=queries.loc[queries['_consulta'].notna(),keys+['_consulta']].copy()
    if q.empty:
        return answer
    q['_indice_original']=q.index
    merged=pd.merge_asof(q.sort_values('_consulta'),daily.sort_values('fecha'),
                         left_on='_consulta',right_on='fecha',by=keys,
                         direction='backward',allow_exact_matches=False)
    answer.loc[merged['_indice_original'],values]=merged[values].fillna(0).to_numpy()
    return answer


def historical_features(d,events,lag=30):
    q=d.copy()
    q['_consulta']=prev.dates(q.FechaEnvioOferta)
    for part,(start,end) in prev.BOUNDARIES.items():
        if part!='train':
            # Congela el conocimiento al inicio de cada etapa; ninguna etiqueta de
            # validación/test actualiza la historia dentro de esa misma etapa.
            mask=q._particion.eq(part)
            q.loc[mask,'_consulta']=q.loc[mask,'_consulta'].clip(upper=pd.Timestamp(start))
    x=pd.DataFrame(index=d.index)
    for label,keys in PAIRS.items():
        participation,results=event_ledger(events,keys,lag)
        n=lookup(participation,q,keys,['n']).n
        r=lookup(results,q,keys,['n','w'])
        x[label+'_participaciones_log']=np.log1p(n)
        x[label+'_resultados_log']=np.log1p(r.n)
        x[label+'_adjudicadas_log']=np.log1p(r.w)
        # Prior fijo, no estimado mirando el futuro; mide ganar algún ítem del proceso.
        x[label+'_tasa_suavizada']=(r.w+1)/(r.n+5)
    return x


def cohort(raw,asof):
    d,parts,audit=prev.partition(raw,asof)
    duplicate=d.duplicated(KEY,keep=False)
    key_missing=pd.concat([prev.clean(d[c]).isna() for c in KEY],axis=1).any(axis=1)
    send=prev.dates(d.FechaEnvioOferta)
    timestamp_valid=send.notna()&send.ge(d._pub)&send.le(d._adj)
    good=~duplicate&~key_missing&timestamp_valid&d._particion.ne('excluida')
    d=d.loc[good].copy().reset_index(drop=True)
    # Un ID nunca cambia de conjunto; se filtran ofertas ambiguas, no se reasignan.
    parts={k:np.flatnonzero(d._particion.eq(k)) for k in prev.BOUNDARIES}
    for k,idx in parts.items():
        if len(idx)<200 or d.iloc[idx]._y.nunique()!=2:
            raise ValueError('Población insuficiente en '+k)
    counts={'muestra_original':len(raw),'filas_clave_repetida':int(duplicate.sum()),
            'filas_clave_incompleta':int(key_missing.sum()),
            'filas_envio_invalido':int((~timestamp_valid).sum()),
            'filas_utilizadas':len(d),'nota':'Causas solapadas; se excluyen claves ambiguas sin deduplicar ofertas distintas.'}
    return d,parts,audit,counts


def design(d,h,variant):
    x=prev.features(d,'proveedor')
    if variant=='participaciones':
        return pd.concat([x,h[[c for c in h if c.endswith('_participaciones_log')]]],axis=1)
    if variant=='resultados_proxy':
        return pd.concat([x,h],axis=1)
    if variant!='base':
        raise ValueError(variant)
    return x


def ranking(frame,prob_col,ks=(10,20)):
    """Top-k por proveedor y mes; desempate estable, sin etiqueta.

    Se exige >=k ofertas por lote. Es un diagnóstico retrospectivo de priorización,
    no una lista de oportunidades disponible simultáneamente en una fecha real.
    """
    result=[]
    for k in ks:
        groups=0
        wins=total=positive=0
        expected=0.
        recall_macro=[]
        for _,g in frame.groupby(['CodigoProveedor','mes'],sort=False):
            if len(g)<k:
                continue
            top=g.sort_values([prob_col,'Codigo','Codigoitem','Correlativo'],ascending=[False,True,True,True],kind='stable').head(k)
            hit=int(top.y.sum())
            groups+=1
            wins+=hit
            total+=k
            positive+=int(g.y.sum())
            expected+=k*g.y.mean()
            if g.y.sum():
                recall_macro.append(hit/g.y.sum())
        result.append({'k':k,'lotes_elegibles':groups,'recomendaciones':total,
                       'seleccionadas':wins,'precision_topk':wins/total if total else 0.,
                       'precision_aleatoria_esperada':expected/total if total else 0.,
                       'lift_sobre_aleatorio':wins/expected if expected else 0.,
                       'recall_lotes_elegibles':wins/positive if positive else 0.,
                       'recall_macro_lotes_con_positivos':float(np.mean(recall_macro)) if recall_macro else 0.})
    return result


def train(raw,events,output,args):
    d,parts,audit,counts=cohort(raw,args.asof)
    prev.save_json(output/'cohorte.json',counts)
    audit.to_csv(output/'auditoria_licitaciones.csv',encoding='utf-8-sig')
    history=historical_features(d,events,args.demora)
    d[['Codigo','CodigoProveedor','CodigoUnidad','CodigoProductoONU','FechaEnvioOferta','_particion']].join(history).to_parquet(output/'historia_consultada.parquet',index=False)
    fit=prev.limit_train(d,parts['train'],args.max_train)
    cal,val,test=parts['calibracion'],parts['validacion'],parts['test']
    summary={k:{'filas':len(idx),'licitaciones':int(d.iloc[idx].Codigo.nunique()),'prevalencia':float(d.iloc[idx]._y.mean())} for k,idx in parts.items()}
    summary['train']['filas_ajuste']=len(fit)
    prev.save_json(output/'particiones.json',summary)
    fitted,rows={},[]
    for variant in VARIANTS:
        x=design(d,history,variant)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always',ConvergenceWarning)
            base=prev.pipeline(x,'logistica').fit(x.iloc[fit],d._y.iloc[fit])
        model=CalibratedClassifierCV(FrozenEstimator(base),method='sigmoid').fit(x.iloc[cal],d._y.iloc[cal])
        p=model.predict_proba(x.iloc[val])[:,1]
        t=prev.threshold(d._y.iloc[val],p,args.recall,200)
        converged=not any(issubclass(w.category,ConvergenceWarning) for w in caught)
        row={'variante':variant,'umbral':t,'convergio':converged,
             **prev.metrics(d._y.iloc[val],p,t if t is not None else .5)}
        rows.append(row)
        fitted[variant]=(model,x.columns.tolist())
        print('Validación',variant,'precision',row['precision'],'recall',row['recall'],flush=True)
    prev.save_json(output/'validacion.json',rows)
    # Las fechas proxy impiden promover tasas de éxito; selección primaria solo
    # entre base y participaciones. La tercera variante es sensibilidad declarada.
    eligible=pd.DataFrame(rows).query("variante != 'resultados_proxy' and convergio").dropna(subset=['umbral'])
    winner=eligible.sort_values(['precision','recall','brier'],ascending=[False,False,True]).iloc[0]
    decision={'variante':winner.variante,'umbral':float(winner.umbral),'antes_de_test':True,
              'criterio':'Precisión con recall mínimo en validación; resultados_proxy excluida de selección primaria',
              'recall_minimo':args.recall,'min_alertas':200}
    prev.save_json(output/'decision.json',decision)
    predictions=d.iloc[test][KEY+['FechaEnvioOferta','sector','_y']].rename(columns={'_y':'y'}).reset_index(drop=True)
    predictions['mes']=prev.dates(predictions.FechaEnvioOferta).dt.strftime('%Y-%m')
    scores,rankings,thresholds={},[],{}
    for row in rows:
        variant=row['variante']
        model,_=fitted[variant]
        p=model.predict_proba(design(d.iloc[test],history.iloc[test],variant))[:,1]
        t=row['umbral'] if row['umbral'] is not None else .5
        predictions[variant]=p
        scores[variant]=prev.metrics(d._y.iloc[test],p,t)
        thresholds[variant]=t
        rankings.extend([{'variante':variant,**r} for r in ranking(predictions,variant)])
    chosen=[(winner.variante,thresholds[winner.variante])]
    if winner.variante!='base':
        chosen.append(('base',thresholds['base']))
    bootstrap=prev.cluster_bootstrap(predictions,chosen)
    predictions.to_parquet(output/'predicciones_test.parquet',index=False)
    pd.DataFrame(rankings).to_csv(output/'ranking_topk.csv',index=False)
    diagnostics=[]
    for month,g in predictions.groupby('mes'):
        if g.y.nunique()==2:
            for variant in VARIANTS:
                diagnostics.append({'mes':month,'variante':variant,**prev.metrics(g.y,g[variant],thresholds[variant])})
    prev.save_json(output/'diagnosticos_mensuales.json',diagnostics)
    report={'version':'0.4.0','decision':decision,'particiones':summary,'cohorte':counts,
            'test':scores,'ranking':rankings,'bootstrap_por_licitacion':bootstrap,
            'promovido':False,'historia_desde_original_completo':True,
            'demora_resultados_proxy_dias':args.demora,
            'limitaciones':[
                'Solo 2024; todo el periodo ya fue observado. No hay confirmación independiente.',
                'La historia de participaciones supone que envíos previos eran observables; para competidores puede ser falso. Debe existir registro propio o acceso efectivo.',
                'Las tasas de adjudicación usan fecha final más demora, no fecha efectiva verificada; variante solo experimental.',
                'El modelo clasifica ofertas de distintos proveedores; una aplicación para una empresa requiere evaluar su cartera específica.',
                'Top-k agrupa retrospectivamente por proveedor y mes de envío; no demuestra disponibilidad simultánea de oportunidades.',
                'Se excluyen claves repetidas y envíos inválidos. La población difiere de V0.3 y no se comparan directamente sus cifras.',
                'Se congela la historia en el inicio de calibración/validación/test; no se actualiza con etiquetas del mismo conjunto.',
                'Historia desde enero: no hay antecedentes de años anteriores y el entrenamiento tiene arranque sin historia.',
                'La selección por madurez de resultado introduce sesgo hacia procesos rápidos; faltan snapshots y horizontes comparables.'
            ]}
    prev.save_json(output/'informe.json',report)
    model,columns=fitted[winner.variante]
    joblib.dump({'version':'0.4.0','modelo':model,'columnas':columns,'variante':winner.variante,
                 'umbral':float(winner.umbral),'promovido':False,
                 'contrato':'predict() requiere campos crudos e historia anterior auditada; no acepta historia futura'},output/'modelo_candidato.joblib')
    # Guarda una entrada preparada y comprueba paridad tras serialización.
    example=design(d.iloc[test[:12]],history.iloc[test[:12]],winner.variante)
    example.to_parquet(output/'ejemplo_preparado.parquet',index=False)
    restored=joblib.load(output/'modelo_candidato.joblib')
    np.testing.assert_allclose(restored['modelo'].predict_proba(example)[:,1],predictions[winner.variante].iloc[:len(example)])
    prev.save_json(output/'estado_ejecucion.json',{'estado':'completa'})
    print(json.dumps(report['test'],ensure_ascii=False,indent=2),flush=True)
    print('Resultados:',output,flush=True)
    return report


def predict(artifact,raw,history):
    """La historia debe provenir de historical_features con cortes auditados."""
    if artifact.get('version')!='0.4.0' or not raw.index.equals(history.index):
        raise ValueError('Versión o índices incompatibles')
    x=design(raw,history,artifact['variante'])
    if x.columns.tolist()!=artifact['columnas']:
        raise ValueError('Esquema incompatible')
    p=artifact['modelo'].predict_proba(x)[:,1]
    return pd.DataFrame({'probabilidad':p,'prediccion':(p>=artifact['umbral']).astype(int)},index=raw.index)


def main():
    if hasattr(sys.stdout,'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8',errors='backslashreplace')
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fuente',default='D:/Licit')
    parser.add_argument('--max-train',type=int,default=80000)
    parser.add_argument('--demora',type=int,default=30)
    parser.add_argument('--recall',type=float,default=.30)
    parser.add_argument('--asof',default='2026-11-04')
    parser.add_argument('--datos-preparados',type=Path,help='Carpeta de extracción previa; verifica SHA256 de los CSV antes de usarla')
    args=parser.parse_args()
    if args.demora<1 or not 0<args.recall<=1:
        parser.error('Demora o recall inválidos')
    output=ROOT/'resultados'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output.mkdir(parents=True)
    arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    prev.save_json(output/'configuracion.json',{'argumentos':arguments,'fraccion':.25,
        'sha256_codigo':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'sha256_v03':hashlib.sha256(Path(prev.__file__).read_bytes()).hexdigest(),
        'python':sys.version,'pandas':pd.__version__,'numpy':np.__version__,
        'sklearn':prev.sklearn.__version__,'plan':'Base vs participaciones; resultados_proxy sensibilidad no elegible; selección validación y ranking top10/20 proveedor-mes'})
    if args.datos_preparados:
        cache=args.datos_preparados
        manifest=json.loads((cache/'fuentes.json').read_text(encoding='utf-8'))
        if len(manifest)!=12:
            raise ValueError('Caché incompleta')
        for f in manifest:
            with (Path(args.fuente)/f['archivo']).open('rb') as stream:
                if hashlib.file_digest(stream,'sha256').hexdigest()!=f['sha256']:
                    raise ValueError('La fuente cambió: '+f['archivo'])
        raw=pd.read_parquet(cache/'muestra_cruda.parquet')
        events=pd.read_parquet(cache/'eventos_completos.parquet')
        raw.to_parquet(output/'muestra_cruda.parquet',index=False)
        prev.save_json(output/'fuentes.json',manifest)
    else:
        raw,events=read_data(args.fuente,output)
    train(raw,events,output,args)


if __name__=='__main__':
    main()
