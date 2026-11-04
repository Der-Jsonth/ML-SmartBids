"""Exploración de candidatas con separación cronológica y por licitación.
No modifica los modelos existentes ni acredita mejoras definitivas.
"""
from pathlib import Path
import json
import warnings
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

OUT=Path(__file__).resolve().parent

def num(s):
    return pd.to_numeric(s.str.replace(',', '.',regex=False),errors='coerce')

def main():
    d=pd.read_feather(OUT/'muestra_exploratoria.feather')
    profile=pd.read_csv(OUT/'perfil_columnas.csv')
    valid=d['Oferta seleccionada'].isin(['Seleccionada','No Seleccionada'])
    d=d.loc[valid].copy()
    y=d['Oferta seleccionada'].eq('Seleccionada').astype(int)
    for c in ['MontoEstimado','Valor Total Ofertado','Cantidad','Cantidad Ofertada','MontoUnitarioOferta']:
        d[c+'_num']=num(d[c])
        d[c+'_log']=np.log1p(d[c+'_num'].clip(lower=0))
    dates={c:pd.to_datetime(d[c],format='mixed',dayfirst=False,errors='coerce') for c in
           ['FechaPublicacion','FechaCierre','FechaEnvioOferta','FechaAdjudicacion']}
    d['Mes']=dates['FechaPublicacion'].dt.month
    d['DiaSemana']=dates['FechaPublicacion'].dt.dayofweek
    d['Plazo_publicacion_cierre']=(dates['FechaCierre']-dates['FechaPublicacion']).dt.total_seconds()/86400
    d['Anticipacion_envio']=(dates['FechaCierre']-dates['FechaEnvioOferta']).dt.total_seconds()/86400
    d['Cobertura_cantidad']=d['Cantidad Ofertada_num']/d['Cantidad_num'].replace(0,np.nan)
    numeric=['MontoEstimado_log','Valor Total Ofertado_log','Cantidad_log','Cantidad Ofertada_log','Mes','DiaSemana']
    categorical=['CodigoUnidad','CodigoProductoONU','sector','Tipo','RegionUnidad']
    base=numeric+categorical
    contract=['TipoConvocatoria','Etapas','Contrato','Obras','FuenteFinanciamiento','VisibilidadMonto',
              'Modalidad','TipoPago','SubContratacion','UnidadTiempoDuracionContrato',
              'TiempoDuracionContrato','TipoDuracionContrato','EsRenovable','EsBaseTipo']
    variants={'base_aproximada':[], 'base_mas_condiciones':contract,
              'base_mas_producto':['UnidadMedida','Rubro1','Rubro2','Rubro3'],
              'base_mas_proveedor':['CodigoProveedor'],
              'base_mas_derivadas':['MontoUnitarioOferta_log','Cobertura_cantidad','Plazo_publicacion_cierre','Anticipacion_envio'],
              'base_mas_precio_unitario':['MontoUnitarioOferta_log'],
              'base_mas_cobertura':['Cobertura_cantidad'],
              'base_mas_plazo':['Plazo_publicacion_cierre'],
              'base_mas_anticipacion':['Anticipacion_envio'],
              'adjudicadas_base':[],
              'adjudicadas_producto':['UnidadMedida','Rubro1','Rubro2','Rubro3'],
              'adjudicadas_proveedor':['CodigoProveedor'],
              'adjudicadas_cobertura':['Cobertura_cantidad']}
    cutoff=pd.Timestamp('2024-10-01')
    keys=pd.read_csv(OUT/'claves_licitaciones.csv',dtype={'Codigo':str,'CodigoExterno':str},keep_default_na=False)
    keys['fecha']=pd.to_datetime(keys.FechaPublicacion,format='mixed',errors='coerce')
    # Licitaciones presentes a ambos lados se excluyen del entrenamiento.
    test_ids=set(d.loc[dates['FechaPublicacion']>=cutoff,'Codigo'])
    train=(dates['FechaPublicacion']<cutoff)&~d.Codigo.isin(test_ids)
    test=dates['FechaPublicacion']>=cutoff
    scores=[]
    for name,extra in variants.items():
        tr=train.copy()
        te=test.copy()
        if name.startswith('adjudicadas_'):
            tr &= d.Estado.eq('Adjudicada') & dates['FechaAdjudicacion'].lt(cutoff)
            te &= d.Estado.eq('Adjudicada')
        cols=base+extra
        nc=[c for c in cols if c in numeric or c.endswith('_log') or c in ['Cobertura_cantidad','Plazo_publicacion_cierre','Anticipacion_envio']]
        cc=[c for c in cols if c not in nc]
        X=d[cols].copy()
        X[nc]=X[nc].replace([np.inf,-np.inf],np.nan)
        pre=ColumnTransformer([
            ('num',make_pipeline(SimpleImputer(strategy='median'),StandardScaler()),nc),
            ('cat',OneHotEncoder(handle_unknown='ignore',min_frequency=15),cc)])
        model=make_pipeline(pre,LogisticRegression(max_iter=500,solver='liblinear',random_state=42))
        model.fit(X.loc[tr],y.loc[tr])
        p=model.predict_proba(X.loc[te])[:,1]
        scores.append({'variante':name,'train':int(tr.sum()),'test':int(te.sum()),
                       'prevalencia_test':float(y.loc[te].mean()),
                       'average_precision':float(average_precision_score(y.loc[te],p)),
                       'roc_auc':float(roc_auc_score(y.loc[te],p))})
        print(scores[-1],flush=True)
    pd.DataFrame(scores).to_csv(OUT/'comparacion_exploratoria.csv',index=False)
    associations=[]
    for c in contract+['UnidadMedida','Rubro1','Moneda de la Oferta','Estado Oferta','CantidadAdjudicada','MontoLineaAdjudica']:
        t=pd.DataFrame({'valor':d[c],'y':y}).groupby('valor').y.agg(['size','mean'])
        t=t[t['size']>=100].sort_values('size',ascending=False).head(15).reset_index()
        t.insert(0,'columna',c)
        associations.append(t)
    pd.concat(associations).to_csv(OUT/'asociaciones_muestra.csv',index=False,encoding='utf-8-sig')
    pd.crosstab(d['Estado'],d['Oferta seleccionada']).to_csv(OUT/'estado_vs_etiqueta_muestra.csv',encoding='utf-8-sig')
    pd.crosstab(d['_archivo'],d['Estado']).to_csv(OUT/'estado_por_mes_muestra.csv',encoding='utf-8-sig')
    pd.DataFrame({'columna':d.columns,'porcentaje_ausencia_muestra':[
        100*d[c].astype(str).str.strip().str.upper().isin(['','NA','N/A','NULL','NAN','NONE']).mean()
        for c in d]}).to_csv(OUT/'ausencias_muestra.csv',index=False,encoding='utf-8-sig')
    overlap=keys.groupby('Codigo').archivo.nunique()
    def finite_summary(s):
        s=s.replace([np.inf,-np.inf],np.nan)
        return {'validos':int(s.notna().sum()),'negativos':int(s.lt(0).sum()),
                'mediana':float(s.median()),'p99':float(s.quantile(.99))}
    money=(d['Moneda Adquisicion'].str.strip()+' | '+d['Moneda de la Oferta'].str.strip()).value_counts().head(25)
    facts={'muestra_filas':len(d),'train':int(train.sum()),'test':int(test.sum()),
        'fechas_muestra':{c:{'validos':int(v.notna().sum()),'min':str(v.min()),'max':str(v.max())} for c,v in dates.items()},
        'derivadas_muestra':{c:finite_summary(d[c]) for c in ['Plazo_publicacion_cierre','Anticipacion_envio','Cobertura_cantidad']},
        'pares_monedas_muestra':money.to_dict(),
        'licitaciones_en_mas_de_un_archivo':int(overlap.gt(1).sum()),
        'licitaciones_unicas':int(keys.Codigo.nunique()),
        'filas_por_licitacion':keys.groupby('Codigo').filas.sum().describe().to_dict(),
        'codigos_con_fechas_publicacion_multiples':int(keys.groupby('Codigo').FechaPublicacion.nunique().gt(1).sum()),
        'licitaciones_sin_oferta_seleccionada':int(keys.groupby('Codigo').positivas.sum().eq(0).sum()),
        'licitaciones_con_multiples_filas_seleccionadas':int(keys.groupby('Codigo').positivas.sum().gt(1).sum()),
        'monto_estimado_cero_muestra':int(d.MontoEstimado_num.eq(0).sum()),
        'precio_no_positivo_muestra':int(d['Valor Total Ofertado_num'].le(0).sum()),
        'fecha_publicacion_fuera_2024_muestra':int((dates['FechaPublicacion'].notna()&dates['FechaPublicacion'].dt.year.ne(2024)).sum()),
        'advertencia':'Estadísticas con sufijo muestra y modelos usan muestreo por bloques sin ponderar; perfiles y claves abarcan todos los CSV. Comparación exploratoria sin ajuste de hiperparámetros ni validación anidada.'}
    (OUT/'hallazgos_medidos.json').write_text(json.dumps(facts,ensure_ascii=False,indent=2),encoding='utf-8')

if __name__=='__main__':
    main()
