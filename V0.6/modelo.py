"""V0.6: comparación temporal sobre ofertas originales no fusionadas."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import platform
import time
import catboost
from catboost import CatBoostClassifier
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (precision_recall_curve, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, brier_score_loss, confusion_matrix)
from threadpoolctl import threadpool_limits
import preparacion as prep

ROOT=Path(__file__).resolve().parent
ESPECIFICACIONES=[
    {"nombre":"hgb_proxy","algoritmo":"hgb","linea":"proxy","completo":False},
    {"nombre":"hgb_item","algoritmo":"hgb","linea":"item","completo":False},
    {"nombre":"cb_item","algoritmo":"catboost","linea":"item","completo":False},
    {"nombre":"cb_completo","algoritmo":"catboost","linea":"item","completo":True},
]

def division_segura(a,b):
    return (a/b.where(b>0)).replace([np.inf,-np.inf],np.nan)

def categoria(s):
    return prep.limpiar(s).mask(lambda x:x.str.contains("\ufffd",regex=False,na=False)).fillna("sin_dato").astype("category")

def clave_linea(d, modo):
    if modo=="item":
        return ("I:"+d.Codigoitem.astype("string")).fillna("sin_item").astype("category")
    return ("P:"+d.CodigoProductoONU.astype("string")+":"+d.Cantidad.astype("string")).astype("category")

def variables(d,spec):
    """Construye predictores previos al resultado; la identidad física nunca es predictor."""
    numeric=["MontoEstimado","NumeroOferentes","Cantidad","Cantidad Ofertada",
             "Valor Total Ofertado","MontoUnitarioOferta","CodigoProductoONU",
             "DiaSemanaPublicacion","MesPublicacion"]
    x=d[numeric].copy()
    qty=d["Cantidad Ofertada"]
    total=d["Valor Total Ofertado"].where(d["Valor Total Ofertado"]>0)
    unit=d.MontoUnitarioOferta.where(d.MontoUnitarioOferta>0, total/qty.where(qty>0))
    x["precio_unitario"]=unit
    x["fraccion_cantidad"]=division_segura(qty,d.Cantidad)
    x["oferta_sobre_presupuesto"]=division_segura(total,d.MontoEstimado)
    line=clave_linea(d,spec["linea"])
    context=pd.DataFrame({"codigo":d.CodigoExterno,"linea":line,"precio":unit,
                          "proveedor":d.CodigoProveedor},index=d.index)
    g=context.groupby(["codigo","linea"],observed=True,sort=False)
    x["precio_sobre_minimo"]=division_segura(unit,g.precio.transform("min"))
    x["precio_sobre_mediana"]=division_segura(unit,g.precio.transform("median"))
    x["puesto_relativo_precio"]=g.precio.rank(method="min",pct=True)
    x["ofertas_linea"]=g.precio.transform("size").astype("uint16")
    process=d.groupby("CodigoExterno",observed=True,sort=False)
    x["lineas_proceso"]=process.Codigoitem.transform("nunique").astype("uint16")
    x["filas_proceso"]=process.Codigoitem.transform("size").astype("uint32")
    x["ofertas_por_linea"]=division_segura(x.filas_proceso,x.lineas_proceso)

    if spec["algoritmo"]=="hgb":
        low=pd.get_dummies(d[["sector","Tipo","RegionUnidad"]].astype("string").fillna("sin_dato"),
                           prefix=["sector","Tipo","Region"],dtype="uint8")
        x=x.join(low)
    else:
        for c in ["CodigoProductoONU","CodigoUnidad","sector","Tipo","RegionUnidad"]:
            x[c]=categoria(d[c])

    if spec["completo"]:
        for c in ["CodigoProveedor","CodigoSucursalProveedor","UnidadMedida","Rubro1","Rubro2","Rubro3",
                  "TipoPago","Modalidad","SubContratacion","FuenteFinanciamiento",
                  "UnidadTiempoDuracionContrato","TipoDuracionContrato","Moneda Adquisicion","Moneda de la Oferta"]:
            x[c]=categoria(d[c])
        for name in prep.PARES:
            col=f"hist_{name}_participaciones_log"; x[col]=d[col].astype("float32")
        x["duracion_contrato_log"]=np.log1p(d.TiempoDuracionContrato.where(d.TiempoDuracionContrato>=0)).astype("float32")
        x["proveedores_linea"]=g.proveedor.transform("nunique").astype("uint16")
        x["filas_proveedor_linea"]=context.groupby(
            ["codigo","linea","proveedor"],observed=True,sort=False).precio.transform("size").fillna(0).astype("uint16")
        names=prep.limpiar(d["Nombre de la Oferta"]).fillna("sin_nombre")
        named=context[["codigo","linea"]].assign(nombre=names)
        x["nombres_oferta_linea"]=named.groupby(
            ["codigo","linea"],observed=True,sort=False).nombre.transform("nunique").astype("uint16")

    prohibited={"Seleccionada","Oferta seleccionada","Estado","Estado Oferta","FechaAdjudicacion",
                "CodigoExterno","Codigoitem","_id_fila","_archivo","_registro_archivo","CantidadAdjudicada"}
    if prohibited & set(x): raise AssertionError("Entró una etiqueta, resultado o identificador físico al modelo.")
    return x.replace([np.inf,-np.inf],np.nan)

def umbral_f1(y,p):
    precision,recall,cuts=precision_recall_curve(y,p)
    score=2*precision[:-1]*recall[:-1]/np.maximum(precision[:-1]+recall[:-1],1e-12)
    return float(cuts[np.argmax(score)])

def metricas(y,p,t):
    y,p=np.asarray(y),np.asarray(p); pred=p>=t
    return {"filas":len(y),"prevalencia":float(y.mean()),"umbral":float(t),
            "precision":float(precision_score(y,pred,zero_division=0)),
            "recall":float(recall_score(y,pred,zero_division=0)),"f1":float(f1_score(y,pred,zero_division=0)),
            "average_precision":float(average_precision_score(y,p)),
            "roc_auc":float(roc_auc_score(y,p)),"brier":float(brier_score_loss(y,p)),
            "accuracy":float((y==pred).mean()),"alertas":int(pred.sum()),
            "matriz_confusion":confusion_matrix(y,pred,labels=[0,1]).tolist()}

def puntuar(artifact,x):
    if list(x)!=artifact["columnas"]: raise ValueError("Cambió el contrato de variables.")
    p=artifact["modelo"].predict_proba(x)[:,1]
    return pd.DataFrame({"score":p,"seleccion_predicha":(p>=artifact["umbral"]).astype("uint8")},index=x.index)

def bootstrap(frame,selected,reference,thresholds,repetitions,seed):
    arrays=[]
    for name in [selected,reference]:
        pred=frame[name]>=thresholds[name]
        z=pd.DataFrame({"codigo":frame.CodigoExterno,"tp":pred&frame.y.eq(1),
                        "fp":pred&frame.y.eq(0),"fn":~pred&frame.y.eq(1)})
        arrays.append(z.groupby("codigo",observed=True)[["tp","fp","fn"]].sum().sort_index().to_numpy(float))
    rng=np.random.default_rng(seed); diffs=[]
    for _ in range(repetitions):
        idx=rng.integers(0,len(arrays[0]),len(arrays[0])); values=[]
        for a in arrays:
            tp,fp,fn=a[idx].sum(axis=0); values.append(2*tp/max(2*tp+fp+fn,1))
        diffs.append(values[0]-values[1])
    low,high=np.quantile(diffs,[.025,.975])
    return {"seleccionado":selected,"referencia":reference,"repeticiones":repetitions,
            "delta_f1_ic95":[float(low),float(high)],"nota":"Bootstrap pareado por licitación."}

def entrenar(cfg=None):
    cfg=cfg or prep.configuracion(); provenance=json.loads((prep.DATOS/"preparacion.json").read_text(encoding="utf-8"))
    if cfg!=provenance["configuracion"] or provenance["codigo_preparacion_sha256"]!=prep.sha256(Path(prep.__file__)) or provenance["cohorte_sha256"]!=prep.sha256(prep.DATOS/"cohorte.parquet"):
        raise ValueError("Cambió la preparación; ejecute nuevamente el notebook 2.")
    d=pd.read_parquet(prep.DATOS/"cohorte.parquet")
    ids={name:d.index[d._particion.eq(name)] for name in cfg["cortes"]}; train,val,test=[ids[n] for n in ["train","validacion","test"]]
    run=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"); out=ROOT/"resultados"/run; out.mkdir(parents=True)
    prep.guardar_json(out/"estado.json",{"estado":"ajustando","version":prep.VERSION})
    prep.guardar_json(out/"configuracion.json",{"version":prep.VERSION,"configuracion":cfg,"datos":provenance,
        "codigo_modelo_sha256":prep.sha256(Path(__file__)),"especificaciones":ESPECIFICACIONES,
        "filas_train":len(train),"filas_validacion":len(val),"filas_test":len(test),
        "versiones":{"python":platform.python_version(),"pandas":pd.__version__,"numpy":np.__version__,
                     "sklearn":sklearn.__version__,"catboost":catboost.__version__},
        "criterio":"Mayor F1 de validación; test permanece cerrado hasta fijar el candidato."})
    validation=[]; pv=pd.DataFrame({"_id_fila":d.loc[val,"_id_fila"],"CodigoExterno":d.loc[val,"CodigoExterno"],"y":d.loc[val,"Seleccionada"]},index=val)
    for spec in ESPECIFICACIONES:
        name=spec["nombre"]; started=time.perf_counter(); x=variables(d,spec)
        print(f"Ajustando {name}: {len(train):,} filas; {len(x.columns)} variables.",flush=True)
        with threadpool_limits(limits=cfg["hilos"]):
            if spec["algoritmo"]=="hgb":
                model=HistGradientBoostingClassifier(max_iter=cfg["iteraciones_hgb"],early_stopping=False,random_state=cfg["semilla"])
                model.fit(x.loc[train],d.loc[train,"Seleccionada"])
            else:
                model=CatBoostClassifier(iterations=cfg["iteraciones_catboost"],depth=cfg["profundidad_catboost"],
                    learning_rate=.06,l2_leaf_reg=5,loss_function="Logloss",random_seed=cfg["semilla"],
                    thread_count=cfg["hilos"],max_ctr_complexity=1,one_hot_max_size=12,
                    allow_writing_files=False,verbose=100)
                model.fit(x.loc[train],d.loc[train,"Seleccionada"],cat_features=x.select_dtypes(include="category").columns.tolist())
            p=model.predict_proba(x.loc[val])[:,1]
        threshold=umbral_f1(d.loc[val,"Seleccionada"],p); record={"candidato":name,"variables":len(x.columns),
            "segundos":time.perf_counter()-started,**metricas(d.loc[val,"Seleccionada"],p,threshold)}
        validation.append(record); pv[name]=p
        joblib.dump({"version":prep.VERSION,"especificacion":spec,"modelo":model,"columnas":list(x),
                     "umbral":threshold,"promovido":False},out/f"{name}.joblib",compress=3)
        print(f"VALIDACIÓN {name}: F1={record['f1']:.4f}; AP={record['average_precision']:.4f}",flush=True)
        del x,model; gc.collect()
    valid=pd.DataFrame(validation); valid.drop(columns="matriz_confusion").to_csv(out/"validacion.csv",index=False,encoding="utf-8-sig")
    prep.guardar_json(out/"validacion.json",validation); pv.to_parquet(out/"predicciones_validacion.parquet",index=False)
    selected=valid.sort_values(["f1","average_precision"],ascending=False,kind="stable").iloc[0].candidato
    decision={"elegido":selected,"referencia":"hgb_proxy","umbrales":dict(zip(valid.candidato,valid.umbral)),
              "criterio":"F1 de validación","cohorte_sha256":provenance["cohorte_sha256"],"promovido":False}
    prep.guardar_json(out/"seleccion_previa_test.json",decision); prep.guardar_json(out/"estado.json",{"estado":"seleccion_fijada","version":prep.VERSION})
    frame=d.loc[test,["_id_fila","_archivo","_registro_archivo","CodigoExterno","MesPublicacion","Seleccionada","Estado","CodigoProveedor"]].rename(columns={"Seleccionada":"y"}).copy()
    frame["proveedor_identificado"]=frame.CodigoProveedor.notna(); results=[]
    for spec in ESPECIFICACIONES:
        name=spec["nombre"]; artifact=joblib.load(out/f"{name}.joblib"); x=variables(d,spec); got=puntuar(artifact,x.loc[test]); p=got.score.to_numpy()
        frame[name]=p; results.append({"candidato":name,**metricas(frame.y,p,artifact["umbral"])})
        if name==selected:
            joblib.dump(artifact,out/"modelo_elegido.joblib",compress=3)
            if spec["algoritmo"]=="catboost":
                pd.DataFrame({"variable":list(x),"importancia":artifact["modelo"].feature_importances_}).sort_values("importancia",ascending=False).to_csv(out/"importancias.csv",index=False,encoding="utf-8-sig")
        np.testing.assert_allclose(puntuar(joblib.load(out/f"{name}.joblib"),x.loc[test[:100]]).score,p[:100],rtol=1e-7,atol=1e-8)
        print(f"TEST {name}: F1={results[-1]['f1']:.4f}",flush=True); del x,artifact; gc.collect()
    prep.guardar_json(out/"test.json",results); pd.DataFrame(results).drop(columns="matriz_confusion").to_csv(out/"test.csv",index=False,encoding="utf-8-sig")
    frame.to_parquet(out/"predicciones_test.parquet",index=False); finalizar(out); return out

def finalizar(out):
    out=Path(out); decision=json.loads((out/"seleccion_previa_test.json").read_text(encoding="utf-8")); cfg=json.loads((out/"configuracion.json").read_text(encoding="utf-8"))["configuracion"]
    frame=pd.read_parquet(out/"predicciones_test.parquet"); selected,reference=decision["elegido"],decision["referencia"]
    rows=[]
    for dimension in ["MesPublicacion","Estado","proveedor_identificado"]:
        for value,part in frame.groupby(frame[dimension].astype("string").fillna("sin_dato"),sort=True):
            for name in dict.fromkeys([selected,reference]): rows.append({"dimension":dimension,"valor":value,"candidato":name,**metricas(part.y,part[name],decision["umbrales"][name])})
    pd.DataFrame(rows).drop(columns="matriz_confusion").to_csv(out/"segmentos.csv",index=False,encoding="utf-8-sig")
    ci=bootstrap(frame,selected,reference,decision["umbrales"],cfg["bootstrap_repeticiones"],cfg["semilla"]); prep.guardar_json(out/"incertidumbre.json",ci)
    escribir_conclusiones(out); prep.guardar_json(out/"estado.json",{"estado":"completo","elegido":selected,"version":prep.VERSION,"paridad_serializacion":True,"promovido":False})
    prep.guardar_json(ROOT/"resultados"/"ultima_ejecucion.json",{"directorio":out.name}); print("Ejecución completa:",out,flush=True)

def escribir_conclusiones(out):
    table=pd.read_csv(out/"test.csv").set_index("candidato"); val=pd.read_csv(out/"validacion.csv").set_index("candidato")
    decision=json.loads((out/"seleccion_previa_test.json").read_text(encoding="utf-8")); ci=json.loads((out/"incertidumbre.json").read_text(encoding="utf-8")); audit=json.loads((prep.DATOS/"auditoria_cohorte.json").read_text(encoding="utf-8"))
    winner,base=decision["elegido"],decision["referencia"]; delta=table.loc[winner,"f1"]-table.loc[base,"f1"]
    lines=["| Candidato | Precisión | Recall | F1 | AP | ROC AUC |","|---|---:|---:|---:|---:|---:|"]
    for n,r in table.iterrows(): lines.append(f"| {n} | {r.precision:.4f} | {r.recall:.4f} | {r.f1:.4f} | {r.average_precision:.4f} | {r.roc_auc:.4f} |")
    text=f"""# Conclusiones V0.6

Se conservaron **{audit['filas_cohorte']:,} filas originales**: filas eliminadas **0** y filas unidas **0**. Cada observación mantiene `_archivo` y `_registro_archivo`. La clave de negocio sirve para auditar repeticiones, nunca para colapsarlas.

El candidato elegido antes de abrir test fue **{winner}** (F1 de validación **{val.loc[winner,'f1']:.4f}**). En test logró **{table.loc[winner,'f1']:.4f}**, frente a **{table.loc[base,'f1']:.4f}** del baseline interno: diferencia **{100*delta:+.2f} puntos**. El IC95 bootstrap por licitación es **[{100*ci['delta_f1_ic95'][0]:+.2f}, {100*ci['delta_f1_ic95'][1]:+.2f}] puntos**.

## Resultados de test

{chr(10).join(lines)}

## Interpretación

- `hgb_proxy` reproduce la agrupación aproximada producto-cantidad; `hgb_item` usa el `Codigoitem` real.
- `cb_completo` añade proveedor, sucursal, rubros, condiciones comerciales e historia previa. `Nombre de la Oferta` se usa para contar propuestas distintas dentro de la línea, no como identificador de alta cardinalidad.
- Estado de la oferta, adjudicación, selección y la identidad física están prohibidos como predictores.
- Los resultados de V0.5 no son directamente comparables: aquella versión había eliminado identificadores y deduplicado filas. Aquí todos los modelos comparten exactamente la nueva cohorte original y los mismos cortes.
- El benchmark usa precios de las otras ofertas de una línea; corresponde a análisis después de la apertura, no a una predicción previa a presentar oferta.
"""
    (ROOT/"CONCLUSIONES.md").write_text(text,encoding="utf-8"); (out/"CONCLUSIONES.md").write_text(text,encoding="utf-8")

if __name__=="__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("--entrenar",action="store_true"); parser.add_argument("--finalizar",type=Path); args=parser.parse_args()
    if args.entrenar: entrenar()
    elif args.finalizar: finalizar(args.finalizar)
