"""Amplía el benchmark con modelos lineal, LightGBM y XGBoost.

Reutiliza la cohorte y las predicciones ya calculadas. Todos los modelos nuevos
se seleccionan por validación antes de generar sus predicciones de test.
"""
from __future__ import annotations
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import time
import joblib
import lightgbm
from lightgbm import LGBMClassifier
import numpy as np
import pandas as pd
import xgboost
from xgboost import XGBClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import modelo as base
import preparacion as prep

ROOT=Path(__file__).resolve().parent

def variables_lineales(d):
    x=base.variables(d,next(s for s in base.ESPECIFICACIONES if s["nombre"]=="hgb_item"))
    excluir={"CodigoProductoONU"}
    x=x[[c for c in x if c not in excluir]].copy()
    for c in ["MontoEstimado","Cantidad","Cantidad Ofertada","Valor Total Ofertado","MontoUnitarioOferta","precio_unitario"]:
        x[c]=np.sign(x[c])*np.log1p(np.abs(x[c]))
    return x.astype("float32")

def variables_arbol(d):
    """Misma matriz de hgb_item, con nombres portables entre bibliotecas."""
    x=base.variables(d,next(s for s in base.ESPECIFICACIONES if s["nombre"]=="hgb_item"))
    x.columns=[f"f{i:03d}" for i in range(x.shape[1])]
    return x

def modelos(cfg):
    return {
        "sgd_logistico": Pipeline([
            ("imputar",SimpleImputer(strategy="median",add_indicator=True)),
            ("escalar",StandardScaler()),
            ("modelo",SGDClassifier(loss="log_loss",alpha=1e-5,max_iter=100,tol=1e-3,
                                     average=True,random_state=cfg["semilla"],n_jobs=cfg["hilos"])),
        ]),
        "lightgbm_item": LGBMClassifier(
            n_estimators=500,learning_rate=.04,num_leaves=31,max_depth=-1,
            min_child_samples=50,subsample=.9,colsample_bytree=.9,reg_lambda=2,
            random_state=cfg["semilla"],n_jobs=cfg["hilos"],verbosity=-1,
        ),
        "xgboost_item": XGBClassifier(
            n_estimators=400,learning_rate=.05,max_depth=7,min_child_weight=5,
            subsample=.9,colsample_bytree=.9,reg_lambda=2,objective="binary:logistic",
            eval_metric="logloss",tree_method="hist",random_state=cfg["semilla"],
            n_jobs=cfg["hilos"],verbosity=1,
        ),
    }

def ejecutar():
    cfg=prep.configuracion()
    latest=json.loads((ROOT/"resultados"/"ultima_ejecucion.json").read_text(encoding="utf-8"))["directorio"]
    previous=ROOT/"resultados"/latest
    if json.loads((previous/"estado.json").read_text(encoding="utf-8"))["estado"]!="completo":
        raise ValueError("La comparación base no está completa.")
    d=pd.read_parquet(prep.DATOS/"cohorte.parquet")
    ids={n:d.index[d._particion.eq(n)] for n in cfg["cortes"]}
    train,val,test=[ids[n] for n in ["train","validacion","test"]]
    stamp=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    out=ROOT/"resultados_ampliados"/stamp; out.mkdir(parents=True)
    specs=modelos(cfg); validation=[]; artifacts={}
    prep.guardar_json(out/"configuracion.json",{
        "version":prep.VERSION,"configuracion":cfg,
        "cohorte_sha256":prep.sha256(prep.DATOS/"cohorte.parquet"),
        "codigo_sha256":prep.sha256(Path(__file__)),
        "modelos":list(specs),
        "versiones":{"lightgbm":lightgbm.__version__,"xgboost":xgboost.__version__},
    })
    pv=pd.read_parquet(previous/"predicciones_validacion.parquet")
    old_val=pd.read_csv(previous/"validacion.csv")

    for name,model in specs.items():
        started=time.perf_counter()
        x=variables_lineales(d) if name=="sgd_logistico" else variables_arbol(d)
        print(f"Ajustando {name}: {len(train):,} filas y {x.shape[1]} variables",flush=True)
        model.fit(x.loc[train],d.loc[train,"Seleccionada"])
        prob=model.predict_proba(x.loc[val])[:,1]
        threshold=base.umbral_f1(d.loc[val,"Seleccionada"],prob)
        record={"candidato":name,"segundos":time.perf_counter()-started,
                **base.metricas(d.loc[val,"Seleccionada"],prob,threshold)}
        validation.append(record); pv[name]=prob
        artifact={"modelo":model,"columnas":list(x),"umbral":threshold,
                  "tipo_variables":"lineales" if name=="sgd_logistico" else "hgb_item",
                  "version":prep.VERSION}
        joblib.dump(artifact,out/f"{name}.joblib",compress=3); artifacts[name]=artifact
        print(f"VALIDACIÓN {name}: F1={record['f1']:.4f}; AP={record['average_precision']:.4f}",flush=True)
        del x,prob; gc.collect()

    new_val=pd.DataFrame(validation)
    combined_val=pd.concat([old_val,new_val],ignore_index=True,sort=False)
    chosen=combined_val.sort_values(["f1","average_precision"],ascending=False,kind="stable").iloc[0].candidato
    decision={"elegido":chosen,"criterio":"Mayor F1 de validación; desempate AP",
              "comparacion_base":str(previous),"test_reutilizado":True,
              "nota_test":"2024 ya fue consultado en versiones previas; resultado exploratorio, no confirmatorio.",
              "versiones":{"lightgbm":lightgbm.__version__,"xgboost":xgboost.__version__}}
    prep.guardar_json(out/"seleccion_previa_test.json",decision)
    combined_val.drop(columns="matriz_confusion",errors="ignore").to_csv(out/"validacion_ampliada.csv",index=False,encoding="utf-8-sig")
    pv.to_parquet(out/"predicciones_validacion.parquet",index=False)
    print("Selección fijada antes de puntuar modelos nuevos en test:",chosen,flush=True)

    old_test=pd.read_csv(previous/"test.csv")
    frame=pd.read_parquet(previous/"predicciones_test.parquet")
    test_rows=[]
    for name,artifact in artifacts.items():
        x=variables_lineales(d) if artifact["tipo_variables"]=="lineales" else variables_arbol(d)
        prob=artifact["modelo"].predict_proba(x.loc[test])[:,1]
        frame[name]=prob
        test_rows.append({"candidato":name,**base.metricas(frame.y,prob,artifact["umbral"])})
        print(f"TEST {name}: F1={test_rows[-1]['f1']:.4f}",flush=True)
        del x,prob; gc.collect()
    combined_test=pd.concat([old_test,pd.DataFrame(test_rows)],ignore_index=True,sort=False)
    combined_test.drop(columns="matriz_confusion",errors="ignore").to_csv(out/"test_ampliado.csv",index=False,encoding="utf-8-sig")
    frame.to_parquet(out/"predicciones_test.parquet",index=False)
    prep.guardar_json(out/"estado.json",{"estado":"completo","elegido":chosen,"version":prep.VERSION})
    prep.guardar_json(ROOT/"resultados_ampliados"/"ultima_ejecucion.json",{"directorio":out.name})
    print("Comparación ampliada completa:",out,flush=True)
    return out

if __name__=="__main__":
    ejecutar()
