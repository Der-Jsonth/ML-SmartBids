# Conclusiones V0.6

Se conservaron **2,846,181 filas originales**: filas eliminadas **0** y filas unidas **0**. Cada observación mantiene `_archivo` y `_registro_archivo`. La clave de negocio sirve para auditar repeticiones, nunca para colapsarlas.

El candidato elegido antes de abrir test fue **cb_item** (F1 de validación **0.4289**). En test logró **0.4044**, frente a **0.3539** del baseline interno: diferencia **+5.05 puntos**. El IC95 bootstrap por licitación es **[+3.87, +6.55] puntos**.

## Resultados de test

| Candidato | Precisión | Recall | F1 | AP | ROC AUC |
|---|---:|---:|---:|---:|---:|
| hgb_proxy | 0.5010 | 0.2736 | 0.3539 | 0.4181 | 0.9099 |
| hgb_item | 0.4886 | 0.3043 | 0.3750 | 0.4252 | 0.9118 |
| cb_item | 0.5231 | 0.3296 | 0.4044 | 0.4466 | 0.9013 |
| cb_completo | 0.5601 | 0.3007 | 0.3913 | 0.4485 | 0.8932 |

## Interpretación

- `hgb_proxy` reproduce la agrupación aproximada producto-cantidad; `hgb_item` usa el `Codigoitem` real.
- `cb_completo` añade proveedor, sucursal, rubros, condiciones comerciales e historia previa. `Nombre de la Oferta` se usa para contar propuestas distintas dentro de la línea, no como identificador de alta cardinalidad.
- Estado de la oferta, adjudicación, selección y la identidad física están prohibidos como predictores.
- Los resultados de V0.5 no son directamente comparables: aquella versión había eliminado identificadores y deduplicado filas. Aquí todos los modelos comparten exactamente la nueva cohorte original y los mismos cortes.
- El benchmark usa precios de las otras ofertas de una línea; corresponde a análisis después de la apertura, no a una predicción previa a presentar oferta.

## Comparación ampliada

También se probaron un modelo lineal SGD, LightGBM y XGBoost. Sus F1 de test fueron **0.2552**, **0.3681** y **0.3197**, respectivamente. Ninguno superó a `cb_item` (**0.4044**). La tabla completa está en `docs/COMPARACION_MODELOS.md`.
