# Comparación ampliada de modelos

Se probaron cinco familias sobre la misma cohorte, cortes temporales y definición de línea. La selección se realizó por F1 de validación antes de puntuar los modelos nuevos en test.

| Modelo | Familia | F1 validación | F1 test | AP test |
|---|---|---:|---:|---:|
| `sgd_logistico` | Lineal SGD | 0,3542 | 0,2552 | 0,3244 |
| `hgb_proxy` | HistGradientBoosting | 0,4049 | 0,3539 | 0,4181 |
| `hgb_item` | HistGradientBoosting | 0,4198 | 0,3750 | 0,4252 |
| `lightgbm_item` | LightGBM | 0,4214 | 0,3681 | 0,4262 |
| `xgboost_item` | XGBoost | 0,4200 | 0,3197 | 0,4131 |
| `cb_item` | CatBoost | **0,4289** | **0,4044** | 0,4466 |
| `cb_completo` | CatBoost | 0,4252 | 0,3913 | **0,4485** |

## Decisión

`cb_item` continúa siendo el modelo elegido. Logró el mayor F1 de validación y también el mayor F1 exploratorio de test. `cb_completo` ordena ligeramente mejor los casos según AP, pero el umbral elegido en validación produce menor F1.

LightGBM queda como segunda alternativa entre los modelos basados solo en las variables comparables de ítem. XGBoost mostró una caída temporal mayor, principalmente por menor recall en noviembre-diciembre. El modelo lineal confirma que las relaciones relevantes no son suficientemente lineales.

El test de 2024 ya fue observado en versiones anteriores, por lo que estos resultados son exploratorios. Para una conclusión confirmatoria se necesita otro periodo que no se haya utilizado durante el desarrollo.
