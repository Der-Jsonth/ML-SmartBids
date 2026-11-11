# V0.6 — ofertas originales sin fusionar

Esta versión conserva los **2.846.181 registros** de los doce CSV de 2024. No elimina duplicados ni une filas ambiguas.

## Estructura

1. `1.0 Juntado de CSV.ipynb`: lectura por bloques, tipos compactos, Parquet y hashes.
2. `2.0 Profiling y depuracion.ipynb`: auditoría de identidad, cortes temporales e historia previa.
3. `3.0 Comparacion de modelos.ipynb`: HGB con clave aproximada, HGB con `Codigoitem`, CatBoost con ítem y CatBoost completo.

El código reutilizable está en `preparacion.py` y `modelo.py`. Las pruebas están en `test_v06.py`.

## Identidad de una observación

- Línea: `Codigo` + `Codigoitem`.
- Oferta identificable: línea + `CodigoProveedor` + `CodigoSucursalProveedor`.
- Diferenciadores adicionales: `Nombre de la Oferta`, `FechaEnvioOferta`, moneda, cantidad y montos.
- Identidad definitiva: `_archivo` + `_registro_archivo`.

La clave de negocio solo se audita. Aunque dos registros coincidan en todos sus campos, permanecen separados. La identidad física no se entrega al modelo.

## Resultados actuales

| Modelo | Familia | F1 validación | F1 test |
|---|---|---:|---:|
| `sgd_logistico` | Lineal | 0,3542 | 0,2552 |
| `hgb_proxy` | HistGradientBoosting | 0,4049 | 0,3539 |
| `hgb_item` | HistGradientBoosting | 0,4198 | 0,3750 |
| `lightgbm_item` | LightGBM | 0,4214 | 0,3681 |
| `xgboost_item` | XGBoost | 0,4200 | 0,3197 |
| `cb_item` | CatBoost | **0,4289** | **0,4044** |
| `cb_completo` | CatBoost | 0,4252 | 0,3913 |

`cb_item` fue elegido únicamente con validación. El modelo completo tiene mayor AP, pero menor F1; por eso no se promovió. La comparación ampliada se ejecuta con `comparar_modelos_extra.py` y queda en `resultados_ampliados/`.

Los valores de V0.5 no son comparables directamente: esa versión trabajaba con una población reducida que ya no contenía los identificadores de oferta.

## Reproducción

Desde la raíz del repositorio y usando el entorno existente:

```powershell
& './V0.2/.venv/Scripts/python.exe' 'V0.6/preparacion.py' juntar
& './V0.2/.venv/Scripts/python.exe' 'V0.6/preparacion.py' preparar
& './V0.2/.venv/Scripts/python.exe' 'V0.6/test_v06.py'
& './V0.2/.venv/Scripts/python.exe' 'V0.6/modelo.py' --entrenar
```

La última ejecución completa está referenciada en `resultados/ultima_ejecucion.json`.
