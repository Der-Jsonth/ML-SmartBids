# Revisión de los datos originales de licitaciones 2024

Fecha: 4 de noviembre de 2026.

## Conclusión

Sí hay información descartada que merece recuperarse. Priorizaría **UnidadMedida y Rubro1/Rubro2/Rubro3**, la identidad e historia previa del proveedor, y las claves y fechas originales. Sin embargo, el hallazgo más relevante es la calidad de la etiqueta: los archivos mezclan estados de licitación y no permiten interpretar automáticamente cada «No Seleccionada» como una derrota definitiva.

La cobertura de cantidad parece prometedora en una primera prueba, pero su ventaja desaparece al restringir el análisis a adjudicaciones. No la presentaría como una mejora confirmada. Antes de ampliar el modelo, reconstruiría la unidad de observación y la disponibilidad temporal de las variables.

## Alcance y método

Interpreté «lic_2024» como los doce CSV `D:/Licit/lic_2024-01.csv` a `lic_2024-12.csv`: no se especificó un mes concreto. Leí completos los **2.846.181 registros y 105 columnas**, sin modificar los originales. Comparé sus campos con la preparación de V0.1 y los predictores de `V0.2/02_Comparacion_de_Modelos.ipynb`.

El perfil por columna y el recuento mensual abarcan todos los registros. Para asociaciones, fechas detalladas y modelos utilicé **43.740 filas**: 180 filas aleatorias por bloque de hasta 12.000, con semillas reproducibles. Los bloques finales de cada mes quedan sobrerrepresentados; se guardan pesos, pero las métricas exploratorias siguientes no los aplican. No son estimaciones definitivas del desempeño poblacional.

Los originales requieren `cp1252`; la lectura estricta falla por bytes indefinidos. Se contabilizan los caracteres de reemplazo por columna, sin eliminar automáticamente toda la fila. En el perfil, «vacíos» incluye cadenas vacías y marcadores `NA`, `N/A`, `NULL`, `NAN` y `NONE`; no incluye ceros ni `-1`. La cardinalidad de campos grandes está truncada y sus frecuencias son parciales, señaladas expresamente en el CSV.

## 1. La etiqueta mezcla resultados y estados pendientes

Recuentos completos por estado:

| Estado | Filas |
|---|---:|
| Adjudicada | 2.032.965 |
| Cerrada | 520.408 |
| Publicada | 157.255 |
| Desierta | 95.592 |
| Revocada | 39.856 |
| Suspendida | 105 |

Hay **358.724 filas seleccionadas, aproximadamente 12,60 %**, frente a 2.487.457 no seleccionadas. Esta proporción cambia mucho por mes: enero tiene 43.056/203.486 seleccionadas; octubre, 8.989/161.029.

En la muestra, las **2.524 filas con Estado=Publicada son No Seleccionada**. Las cerradas incluyen 7.986 negativas y 160 positivas. Incluso aparecen positivas en estados Revocada y Suspendida. Esto demuestra que estado y etiqueta no tienen una correspondencia simple; puede haber resultados pendientes, actualizaciones o inconsistencias. No demuestra por sí solo cuál es la causa.

**Acción:** conservar `Estado`, `CodigoEstado`, `FechaAdjudicacion` y fecha de extracción para auditar la madurez del resultado. Separar pendientes, cancelaciones y desiertas según el objetivo del negocio. Si se modela ganar entre procesos adjudicados, construir esa población explícitamente. El estado final sirve para seleccionar y auditar resultados; no debe entrar como predictor antes de conocerse.

## 2. Se perdieron las claves necesarias para evaluar el aprendizaje

Los originales contienen `Codigo`, `CodigoExterno`, `Codigoitem`, `CodigoProveedor`, `CodigoSucursalProveedor` y `Correlativo`. Encontré **145.375 códigos de licitación**, con una mediana de 3 filas y un máximo de **20.987 filas por licitación**. Un código aparece en más de un archivo y con más de una fecha de publicación: debe revisarse, no deduplicarse a ciegas.

La granularidad aparente es licitación–ítem–proveedor/oferta; hay que validar la unicidad de esa combinación y el significado de Correlativo antes de fijar la clave. Una fila no representa necesariamente una licitación independiente.

Tu V0.2 ya reconoce el problema y agrupa por huella de predictores. Recuperar el ID original permitiría una separación real por licitación: filas del mismo proceso pueden tener variables distintas y, por ello, huellas distintas.

**Acción:** conservar claves como metadatos y agrupar la partición por `Codigo`. Separar por fecha de publicación y comprobar también que los resultados de entrenamiento ya fueran conocidos al momento del corte. No usar los identificadores de licitación o ítem como números continuos del modelo.

## 3. Variables descartadas que probaría

| Prioridad | Campos o derivadas | Utilidad y condiciones |
|---|---|---|
| Alta | `UnidadMedida`, `Rubro1`, `Rubro2`, `Rubro3` | Hay 95 unidades y una jerarquía de rubros de 58/348/1.822 categorías. Facilitan compartir información entre productos poco frecuentes y distinguir cajas, unidades, comprimidos, etc. Parte de la jerarquía puede ser redundante con Código ONU: medir su aporte. |
| Alta, condicionada | `CodigoProveedor`; historia proveedor–comprador/producto | Permite representar experiencia previa y relaciones comerciales. Usar identidad como categoría y construir número de participaciones, adjudicaciones y tasas suavizadas solo con resultados anteriores al instante de predicción. Nunca calcular tasas con todo 2024 antes de dividir. Evaluar proveedores nuevos por separado. |
| Media | `MontoUnitarioOferta` | Diferencia precio y volumen; podría ser redundante con total/cantidad. Validar moneda, impuestos y si las magnitudes reconcilian antes de usarlo. |
| Media, con cautela | `Cantidad Ofertada / Cantidad` | Cobertura de lo solicitado, cuando ambas cantidades son positivas y comparables. La mediana y el percentil 99 de la muestra válida son 1. Su aparente ganancia no resistió la comprobación con adjudicadas. |
| Media | `FechaEnvioOferta` y anticipación al cierre | En la muestra hay 42.647 fechas válidas; la anticipación mediana es 1 día y p99 es 18. Disponible después de enviar la oferta; no para decidir antes de ofertar. Las fechas tienen resolución diaria, no permiten medir horas. |
| Media | `TipoPago`, `Modalidad`, duración/unidad del contrato, `SubContratacion`, `FuenteFinanciamiento` | Describen condiciones comerciales. Interpretar códigos, distinguir desconocidos y normalizar financiación; esta última tiene aproximadamente 47,5 % de ausencia en la muestra. La duración no debe combinarse sin su unidad. |
| Exploratoria | Descripción del producto/línea y `DescripcionProveedor` | Podrían representar ajuste técnico mediante texto. No se entrenó un modelo textual. Revisar contenido y disponibilidad; evitar textos actualizados después de resolver o que mencionen la adjudicación. |

`DuracionProceso` ya fue creada en V0.1 y V0.2 la excluye por disponibilidad: no es una variable descubierta aquí. El plazo publicación–cierre del archivo final puede contener modificaciones posteriores; recuperarlo no garantiza que sea seguro para una predicción histórica.

`EsRenovable` y `EsBaseTipo` son constantes: ambos valen 0 en los 2.846.181 registros. No aportan discriminación en este conjunto. `TipoConvocatoria` también varía muy poco: 2.842.891 filas con 1 y 3.290 con 0.

## 4. Comprobación exploratoria de las candidatas

Usé regresión logística con imputación, escalado y codificación categórica ajustados solo en entrenamiento. El corte es **1 de octubre de 2024 por fecha de publicación**, sin compartir códigos entre entrenamiento y prueba. Son 27.720 filas de entrenamiento y 16.020 de prueba; la prevalencia positiva de prueba es 0,06798.

La base es una aproximación a las familias de variables actuales usando el original: no reproduce el filtrado, conversión monetaria, modelos ni umbrales de V0.2. Estas métricas solo comparan variantes de esta auditoría.

| Variante | Average precision | ROC AUC |
|---|---:|---:|
| Base aproximada | 0,08388 | 0,64014 |
| + condiciones del contrato | 0,08529 | 0,64297 |
| + unidad de medida y rubros | 0,08846 | 0,64879 |
| + código del proveedor | 0,08910 | 0,66226 |
| + precio unitario | 0,08551 | 0,64538 |
| + cobertura de cantidad | 0,10140 | 0,60155 |
| + plazo publicación–cierre | 0,08437 | 0,64185 |
| + anticipación del envío | 0,08449 | 0,64215 |

Average precision resume la curva precisión–recall; **no es la precisión de las alertas a un umbral elegido**. La cobertura mejora esta métrica pero empeora ROC AUC. No hay una mejora uniforme.

Para comprobar el efecto de los estados pendientes, repetí parte del análisis restringiendo entrenamiento y prueba a `Estado=Adjudicada` y exigiendo adjudicación anterior al corte en entrenamiento. Quedaron 17.614/12.600 filas; prevalencia de prueba 0,08476.

| Variante con adjudicadas | Average precision | ROC AUC |
|---|---:|---:|
| Base | 0,08592 | 0,47206 |
| + unidad de medida y rubros | 0,09486 | 0,48092 |
| + proveedor | 0,08779 | 0,48140 |
| + cobertura de cantidad | 0,08592 | 0,47206 |

La cobertura pierde su ventaja. Producto/rubros mantiene una señal favorable en average precision, pero el ROC AUC inferior a 0,5 evidencia un problema de generalización en esta prueba. Esta segunda población también cambia el entrenamiento: no permite atribuir la diferencia únicamente al estado. La caída exige revisar cambios temporales, etiquetas y preparación antes de afirmar que el modelo sirve para licitaciones futuras.

Estas pruebas no tienen intervalos de confianza ni ajuste de hiperparámetros. Se probaron varias candidatas sobre el mismo periodo de prueba; este periodo queda como exploración y debe reservarse otro periodo intacto para confirmar decisiones. Tampoco existe una fotografía histórica completa que garantice que todos los campos ya tenían su valor final al corte.

## 5. Otros problemas detectados

**Moneda de la oferta.** Hay **678.843 filas con «Moneda revisar»** en `Moneda de la Oferta`, aproximadamente 23,85 %. La preparación inicial descarta esa columna y después aplica la moneda de adquisición al total ofertado. En los pares identificados de la muestra, las monedas coinciden; no encontré evidencia suficiente para afirmar una conversión errónea generalizada. Sí hay una ambigüedad que actualmente se pierde: no se puede asumir que una moneda desconocida sea CLP. Conservar ambas monedas y un indicador de incertidumbre; comprobar conversiones antes de usar relaciones de precios.

**Ausencia y selección de filas.** En los originales completos, 586.074 filas (20,59 %) carecen de cantidad ofertada y precio unitario; 1.339.958 (47,08 %) carecen de fuente de financiación y 3.617 de monto estimado. El Feather limpio tiene 2.248.494 filas, 597.687 menos que el original; esa diferencia es neta y no identifica por sí sola qué regla eliminó cada fila. V0.2 además exige cantidades y total de oferta positivos. El rendimiento obtenido en esa población no representa automáticamente todas las ofertas originales. Registrar exclusiones por motivo, estado, mes y etiqueta antes de comparar versiones.

**Fechas de adjudicación dudosas.** En la muestra hay fechas hasta **2027-07-22**, posteriores a la fecha de esta revisión; no deben asumirse como adjudicaciones efectivas verificadas. El máximo de cierre/envío es 2025-04-23, lo cual puede ser válido para una licitación publicada en 2024. Separar fechas previstas, fechas efectivas y valores inconsistentes antes de crear historia de proveedores.

**Etiquetas por licitación.** Hay 52.375 códigos sin ninguna fila seleccionada y 25.967 con varias filas seleccionadas. Las múltiples positivas pueden corresponder a diferentes ítems o adjudicación dividida: no son prueba automática de error. No imponer un único ganador por licitación sin estudiar la granularidad.

**Presupuesto y precio relativo.** `MontoEstimado` corresponde al proceso y no necesariamente a cada línea. Evitar interpretar total ofertado/presupuesto como descuento del ítem. El ranking de precios contra competidores sería otra candidata si se predice después de abrir todas las ofertas; para decidir antes del envío utilizaría solo referencias históricas disponibles.

**Fuga directa de resultado.** No añadir como predictores `CantidadAdjudicada`, `MontoLineaAdjudica`, `Monto Estimado Adjudicado`, el estado final ni la fecha efectiva de adjudicación. `Estado Oferta` debe excluirse para predicción antes de evaluación; podría utilizarse solo en una tarea definida explícitamente después de conocer admisibilidad. V0.2 ya excluye `NumeroOferentes` y `CantidadReclamos`; esa precaución sigue siendo pertinente.

## Recomendación de implementación

1. Reconstruir una tabla que conserve claves, estado, fechas, proveedor, ambas monedas y reglas de exclusión; mantener estos metadatos separados de los predictores.
2. Definir el instante de predicción: antes de ofertar, después del envío o después de apertura. De ello depende qué variables son utilizables.
3. Resolver madurez de etiquetas, fechas inconsistentes y granularidad; separar por tiempo y licitación.
4. Probar primero unidad de medida y rubros. Después, historia del proveedor construida sin futuro y precio unitario validado. Reexaminar cobertura solo con reglas de calidad explícitas.
5. Confirmar sobre otro periodo intacto, con precisión al recall mínimo que usa V0.2, calibración y resultados por sector, mes y proveedores conocidos/nuevos. No reemplazar el modelo actual con estas pruebas exploratorias.

## Archivos y reproducción

- `analizar_original.py`: lectura completa, perfiles, recuentos y muestreo.
- `evaluar_variables.py`: asociaciones, comprobaciones y comparación exploratoria.
- `perfil_columnas.csv`, `resumen.json`, `resumen_mensual.json`: evidencia de todos los originales.
- `comparacion_exploratoria.csv`, `hallazgos_medidos.json`, `asociaciones_muestra.csv`, `estado_vs_etiqueta_muestra.csv`, `estado_por_mes_muestra.csv`, `ausencias_muestra.csv`: evidencia exploratoria.
- `muestra_exploratoria.feather` y `claves_licitaciones.csv`: artefactos locales reproducibles, excluidos de Git.

Desde la raíz del proyecto, en PowerShell:

```powershell
& ./V0.2/.venv/Scripts/python.exe ./V0.2/revision_original_2024/analizar_original.py --fuente D:/Licit
& ./V0.2/.venv/Scripts/python.exe ./V0.2/revision_original_2024/evaluar_variables.py
```

Los notebooks y modelos existentes no fueron modificados. Las conclusiones se basan en los archivos locales; no se consultó una fuente externa para interpretar códigos administrativos.
