# V0.3 — aprendizaje temporal desde los originales

Esta versión conserva el trabajo de V0.1/V0.2 y cambia la preparación y la evaluación a partir de la auditoría del original. Mantiene regresión logística, árbol, red neuronal y bosque; selecciona el candidato por **precisión con recall mínimo del 30 % y al menos 200 alertas en validación**, como el punto operativo intermedio de V0.2.

El objetivo es predecir si **una oferta propia ya enviada será seleccionada**, antes de conocer admisibilidad o adjudicación, entre procesos finalmente adjudicados con datos y fechas consistentes. No estima directamente la probabilidad de ganar cualquier licitación publicada ni sirve todavía como recomendador antes de enviar una oferta.

## Qué se nos había pasado

- La muestra exploratoria anterior tomaba filas, sin conservar todas las filas de una licitación. Aquí se selecciona por SHA256 del código, conservando grupos enteros y la prevalencia natural, sin balanceo artificial 50/50.
- Una huella de predictores no identifica una licitación: el mismo proceso podía quedar en varios conjuntos. Aquí los conjuntos son cronológicos y no comparten códigos.
- Una adjudicación posterior al corte no podía ser una etiqueta disponible para entrenar en ese momento. Ahora se exige disponibilidad anterior al corte de cada etapa. Estado y fecha final siguen siendo proxies; falta verificar cuándo se conoció realmente el resultado.
- «No Seleccionada» en procesos pendientes no equivale necesariamente a una derrota. Se excluyen procesos no adjudicados, fechas inválidas/inconsistentes y resultados posteriores a la fecha de referencia. La exclusión se aplica a la licitación completa y se audita, sin pretender que esta población represente todos los procesos.
- Habíamos consultado repetidamente el mismo test de 2024. Los nuevos resultados se declaran retrospectivos y exploratorios. Ningún cambio de partición vuelve independiente un histórico ya observado.
- La moneda de adquisición no identifica necesariamente la moneda de oferta. Se conservan ambas y se separan las escalas CLP/USD/UF/EUR. Los montos de moneda desconocida quedan sin valor numérico; no se asumen CLP.
- La cantidad y monto faltantes pueden seleccionar una población distinta. Se conservan esas filas, se imputa dentro del ajuste y se añaden indicadores de calidad.
- Una ganancia de average precision no garantiza más precisión al umbral operativo. Modelo y umbral se eligen en validación con la política explícita; el test no participa en la elección.
- La parada temprana aleatoria interna de MLP puede compartir grupos y mezclar fechas. Aquí no usa validación aleatoria interna y no se permite ganar a un candidato que emita advertencia de falta de convergencia.

## Datos y particiones

La ejecución predeterminada lee los **12 CSV completos** y retiene aproximadamente el 25 % de las licitaciones por hash reproducible. Esto no significa el 25 % exacto de las filas: los procesos tienen tamaños muy distintos. El máximo de ajuste es de 80.000 filas, seleccionando grupos completos, con el número real utilizado registrado. Calibración, validación y test no reciben ese límite adicional.

| Etapa | Fecha de publicación | Resultado conocido antes de |
|---|---|---|
| Entrenamiento | enero–mayo de 2024 | 1 de junio de 2024 |
| Calibración sigmoide | junio–julio de 2024 | 1 de agosto de 2024 |
| Selección de modelo y umbral | agosto–septiembre de 2024 | 1 de octubre de 2024 |
| Prueba retrospectiva | octubre–diciembre de 2024 | fecha de referencia, 4 de noviembre de 2026 inclusive |

El requisito sobre resultados introduce una selección de procesos resueltos con rapidez en los conjuntos de desarrollo; el test admite más madurez. Es una limitación que debe estudiarse con snapshots, otros cortes y horizontes de seguimiento comparables. No se trasladan automáticamente las métricas a una población con procesos pendientes.

## Variables y comparaciones

Se comparan cuatro familias con regresión logística sobre las mismas particiones: base aproximada, base + unidad de medida/rubros, anterior + proveedor, anterior + precio unitario/cobertura. Esta secuencia mide incrementos acumulados; no identifica por sí sola interacciones o efectos causales independientes. Árbol, bosque y MLP usan la última familia. La base aproxima las familias de V0.2 desde el CSV: no reproduce su conversión monetaria ni su población.

La información final —etiqueta, estado, fecha de adjudicación y montos adjudicados— queda fuera de los predictores mediante una lista explícita. Código de licitación y código de ítem quedan como metadatos. Proveedor y Código ONU se codifican como categorías, sin imponerles orden numérico. No se usa el presupuesto global como presupuesto de cada ítem.

Se excluyeron por ahora anticipación al cierre y condiciones contractuales: el archivo final podría incluir cambios posteriores al envío. **No se calcularon tasas históricas de éxito de proveedores**, porque antes hay que validar las fechas efectivas de resultado. La identidad del proveedor es una candidata; no sustituye una historia temporal verificada.

El test compara solo el ganador decidido en validación, la regresión logística base con el mismo protocolo y un predictor de prevalencia. Se guardan diagnósticos por mes, sector y proveedor conocido/nuevo, calibración e intervalos bootstrap de precisión/recall por licitación. Los diagnósticos no reajustan la selección. Se exige una nueva evaluación después de cualquier cambio inspirado por ellos.

## Uso

Se utiliza el entorno existente de V0.2; no hace falta instalar otro para esta ejecución. Desde la raíz, en PowerShell:

```powershell
& ./V0.2/.venv/Scripts/python.exe ./V0.3/modelo_temporal.py
& ./V0.2/.venv/Scripts/python.exe -m unittest discover -s ./V0.3 -p 'test_*.py' -v
```

Para entrenar con todos los grupos y un límite de ajuste diferente:

```powershell
& ./V0.2/.venv/Scripts/python.exe ./V0.3/modelo_temporal.py --fraccion 1 --max-train 250000
```

Para reutilizar la muestra existente, manteniendo fracción y fuentes, y volver a entrenar:

```powershell
& ./V0.2/.venv/Scripts/python.exe ./V0.3/modelo_temporal.py --reusar-datos
```

El comando verifica hashes de los CSV antes de reutilizar datos. Cada ejecución genera una carpeta nueva en `resultados/`; nunca sobrescribe el modelo V0.2. `fuentes.json` registra SHA256 de los originales; `configuracion.json` registra parámetros, código, versiones y cortes. Los datos/modelos generados se excluyen de Git; los scripts y la documentación se pueden versionar sin incorporar los CSV.

Abre `01_Revision_y_Resultados.ipynb` con el kernel de V0.2 para inspeccionar la última ejecución terminada sin reentrenar.

## Inferencia reproducible

```python
import sys, json, joblib, pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path('V0.3').resolve()))
from modelo_temporal import predict

# Sustituir por la ruta concreta de la ejecución que se quiere usar.
ejecucion = Path('V0.3/resultados/ID_DE_EJECUCION')
artefacto = joblib.load(ejecucion / 'modelo_candidato.joblib')
ofertas = pd.read_json(ejecucion / 'ejemplo_entrada.json')
salida = predict(artefacto, ofertas)
```

`predict` aplica exactamente la misma preparación de campos crudos que el entrenamiento y valida el esquema. Las probabilidades son del candidato exploratorio, no una garantía de adjudicación. Para evaluar otra política de recall, volver a seleccionar en validación y reservar otra evaluación independiente.

## Qué falta para declarar una mejora

El modelo queda como **candidato exploratorio**, incluso si supera a la base. Hace falta verificar la fecha efectiva de resultados y los valores disponibles al envío, evaluar procesos nuevos de un periodo no explorado, estudiar la selección por madurez y confirmar la política operativa con el mismo objetivo. Si el recall cae por debajo del 30 % en test, se informa: el requisito se elige en validación y no se garantiza bajo cambio temporal.

`resultados/<ejecucion>/RESULTADO.md` contiene las cifras de la ejecución; `informe.json` incluye sus limitaciones. `RESUMEN_EJECUCION.md` conserva un resumen revisable junto al código.
