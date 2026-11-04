# V0.4 — prueba de historia y priorización de ofertas

Esta prueba implementa la recomendación posterior a V0.3: define una tarea de ofertas propias ya preparadas/enviadas, controla claves ambiguas, construye antecedentes desde los originales y evalúa utilidad de priorización. Mantiene V0.2 y V0.3 sin cambios.

## Diseño fijado antes de evaluar

- Se leen los doce CSV de 2024. No se encontraron CSV de 2025 en `D:/Licit`; por eso toda la evaluación sigue siendo retrospectiva sobre un periodo ya explorado.
- La historia se calcula desde los registros de **todos** los proveedores/procesos del original, no desde la muestra de entrenamiento. Se mantiene el muestreo del 25 % de licitaciones de V0.3 para las ofertas a modelar, sin balanceo 50/50.
- Se conserva el protocolo temporal y de madurez de V0.3: entrenamiento enero–mayo, calibración junio–julio, validación agosto–septiembre y test octubre–diciembre por publicación. Para desarrollo, el resultado debe existir antes del siguiente corte. Solo se modelan procesos finalmente adjudicados con fechas consistentes.
- El instante de consulta es `FechaEnvioOferta`. Se excluyen ofertas con envío inválido, anterior a publicación o posterior a adjudicación. También se excluyen claves incompletas y todas las filas de claves repetidas `Codigo/Codigoitem/CodigoProveedor/CodigoSucursalProveedor/Correlativo`. No se borran ofertas distintas como si fueran duplicados.
- Se comparan tres regresiones logísticas calibradas: **base con producto/proveedor**, base + **participaciones previas**, y base + participaciones/resultados previos **experimentales**. La base no es la base reducida de V0.3; todas las variantes de esta prueba comparten exactamente la misma población.
- La elección primaria es entre base y participaciones: máxima precisión en validación con recall ≥30 % y al menos 200 alertas. La variante de resultados proxy no puede ganar la selección primaria.

## Cómo se construye la historia

Se agregan eventos por licitación y proveedor, proveedor–unidad compradora y proveedor–producto ONU. Varios ítems o filas repetidas no multiplican el número de participaciones de una misma combinación. Una adjudicación significa que el proveedor ganó al menos un ítem de esa combinación; no se confunde con la proporción de filas ganadas.

Las participaciones cuentan solo envíos **estrictamente anteriores al día de consulta**, usando la última fecha de envío del evento para evitar contar la misma participación mientras todavía se registra. No se incorporan eventos del mismo día.

La variante experimental incorpora un resultado solo después de la fecha final de adjudicación **más 30 días**, siempre que el estado, etiqueta y fechas del evento sean válidos. Usa número de resultados, número de adjudicaciones y tasa suavizada `(adjudicaciones + 1) / (resultados + 5)`. Este prior fijo no se estima mirando el futuro. La demora no demuestra que la fecha final sea efectiva: esta variante es una sensibilidad, no una historia verificada.

En entrenamiento se consulta la historia anterior a cada envío. En calibración, validación y test se congela el conocimiento al **inicio de la etapa**. Ningún resultado de test actualiza las variables de otras filas del mismo test. Esto representa un historial congelado, más conservador que una actualización diaria con resultados verificados.

La historia de participaciones necesita registros propios o acceso efectivo a envíos anteriores. El original puede revelar datos que aún no fueran públicos para un tercero. Antes de aplicar esto a una empresa debe verificarse la disponibilidad real de la fuente.

## Ranking y límites

Además de precisión/recall y calibración, se calcula **top 10 y top 20 por proveedor y mes de envío**. Solo entran lotes con al menos k ofertas. Se informa cuántas seleccionadas aparecen, precisión, recall y mejora frente a la selección uniforme esperada dentro del mismo lote. No se eligen umbrales mirando este resultado.

El ranking agrupa ofertas retrospectivamente, incluidos envíos de distintos días. No demuestra que fueran oportunidades disponibles simultáneamente ni que el usuario representara a todos esos proveedores. Las recomendaciones son filas de oferta/ítem; no son necesariamente licitaciones distintas. Para una cartera real se debe definir empresa, fecha y unidad de decisión.

La población cambia frente a V0.3 por la exclusión de claves ambiguas y envíos inválidos: no se comparan directamente sus porcentajes. Las causas de exclusión pueden solaparse. La historia arranca en enero de 2024; no hay antecedentes de años previos y existe un arranque sin historia.

## Reproducción

Desde la raíz del proyecto, usando el entorno existente:

```powershell
& ./V0.2/.venv/Scripts/python.exe ./V0.4/probar_historia.py
& ./V0.2/.venv/Scripts/python.exe -m unittest discover -s ./V0.4 -p 'test_*.py' -v
```

Cada ejecución recibe una carpeta nueva en `resultados/`, con hashes de los originales y ambos módulos de preparación, parámetros, cohortes, historia consultada, validación, decisión, predicciones, ranking, intervalos por licitación y artefacto. Los datos y modelos generados se excluyen de Git. El código reutiliza la preparación, particiones, métricas y bootstrap de V0.3; debe conservarse esa carpeta para reproducirlo.

`predict(artefacto, datos_crudos, historia)` exige historia construida con el mismo protocolo e índices alineados. No debe rellenarse con resultados futuros. El ejemplo preparado permite verificar la carga del modelo; no representa una historia autónoma para despliegue.

Las ocho pruebas verifican exclusión de eventos futuros y del mismo día, demora de resultados, invariancia a filas repetidas, independencia de las participaciones respecto de etiquetas, congelamiento del test y agrupación de top-k. Los resultados concretos de esta ejecución se documentan en `CONCLUSIONES.md`. Ningún candidato se promueve automáticamente.

Si una extracción ya existe, `--datos-preparados resultados/ID_DE_EXTRACCION` permite utilizar sus archivos `muestra_cruda.parquet`, `eventos_completos.parquet` y `fuentes.json`, previa verificación SHA256 de los CSV. Las extracciones interrumpidas no se consideran entrenamientos completos y el notebook solo abre ejecuciones con estado completo.
