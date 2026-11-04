# Qué mejoró y qué sigue pendiente en V0.3

Revisión del 4 de noviembre de 2026. Se ejecutó el entrenamiento con los CSV originales de 2024 y el entorno existente de V0.2. La versión mejora la trazabilidad, la preparación y la evaluación. **No se confirmó una mejora de precisión del modelo enriquecido y no se reemplazó V0.2.**

## Trabajo incorporado

Se conservan las ideas de las versiones anteriores: códigos como categorías, transformación logarítmica, comparación de algoritmos, calibración separada y elección del umbral por precisión con cobertura mínima. Ahora se recuperan el ID real de licitación, fechas, moneda de cada oferta, unidad de medida, rubros y proveedor. Los resultados de adjudicación se usan para auditar la población, sin entrar en los predictores.

Se seleccionaron 691.105 filas mediante un hash del código de licitación, aproximadamente el 25 % de los procesos, conservando todas sus filas. Después se aplicaron las reglas de madurez y calidad por proceso. No se balancearon las clases 50/50 ni se inventaron ofertas. El ajuste efectivo usó 80.000 filas de 4.045 licitaciones completas; el límite y las exclusiones quedan registrados.

| Etapa | Filas elegibles | Licitaciones | Seleccionadas |
|---|---:|---:|---:|
| Entrenamiento, enero–mayo | 185.071 | 9.926 | 23,53 % |
| Calibración, junio–julio | 17.778 | 1.424 | 24,74 % |
| Validación, agosto–septiembre | 20.835 | 1.655 | 24,82 % |
| Prueba retrospectiva, octubre–diciembre | 178.143 | 4.959 | 9,26 % |

La partición de entrenamiento tiene más filas elegibles que las utilizadas por el límite de ajuste. Los porcentajes de esta tabla corresponden a la población elegible, no necesariamente al subconjunto de ajuste. Los procesos con resultados no disponibles al corte se excluyen; no se trasladan a otro conjunto.

## Resultado del aprendizaje

Se compararon siete candidatos: cuatro variantes acumuladas de regresión logística y árbol, bosque y MLP con las variables ampliadas. El ganador en validación fue **regresión logística con rubros, unidad de medida y proveedor**, con umbral 0,368052. Alcanzó aproximadamente 46,3 % de precisión y 31,8 % de recall en validación. La MLP no convergió dentro del presupuesto y no fue elegible.

El modelo y su umbral se fijaron antes de calcular sus métricas de prueba. La comparación con la base se hace sobre idéntica población, preparación monetaria y particiones, y no con las antiguas métricas de V0.2.

| Prueba retrospectiva | Precisión | Recall | Average precision | ROC AUC | Brier |
|---|---:|---:|---:|---:|---:|
| Candidato con proveedor y producto | 12,30 % | 35,07 % | 0,12399 | 0,58243 | 0,14326 |
| Base con mismo protocolo | 12,90 % | 30,31 % | 0,12016 | 0,59662 | 0,12291 |

El candidato mejora detección y ligeramente average precision, pero empeora precisión, ROC AUC y Brier. El intervalo bootstrap del 95 % por licitación para la diferencia de precisión candidato menos base es aproximadamente **−5,89 a +1,01 puntos porcentuales**: no respalda una mejora de precisión. El predictor constante de prevalencia de entrenamiento obtiene Brier 0,11005, mejor que ambos modelos en esta prueba; las probabilidades calibradas en meses anteriores tampoco se mantienen bien calibradas bajo este cambio de población.

## Lo que se nos había pasado

**1. El cambio temporal no era solo una cuestión de añadir variables.** La tasa de seleccionadas cae del 24,82 % en validación al 9,26 % en prueba. En los conjuntos de desarrollo exigimos resolver antes del siguiente corte, mientras que en prueba hay más tiempo para observar resultados. Esa selección por rapidez de resolución, la composición de procesos y posibles diferencias de actualización pueden contribuir al cambio. No se puede atribuirlo únicamente al comportamiento del mercado.

**2. Las licitaciones grandes dominan las métricas por fila.** Diez procesos concentran el **32,89 %** de las 178.143 filas de prueba. El mayor aporta 15.215 filas. Una muestra de filas podía ocultar esta concentración. Ahora se conserva la licitación completa y la incertidumbre se calcula remuestreando procesos; aún falta decidir si la utilidad del negocio debe ponderarse por oferta, ítem, licitación o monto.

**3. El promedio oculta meses y grupos muy distintos.** Con el umbral fijo del candidato, la precisión es aproximadamente 46,76 % en octubre, 16,49 % en noviembre y 7,92 % en diciembre. Estos valores son diagnósticos del mismo test, no nuevas oportunidades para elegir un umbral retrospectivo.

**4. Los datos fuera de la distribución de entrenamiento son un problema concreto.** En registros con proveedores presentes en el ajuste, la precisión del candidato es 13,92 %; con proveedores nuevos, 9,96 %. Las 15.917 filas con sector `NA` tienen precisión de apenas 0,44 % y el modelo marca aproximadamente el 96,52 % como positivas. No basta con imputar y aceptar cualquier predicción. Hay que estudiar la procedencia y el significado de esos registros y validar una política de abstención o tratamiento específico en nuevos datos; no eliminarlos después de mirar el test para mejorar una cifra.

**5. Las claves todavía requieren depuración semántica.** En la muestra de originales hay 15.692 filas cuya combinación candidata `Codigo/Codigoitem/CodigoProveedor/CodigoSucursalProveedor/Correlativo` se repite. No demuestra que sean duplicados: podrían ser ofertas/versiones distintas o diferencias en campos no leídos. Se registran sin borrar automáticamente. Esta granularidad puede alterar cuánto pesa cada proceso y cómo se define una oportunidad de ganar.

**6. Las reglas de madurez son proxies.** «Adjudicada» y la fecha incluida en el CSV no demuestran cuándo estuvo disponible el resultado ni que todos los campos conservan su valor al enviar la oferta. No se construyeron tasas históricas de éxito de proveedores con esas fechas sin verificar. Tampoco se añadió el estado final como predictor.

**7. El mismo histórico no vuelve a ser una prueba independiente.** Octubre–diciembre de 2024 ya se había observado durante la auditoría. Este entrenamiento no confirma desempeño futuro, aunque separe correctamente los conjuntos. Una nueva selección basada en los diagnósticos necesitará otra evaluación independiente.

## Siguiente decisión técnica

Antes de promover un modelo, verificar claves/versiones de oferta y fechas efectivas, obtener datos posteriores no explorados y reconstruir lo disponible al instante de predicción. Luego comparar historia previa del proveedor, manejo de categorías nuevas y ponderación por proceso mediante validaciones temporales dentro del desarrollo. Cualquier regla de abstención o recalibración debe fijarse antes de consultar el nuevo test.

No elegí retrospectivamente la base porque su precisión de prueba resultó mayor: eso también usaría el test para seleccionar. Ambos quedan como referencias exploratorias. El artefacto guardado es el ganador de validación y lleva `promovido=False`.

## Archivos de entrada

- `01_Revision_y_Resultados.ipynb`: resultados, curvas, grupos y ejemplo real de inferencia.
- `modelo_temporal.py`: preparación, entrenamiento, evaluación e inferencia sobre campos crudos.
- `test_modelo_temporal.py`: diez pruebas de exclusión de resultados finales, grupos, fechas, monedas, umbrales y paridad al guardar/cargar el modelo.
- `README.md`: protocolo, reproducción y límites.
- `resultados/<ejecucion>/`: manifiesto SHA256, parámetros, auditorías, artefacto, decisión, métricas, predicciones y estado de ejecución. Cada corrida recibe una carpeta nueva; estos artefactos locales se excluyen de Git.
