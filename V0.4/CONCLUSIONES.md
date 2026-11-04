# Resultado de probar la recomendación

4 de noviembre de 2026. Ejecución terminada: `20261104T164912115406Z`.

**La historia de participaciones previas aportó una mejora modesta y consistente con la hipótesis, dentro de esta prueba retrospectiva.** Es una dirección más prometedora que seguir agregando algoritmos. El resultado todavía no acredita rendimiento futuro ni permite presentar el score como una probabilidad confiable de ganar.

## Qué se probó realmente

La tarea fue priorizar ofertas propias ya enviadas, con proveedor, comprador, producto y precio conocidos, entre procesos finalmente adjudicados con fechas válidas. La base de esta prueba ya incorpora proveedor, unidad de medida y rubros; las variantes añaden historia. Se compara todo sobre la misma población y los mismos cortes.

Se leyó el original completo de 2024 para reconstruir 1.468.411 eventos licitación–proveedor–comprador–producto. Se retuvieron 691.105 filas por muestreo de licitaciones completas; tras controlar madurez, claves y fecha de envío, quedaron 391.457 filas elegibles entre las cuatro etapas. El ajuste efectivo se limitó a 80.000 filas de grupos completos.

En la muestra había 15.692 filas con clave candidata repetida, 5.040 con clave incompleta y 24.325 con fecha de envío inválida respecto de publicación/adjudicación. Estas causas se solapan y sus cifras incluyen procesos excluidos por otras reglas. No se suman para reconstruir una exclusión neta. Se excluyeron todas las filas de claves ambiguas, sin asumir que fueran duplicados: algunas tienen precios y resultados diferentes.

La historia primaria usa únicamente participaciones previas, sin etiquetas ganadoras. Cuenta una participación por proceso y proveedor, proceso y pareja proveedor–comprador, y proceso y pareja proveedor–producto. Se excluyen eventos del mismo día y futuros. En calibración, validación y test se congela al inicio de cada etapa.

La variante experimental añade resultados conocidos, según el proxy fecha final de adjudicación más 30 días. Esa demora no verifica que se trate de una fecha efectiva; la variante quedó expresamente excluida de la selección primaria.

## Resultado de clasificación

Se eligió modelo y umbral exclusivamente en validación para maximizar precisión con recall mínimo del 30 % y al menos 200 alertas. La variante primaria elegida fue **participaciones**, con umbral **0,391724**. La diferencia de precisión entre ambas candidatas en validación fue muy pequeña: 45,531 % frente a 45,541 %; no es una ventaja amplia o estable por sí sola.

La prueba contiene **169.829 filas de 4.930 licitaciones**, publicadas en octubre–diciembre de 2024, con una prevalencia positiva del 9,60 %.

| Variante | Precisión | Recall | Average precision | ROC AUC | Brier |
|---|---:|---:|---:|---:|---:|
| Base comparable | 14,04 % | 32,78 % | 0,13487 | 0,60065 | 0,13816 |
| + participaciones anteriores | **16,15 %** | **32,37 %** | **0,15980** | **0,65300** | **0,12992** |
| + resultados proxy, experimental | 17,65 % | 39,16 % | 0,17331 | 0,66119 | 0,13338 |

La historia primaria añade **2,11 puntos porcentuales de precisión**, alrededor del 15 % relativo, con una caída de recall de 0,42 puntos. Mejora también average precision, ROC AUC y Brier frente a la base de la misma prueba. Las probabilidades siguen mostrando problemas de transferencia temporal; mejor Brier relativo no equivale a calibración suficiente.

El bootstrap pareado por licitación estima un intervalo del 95 % para la diferencia de precisión de **+0,48 a +5,18 puntos porcentuales**. Es evidencia favorable dentro de esta muestra retrospectiva. Los intervalos individuales son amplios: precisión del candidato aproximadamente 9,57–25,39 %, y recall 29,32–35,15 %. El recall mínimo del 30 % no queda garantizado por todo el intervalo.

En términos de revisión de ofertas:

| Medida con umbral fijado en validación | Base | Participaciones |
|---|---:|---:|
| Ofertas marcadas | 38.066 | 32.667 |
| Seleccionadas correctamente detectadas | 5.344 | 5.276 |
| Falsas alertas | 32.722 | 27.391 |

La historia permitió **5.399 revisiones menos**, con 68 seleccionadas detectadas menos: reduce carga y falsas alertas, aunque no aumenta el total de aciertos al umbral elegido. Sigue habiendo muchas falsas alertas; 16,15 % de precisión es aproximadamente una seleccionada por cada seis ofertas marcadas, dentro de esta población.

## Resultado para priorizar una cartera

Se agruparon ofertas retrospectivamente por proveedor y mes de envío. Solo entraron lotes con al menos 10 o 20 ofertas, respectivamente. El ranking no usa un umbral ni utiliza la etiqueta para desempatar.

| Lote | Base | Participaciones | Resultado proxy experimental |
|---|---:|---:|---:|
| Precisión entre las primeras 10 | 16,91 % | **17,91 %** | 18,36 % |
| Seleccionadas entre las primeras 10 | 3.089 | **3.272** | 3.355 |
| Precisión entre las primeras 20 | 13,11 % | **13,92 %** | 14,24 % |
| Seleccionadas entre las primeras 20 | 3.331 | **3.536** | 3.618 |

Top 10 abarca 1.827 lotes y 18.270 recomendaciones: participaciones encuentra **183 seleccionadas adicionales** con la misma cantidad de revisiones. Top 20 abarca 1.270 lotes y 25.400 recomendaciones: encuentra **205 adicionales**. Estas cifras no son ganancias de una empresa particular ni un aumento de adjudicaciones causado por usar el modelo.

La selección uniforme esperada dentro de los mismos lotes obtiene 13,60 % de precisión para top 10 y 10,81 % para top 20. Participaciones alcanza un lift aproximado de **1,32 y 1,29**, respectivamente. Es una ayuda de priorización modesta, no un filtro definitivo de oportunidades.

Los lotes top 10 y top 20 no son la misma población, por su requisito de tamaño. El ranking agrupa retrospectivamente envíos de diferentes días y puede incluir varios ítems de una licitación; no demuestra que todas las oportunidades estuvieran disponibles simultáneamente.

## Qué concluyo y qué haría ahora

1. **Seguiría la línea de historia previa**, especialmente la relación proveedor–comprador/producto. Esta prueba respalda continuar esa investigación.
2. **Usaría el candidato para estudiar priorización**, conservando visibilidad de las ofertas descartadas. No lo usaría para excluir automáticamente una oportunidad ni como probabilidad financiera de ganar.
3. **No promovería todavía la variante de resultados proxy**, pese a sus mejores cifras. Primero verificaría fechas efectivas y cuándo se conoció cada resultado.
4. **Definiría una empresa y una unidad de decisión.** El modelo se entrenó con muchos proveedores; una cartera real requiere evaluar por empresa, fecha y oferta/ítem o licitación, según el negocio. La historia propia de participaciones debe existir en un registro verificable.
5. **Reservaría un nuevo periodo publicado en 2025 o posterior** y lo evaluaría una sola vez con estas reglas fijadas. En `D:/Licit` solo se encontraron los doce CSV de publicación 2024. Algunas ofertas de esas licitaciones se enviaron en 2025; no constituyen un conjunto independiente de licitaciones publicadas en 2025.

## Límites que siguen vigentes

- Todo 2024 ya fue explorado en trabajos anteriores; la separación correcta no convierte este test en confirmación independiente.
- La población actual excluye ofertas ambiguas o sin envío válido y difiere de V0.3. **No debe compararse el 16,15 % con cifras anteriores como si solo hubiera cambiado el modelo.** La comparación válida aquí es contra el 14,04 % de la misma tabla.
- La historia arranca en enero y no incluye años anteriores. Entrenamiento y prueba tienen distinta cantidad de antecedentes disponibles.
- El acceso a envíos previos de otros proveedores puede no haber existido en la fecha de predicción. La historia debe provenir de información propia o efectivamente disponible.
- Estado y fechas finales son proxies de madurez y pueden haber sido actualizados. La demora de 30 días no resuelve por sí sola ese problema.
- El requisito de resultado conocido al corte selecciona procesos de resolución rápida en desarrollo; la prueba permite más madurez. Faltan snapshots y un seguimiento comparable.
- El cambio temporal continúa: por mes de envío, la precisión de participaciones es aproximadamente 48,08 % en octubre, 22,26 % en noviembre y 15,92 % en diciembre de 2024; cae a 6,48 % en enero de 2025 dentro de procesos publicados en 2024. Un promedio no garantiza desempeño en cada mes.
- El bootstrap refleja incertidumbre entre licitaciones de la muestra, condicionada al modelo ya ajustado; no recoge toda la incertidumbre de entrenamiento ni confirma causalidad.

## Entregables y verificación

`01_Resultados_Historia.ipynb` muestra población, comparación, ranking, meses y paridad de inferencia. `probar_historia.py` reproduce la prueba y `test_historia.py` contiene ocho controles de temporalidad y cálculo. Las ocho pruebas pasaron. El artefacto queda con **`promovido=False`** y las versiones anteriores permanecen intactas.

Los parámetros, hashes, decisión previa al test, historia consultada y resultados completos están en `resultados/20261104T164912115406Z/`. El candidato conserva una mejora retrospectiva que merece validación adicional; no se declaró una mejora confirmada en datos futuros.
