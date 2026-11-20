# ML-SmartBids

**Experimentación de aprendizaje automático aplicada a las licitaciones públicas de Chile.**

Este repositorio reúne las distintas iteraciones de investigación y desarrollo de modelos de *machine learning* para **SmartBids**, una plataforma orientada al análisis de oportunidades de negocio en el mercado de compras públicas.

El propósito de esta línea de trabajo es **estimar la probabilidad de que una oferta sea seleccionada para un ítem de una licitación**, utilizando características disponibles en los datos históricos. Los experimentos permiten evaluar si los modelos aportan información útil antes de considerar su integración en la plataforma.

> **Estado:** proyecto experimental. Las predicciones no representan una garantía de adjudicación y no deben interpretarse como un modelo validado para producción.

## Objetivo

Abordar un problema de **clasificación binaria** en el que la variable objetivo indica si una oferta fue seleccionada:

| Valor | Interpretación |
|:---:|---|
| `1` | Oferta seleccionada |
| `0` | Oferta no seleccionada |

La unidad de análisis es la **oferta asociada a un ítem**, no la licitación completa. Por tanto, la salida buscada es una estimación de probabilidad para una oferta concreta, no una predicción de qué licitaciones serán publicadas ni una recomendación automática de precios.

## Estructura del repositorio

Las carpetas conservan el historial de iteraciones de los experimentos:

```text
ML-SmartBids/
├── V0.1/
├── V0.2/
├── V0.3/
├── V0.4/
├── V0.5 (MM)/
├── V0.6/
├── Columnas y Datos - Hoja 1.pdf
└── README.md
```

- **`V0.1` a `V0.6`:** versiones sucesivas del trabajo experimental. Cada una debe consultarse por separado, ya que pueden variar los datos, las transformaciones y los modelos utilizados.
- **`Columnas y Datos - Hoja 1.pdf`:** documento de referencia para comprender columnas y datos del proyecto.

**Alcance de esta documentación:** la rama `main` consultada contiene versiones hasta `V0.6`. Los trabajos posteriores no se presentan aquí como si ya estuvieran publicados en el repositorio.

## Metodología general

El proceso de experimentación abarca, según la versión:

1. **Exploración de datos:** revisión de columnas, tipos, valores faltantes, distribuciones y calidad de registros.
2. **Preparación:** limpieza, selección de variables y transformaciones de datos numéricos y categóricos.
3. **Definición del objetivo:** identificación de ofertas seleccionadas (`1`) y no seleccionadas (`0`).
4. **Separación de datos:** construcción de conjuntos para entrenamiento y evaluación, prestando atención a posibles fugas de información entre ofertas de una misma licitación y a la dimensión temporal.
5. **Entrenamiento:** evaluación de enfoques de clasificación supervisada y comparación con referencias sencillas (*baselines*).
6. **Evaluación:** análisis de discriminación, precisión de las probabilidades y comportamiento del modelo sobre datos no utilizados para entrenarlo.

### Variables de interés

En el proyecto se han explorado variables relacionadas con las **cantidades solicitadas y ofertadas**, los **montos de oferta**, el **producto o grupo de producto**, el **organismo comprador**, el **tipo de licitación** y la **unidad de medida**, entre otras.

Las columnas exactas y las transformaciones empleadas **dependen de cada versión**. Es importante revisar el código de la iteración correspondiente antes de reproducir un resultado.

## Evaluación de modelos

No basta con medir el porcentaje global de aciertos: también interesa identificar correctamente los casos positivos (`seleccionada = 1`) y evaluar la calidad de las probabilidades estimadas.

| Métrica | Qué permite analizar |
|---|---|
| **Average Precision (AP)** | Calidad del ordenamiento de ofertas positivas a lo largo de la curva precisión-recall. |
| **ROC-AUC** | Capacidad de distinguir entre ofertas seleccionadas y no seleccionadas. |
| **Brier score** | Error cuadrático de las probabilidades predichas; **menor es mejor**. |
| **Precision / Recall** | Comportamiento al aplicar un umbral para clasificar ofertas como positivas. |

Los resultados de una versión **no son directamente comparables** con los de otra si cambia el dataset, la definición de la etiqueta, la partición o el procedimiento de evaluación. Deben revisarse junto al experimento que los generó.

## Cómo explorar el proyecto

### 1. Clonar el repositorio

```bash
git clone https://github.com/Der-Jsonth/ML-SmartBids.git
cd ML-SmartBids
```

### 2. Crear un entorno virtual

```bash
python -m venv .venv
```

En **Windows (PowerShell)**:

```powershell
.\.venv\Scripts\Activate.ps1
```

En **Linux o macOS**:

```bash
source .venv/bin/activate
```

### 3. Abrir una iteración

Consulta la carpeta de la versión que te interese. Si contiene notebooks de Jupyter, instala JupyterLab y ejecútalos desde la raíz del repositorio:

```bash
python -m pip install jupyterlab
jupyter lab
```

Instala además las bibliotecas importadas por los archivos de esa versión. **No se proporciona aquí un comando único de instalación de dependencias**, porque los requisitos pueden cambiar entre iteraciones y no se ha verificado un archivo de dependencias unificado.

> **Datos necesarios:** algunos experimentos pueden requerir archivos de origen o rutas locales. Comprueba los requisitos del notebook o script antes de ejecutarlo; clonar el repositorio no garantiza disponer de todos los datasets utilizados originalmente.

## Limitaciones y consideraciones

- **Carácter experimental:** las versiones documentan pruebas e iteraciones, no un servicio de inferencia desplegado.
- **Dependencia de los datos:** la calidad y la representatividad de los registros condicionan el rendimiento.
- **Fuga de información:** ofertas de una misma licitación pueden estar relacionadas; una separación inadecuada produce estimaciones optimistas.
- **Cambio temporal:** el comportamiento de las adjudicaciones puede variar entre períodos.
- **Interpretación:** una probabilidad estimada es una señal estadística, no una adjudicación asegurada ni una sustitución de las bases y criterios de evaluación de cada licitación.

## Relación con SmartBids

Este repositorio corresponde a la **línea de investigación de modelos predictivos** de SmartBids. No contiene necesariamente el código completo de la plataforma web, sus bases de datos operacionales ni sus procesos de despliegue.

## Estado del proyecto

Se mantiene un historial de experimentos para comparar enfoques y documentar decisiones metodológicas. **La incorporación de un modelo a producción depende de que demuestre un rendimiento y una calibración suficientes sobre datos de evaluación adecuados.**

---

*Proyecto desarrollado en el contexto de SmartBids y el análisis de compras públicas chilenas.*
