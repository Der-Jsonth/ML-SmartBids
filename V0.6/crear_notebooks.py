"""Genera los tres notebooks reproducibles de V0.6."""
from pathlib import Path
import nbformat as nbf

ROOT=Path(__file__).resolve().parent

def notebook(title,description,cells):
    nb=nbf.v4.new_notebook()
    nb["metadata"]={"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
                    "language_info":{"name":"python","version":"3.12"}}
    nb["cells"]=[nbf.v4.new_markdown_cell(f"# {title}\n\n{description}")]+cells
    return nb

common=nbf.v4.new_code_cell("from pathlib import Path\nimport sys\nROOT = Path.cwd()\nif ROOT.name != 'V0.6': ROOT = ROOT / 'V0.6'\nsys.path.insert(0, str(ROOT))")

nb1=notebook("1.0 Juntado de CSV",
"Lee los 12 CSV por bloques, reduce tipos y conserva cada registro. No deduplica ni une ofertas.",[
    common,
    nbf.v4.new_code_cell("import preparacion as prep\nmanifest = prep.juntar()\nmanifest"),
    nbf.v4.new_markdown_cell("La identidad física se compone de `_archivo` y `_registro_archivo`. Es única incluso cuando todos los campos de negocio coinciden."),
    nbf.v4.new_code_cell("import json\ninfo=json.loads((ROOT/'datos'/'fuentes.json').read_text(encoding='utf-8'))\n{'filas':info['filas'],'columnas':len(info['columnas']),'nota':info['nota']}")
])

nb2=notebook("2.0 Profiling y depuración",
"Construye la cohorte temporal sin borrar filas y audita qué campos distinguen línea, proveedor y oferta.",[
    common,
    nbf.v4.new_code_cell("import preparacion as prep\nprep.verificar_originales()\ndf = prep.preparar()\ndf.shape"),
    nbf.v4.new_code_cell("import pandas as pd\npd.read_csv(ROOT/'datos'/'comparacion'/'auditoria_identidad.csv')"),
    nbf.v4.new_markdown_cell("`Codigoitem` identifica la línea. Proveedor y sucursal separan la mayoría de las ofertas; `Nombre de la Oferta`, fecha, moneda, cantidad y montos distinguen otras. Las repeticiones restantes siguen siendo filas independientes."),
    nbf.v4.new_code_cell("pd.read_json(ROOT/'datos'/'comparacion'/'particiones.json').T")
])

nb3=notebook("3.0 Comparación de modelos",
"Compara la agrupación aproximada con `Codigoitem` real y dos modelos CatBoost. La elección se fija en validación antes de abrir test.",[
    common,
    nbf.v4.new_code_cell("import json\nimport pandas as pd\nimport modelo\nEJECUTAR_ENTRENAMIENTO = False\nif EJECUTAR_ENTRENAMIENTO:\n    salida = modelo.entrenar()\nelse:\n    ultima=json.loads((ROOT/'resultados'/'ultima_ejecucion.json').read_text(encoding='utf-8'))\n    salida=ROOT/'resultados'/ultima['directorio']\nsalida"),
    nbf.v4.new_code_cell("pd.read_csv(salida/'validacion.csv')"),
    nbf.v4.new_code_cell("pd.read_csv(salida/'test.csv')"),
    nbf.v4.new_code_cell("extra=ROOT/'resultados_ampliados'/'ultima_ejecucion.json'\nif extra.exists():\n    e=json.loads(extra.read_text(encoding='utf-8'))\n    display(pd.read_csv(ROOT/'resultados_ampliados'/e['directorio']/'validacion_ampliada.csv'))\n    display(pd.read_csv(ROOT/'resultados_ampliados'/e['directorio']/'test_ampliado.csv'))"),
    nbf.v4.new_code_cell("print((salida/'CONCLUSIONES.md').read_text(encoding='utf-8'))"),
    nbf.v4.new_markdown_cell("Para ejecutar las pruebas: `python test_v06.py`. Las etiquetas, estados finales y la identidad física están prohibidos como predictores.")
])

for name,nb in [("1.0 Juntado de CSV.ipynb",nb1),("2.0 Profiling y depuracion.ipynb",nb2),("3.0 Comparacion de modelos.ipynb",nb3)]:
    nbf.write(nb,ROOT/name)
    print(name)
