import pandas as pd
import glob
import os
import numpy as np

# Carpeta donde están los CSV mensuales
ruta = "C:/Users/hunte/Downloads/Licit/*.csv"

archivos = sorted(glob.glob(ruta))

print(f"Archivos encontrados: {len(archivos)}")

if len(archivos) == 0:
    raise FileNotFoundError(
        "No se encontraron archivos CSV en datos_licitaciones/"
    )

dataframes = []

for archivo in archivos:

    nombre = os.path.basename(archivo)
    print(f"\nLeyendo: {nombre}")

    df_mes = pd.read_csv(
        archivo,
        sep=";",
        encoding="cp1252",
        encoding_errors="replace",
        low_memory=False
    )

    # Detectar únicamente columnas de texto
    columnas_texto = df_mes.select_dtypes(
        include=["object", "string"]
    ).columns

    # Si una celda contiene el carácter de reemplazo �,
    # convertir TODA esa celda a NaN
    for columna in columnas_texto:

        mascara = (
            df_mes[columna]
            .astype("string")
            .str.contains("�", regex=False, na=False)
        )

        cantidad = mascara.sum()

        if cantidad > 0:
            print(
                f"  ⚠ {columna}: "
                f"{cantidad:,} valores con problemas -> NaN"
            )

            df_mes.loc[mascara, columna] = np.nan

    print(f"  ✓ Filas: {len(df_mes):,}")
    print(f"  ✓ Columnas: {len(df_mes.columns)}")

    dataframes.append(df_mes)


# ============================================================
# UNIR TODOS LOS MESES
# ============================================================

df = pd.concat(
    dataframes,
    ignore_index=True
)

print("\n" + "=" * 60)
print("DATASET ANUAL")
print("=" * 60)

print(f"Archivos unidos: {len(dataframes)}")
print(f"Filas totales: {len(df):,}")
print(f"Columnas: {len(df.columns)}")
print(f"Dimensiones: {df.shape}")


# ============================================================
# GUARDAR
# ============================================================

df.to_feather("licitaciones_anual.feather")

print("\n✓ Guardado como:")
print("licitaciones_anual.feather")