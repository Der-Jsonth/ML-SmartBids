"""Busca en los CSV originales atributos adicionales para ofertas con IDs repetidos."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


BASE = Path(__file__).resolve().parent
ORIGEN = Path(r"D:\Licit")
AUDITORIA = BASE / "docs" / "auditoria_identidad"
CLAVE = ["Codigo", "Codigoitem", "Correlativo", "CodigoProveedor", "CodigoSucursalProveedor"]
EXTRAS = [
    "RutProveedor",
    "NombreProveedor",
    "RazonSocialProveedor",
    "Nombre de la Oferta",
    "FechaEnvioOferta",
    "Cantidad Ofertada",
    "Moneda de la Oferta",
    "MontoUnitarioOferta",
    "Valor Total Ofertado",
    "Estado Oferta",
    "Oferta seleccionada",
]


def limpio(s: pd.Series) -> pd.Series:
    return s.astype("string").str.strip().replace("", pd.NA)


def clave_texto(s: pd.Series) -> pd.Series:
    return limpio(s).str.replace(r"\.0$", "", regex=True)


def main() -> None:
    grupos = pd.read_parquet(AUDITORIA / "grupos_repetidos.parquet", columns=CLAVE)
    for c in CLAVE:
        grupos[c] = clave_texto(grupos[c])
    grupos = grupos.drop_duplicates()

    hallados: list[pd.DataFrame] = []
    for archivo in sorted(ORIGEN.glob("lic_2024-*.csv")):
        inicio = 0
        for trozo in pd.read_csv(
            archivo,
            sep=";",
            encoding="latin1",
            usecols=CLAVE + EXTRAS,
            dtype="string",
            chunksize=150_000,
            low_memory=False,
        ):
            trozo["_registro_archivo"] = np.arange(inicio + 1, inicio + len(trozo) + 1, dtype="uint32")
            for c in CLAVE:
                trozo[c] = clave_texto(trozo[c])
            coincidencias = trozo.merge(grupos, on=CLAVE, how="inner", validate="many_to_one")
            if len(coincidencias):
                coincidencias["_archivo"] = archivo.name
                hallados.append(coincidencias)
            inicio += len(trozo)
        print(f"{archivo.name}: revisado")

    detalle = pd.concat(hallados, ignore_index=True)
    for c in EXTRAS:
        detalle[c] = limpio(detalle[c]).str.lower()
    detalle.to_parquet(AUDITORIA / "detalle_repetidos_original.parquet", index=False)

    g = detalle.groupby(CLAVE, observed=True, dropna=False)
    resumen = g.agg(
        filas=("Codigo", "size"),
        ruts=("RutProveedor", "nunique"),
        nombres_proveedor=("NombreProveedor", "nunique"),
        razones_sociales=("RazonSocialProveedor", "nunique"),
        nombres_oferta=("Nombre de la Oferta", "nunique"),
        fechas=("FechaEnvioOferta", "nunique"),
        cantidades=("Cantidad Ofertada", "nunique"),
        monedas=("Moneda de la Oferta", "nunique"),
        montos_unitarios=("MontoUnitarioOferta", "nunique"),
        totales=("Valor Total Ofertado", "nunique"),
        estados=("Estado Oferta", "nunique"),
        objetivos=("Oferta seleccionada", "nunique"),
    ).reset_index()
    resumen.to_parquet(AUDITORIA / "diferenciadores_csv.parquet", index=False)
    cols = ["ruts", "nombres_proveedor", "razones_sociales", "nombres_oferta", "fechas", "cantidades", "monedas", "montos_unitarios", "totales"]
    print(f"Filas recuperadas: {len(detalle):,}")
    print(f"Grupos: {len(resumen):,}")
    print("Columnas que distinguen grupos:")
    print(resumen[cols].gt(1).sum().to_string())
    print(f"Sin diferencia en esos campos: {resumen[cols].le(1).all(axis=1).sum():,}")


if __name__ == "__main__":
    main()
