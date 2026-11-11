"""Audita qué columnas distinguen las ofertas sin fusionar registros."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


BASE = Path(__file__).resolve().parent
PARQUETS = BASE / "datos" / "juntado"
SALIDA = BASE / "docs" / "auditoria_identidad"

IDS = [
    "Codigo",
    "CodigoExterno",
    "Codigoitem",
    "Correlativo",
    "CodigoProveedor",
    "CodigoSucursalProveedor",
]
DETALLE = [
    "CodigoProductoONU",
    "FechaEnvioOferta",
    "Cantidad Ofertada",
    "Moneda de la Oferta",
    "MontoUnitarioOferta",
    "Valor Total Ofertado",
]
OBJETIVO = "Oferta seleccionada"


def normalizar_texto(s: pd.Series) -> pd.Series:
    return s.astype("string").str.strip().replace("", pd.NA)


def estadistica_clave(df: pd.DataFrame, nombre: str, columnas: list[str]) -> dict:
    validos = df[columnas].notna().all(axis=1)
    tamanos = df.loc[validos].groupby(columnas, observed=True, dropna=False).size()
    repetidos = tamanos[tamanos.gt(1)]
    return {
        "clave": nombre,
        "columnas": columnas,
        "filas_totales": int(len(df)),
        "filas_con_clave_completa": int(validos.sum()),
        "filas_sin_clave_completa": int((~validos).sum()),
        "claves_distintas": int(len(tamanos)),
        "claves_repetidas": int(len(repetidos)),
        "filas_en_claves_repetidas": int(repetidos.sum()),
        "max_repeticiones": int(tamanos.max()) if len(tamanos) else 0,
    }


def main() -> None:
    SALIDA.mkdir(parents=True, exist_ok=True)
    archivos = sorted(PARQUETS.glob("*.parquet"))
    columnas = IDS + DETALLE + [OBJETIVO, "_archivo", "_registro_archivo"]
    partes = [pd.read_parquet(p, columns=columnas) for p in archivos]
    df = pd.concat(partes, ignore_index=True)
    del partes

    for col in ["CodigoExterno", "FechaEnvioOferta", "Moneda de la Oferta", OBJETIVO, "_archivo"]:
        df[col] = normalizar_texto(df[col])
    for col in [c for c in columnas if c not in {"CodigoExterno", "FechaEnvioOferta", "Moneda de la Oferta", OBJETIVO, "_archivo"}]:
        if col != "_registro_archivo":
            df[col] = pd.to_numeric(df[col], errors="coerce")

    claves = [
        ("licitacion", ["Codigo"]),
        ("linea", ["Codigo", "Codigoitem"]),
        ("linea_correlativo", ["Codigo", "Codigoitem", "Correlativo"]),
        ("linea_proveedor", ["Codigo", "Codigoitem", "CodigoProveedor"]),
        ("linea_proveedor_sucursal", ["Codigo", "Codigoitem", "CodigoProveedor", "CodigoSucursalProveedor"]),
        ("oferta_ids", ["Codigo", "Codigoitem", "Correlativo", "CodigoProveedor", "CodigoSucursalProveedor"]),
        ("oferta_ids_fecha", ["Codigo", "Codigoitem", "Correlativo", "CodigoProveedor", "CodigoSucursalProveedor", "FechaEnvioOferta"]),
        ("oferta_ids_fecha_monto", ["Codigo", "Codigoitem", "Correlativo", "CodigoProveedor", "CodigoSucursalProveedor", "FechaEnvioOferta", "MontoUnitarioOferta", "Cantidad Ofertada"]),
    ]
    resumen = [estadistica_clave(df, n, c) for n, c in claves]
    pd.DataFrame(resumen).to_csv(SALIDA / "resumen_claves.csv", index=False, encoding="utf-8-sig")

    clave_base = claves[5][1]
    completos = df[clave_base].notna().all(axis=1)
    dup = df.loc[completos, clave_base].duplicated(keep=False)
    casos = df.loc[completos].loc[dup].copy()
    if len(casos):
        g = casos.groupby(clave_base, observed=True, dropna=False)
        diferencias = g.agg(
            filas=("Codigo", "size"),
            archivos=("_archivo", "nunique"),
            fechas_oferta=("FechaEnvioOferta", "nunique"),
            productos=("CodigoProductoONU", "nunique"),
            cantidades=("Cantidad Ofertada", "nunique"),
            monedas=("Moneda de la Oferta", "nunique"),
            montos_unitarios=("MontoUnitarioOferta", "nunique"),
            totales=("Valor Total Ofertado", "nunique"),
            objetivos=(OBJETIVO, "nunique"),
        ).reset_index()
        diferencias.to_parquet(SALIDA / "grupos_repetidos.parquet", index=False)
        diferencias.sort_values(["filas", "archivos"], ascending=False).head(200).to_csv(
            SALIDA / "muestra_grupos_repetidos.csv", index=False, encoding="utf-8-sig"
        )
        ejemplo_keys = diferencias.sort_values(
            ["fechas_oferta", "montos_unitarios", "totales", "filas"], ascending=False
        ).head(25)[clave_base]
        ejemplos = casos.merge(ejemplo_keys, on=clave_base, how="inner")
        ejemplos.sort_values(clave_base + ["FechaEnvioOferta"]).to_csv(
            SALIDA / "ejemplos_diferenciables.csv", index=False, encoding="utf-8-sig"
        )
    else:
        diferencias = pd.DataFrame()

    faltantes = pd.DataFrame({
        "columna": IDS + DETALLE,
        "faltantes": [int(df[c].isna().sum()) for c in IDS + DETALLE],
        "porcentaje": [float(df[c].isna().mean() * 100) for c in IDS + DETALLE],
        "unicos": [int(df[c].nunique(dropna=True)) for c in IDS + DETALLE],
    })
    faltantes.to_csv(SALIDA / "calidad_columnas.csv", index=False, encoding="utf-8-sig")

    info = {
        "filas": int(len(df)),
        "archivos": len(archivos),
        "grupos_oferta_ids_repetidos": int(len(diferencias)),
        "nota": "_archivo y _registro_archivo identifican siempre la fila fisica y no se usan como atributos del modelo.",
    }
    (SALIDA / "resumen.json").write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(info, indent=2, ensure_ascii=False))
    print(pd.DataFrame(resumen).to_string(index=False))


if __name__ == "__main__":
    main()
