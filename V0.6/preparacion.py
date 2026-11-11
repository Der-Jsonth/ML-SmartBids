"""V0.6: lectura tipificada y cohorte de ofertas sin unir registros."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent
VERSION = "0.6.2"
DATOS = ROOT / "datos" / "comparacion"
IDS = ["Codigo", "CodigoOrganismo", "CodigoUnidad", "Codigoitem", "CodigoProductoONU",
       "CodigoProveedor", "CodigoSucursalProveedor", "Correlativo"]
FECHAS = ["FechaPublicacion", "FechaEnvioOferta", "FechaAdjudicacion",
          "FechaEstimadaAdjudicacion", "FechaCierre", "FechaActoAperturaTecnica",
          "FechaActoAperturaEconomica"]
NUMERICAS = ["MontoEstimado", "Cantidad", "Cantidad Ofertada", "Valor Total Ofertado",
             "MontoUnitarioOferta", "TiempoDuracionContrato", "NumeroOferentes"]
TEXTOS = ["CodigoExterno", "Estado", "CodigoEstado", "sector", "Tipo", "ComunaUnidad",
          "RegionUnidad", "Moneda Adquisicion", "Moneda de la Oferta", "UnidadMedida",
          "Rubro1", "Rubro2", "Rubro3", "TipoPago", "Modalidad", "SubContratacion",
          "FuenteFinanciamiento", "UnidadTiempoDuracionContrato", "TipoDuracionContrato",
          "Estado Oferta", "Oferta seleccionada", "RutProveedor", "NombreProveedor",
          "RazonSocialProveedor", "Nombre de la Oferta"]
COLUMNAS = list(dict.fromkeys(IDS + FECHAS + NUMERICAS + TEXTOS))
CLAVE_OFERTA = ["Codigo", "Codigoitem", "CodigoProveedor", "CodigoSucursalProveedor"]
CLAVE_NEGOCIO = CLAVE_OFERTA + ["Nombre de la Oferta", "FechaEnvioOferta",
                                "Moneda de la Oferta", "MontoUnitarioOferta",
                                "Cantidad Ofertada", "Valor Total Ofertado"]
PARES = {"proveedor": ["CodigoProveedor"],
         "proveedor_comprador": ["CodigoProveedor", "CodigoUnidad"],
         "proveedor_producto": ["CodigoProveedor", "CodigoProductoONU"]}
NA = {"", "NA", "N/A", "NULL", "NAN", "NONE"}

def configuracion():
    return json.loads((ROOT / "configuracion.json").read_text(encoding="utf-8"))

def guardar_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")

def sha256(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()

def limpiar(s):
    s = s.astype("string[pyarrow]").str.strip()
    return s.mask(s.str.upper().isin(NA))

def numero(s):
    return pd.to_numeric(limpiar(s).str.replace(",", ".", regex=False), errors="coerce").astype("float64").replace([np.inf, -np.inf], np.nan)

def entero_exacto(s):
    n = pd.to_numeric(limpiar(s), errors="coerce")
    valid = n.notna() & n.ge(0) & n.le(np.iinfo(np.uint32).max) & n.mod(1).eq(0)
    return n.where(valid).astype("UInt32")

def fecha(s):
    return pd.to_datetime(limpiar(s), format="%Y-%m-%d", errors="coerce").astype("datetime64[ms]")

def normalizar(raw):
    d = pd.DataFrame(index=raw.index)
    for c in IDS: d[c] = entero_exacto(raw[c])
    for c in FECHAS: d[c] = fecha(raw[c])
    for c in NUMERICAS: d[c] = numero(raw[c])
    for c in TEXTOS: d[c] = limpiar(raw[c])
    return d

def perfil(d):
    return pd.DataFrame([{"columna": c, "tipo": str(d[c].dtype), "filas": len(d),
                          "nulos": int(d[c].isna().sum()),
                          "porcentaje_nulos": float(100*d[c].isna().mean()),
                          "valores_unicos": int(d[c].nunique()),
                          "memoria_mb": float(d[c].memory_usage(index=False, deep=True)/1024**2)} for c in d])

def juntar(cfg=None):
    """Convierte los 12 CSV sin eliminar, agregar ni fusionar una sola fila."""
    cfg = cfg or configuracion()
    out = ROOT / "datos" / "juntado"; out.mkdir(parents=True, exist_ok=True)
    files = sorted(Path(cfg["fuente"]).glob(cfg["patron"]))
    if {p.stem[-2:] for p in files} != {f"{i:02d}" for i in range(1, 13)}:
        raise ValueError("Los archivos no cubren los doce meses exactamente una vez.")
    manifest, audits = [], []
    for path in files:
        columns = pd.read_csv(path, sep=";", encoding="cp1252", nrows=0).columns
        missing = set(COLUMNAS) - set(columns)
        if missing: raise ValueError(f"{path.name}: faltan {sorted(missing)}")
        dest, temp = out/f"{path.stem}.parquet", out/f"{path.stem}.parquet.tmp"
        writer, total, before, after, bad = None, 0, 0, 0, {}
        try:
            for raw in pd.read_csv(path, sep=";", encoding="cp1252", encoding_errors="replace",
                                   usecols=COLUMNAS, dtype="string[pyarrow]", keep_default_na=False,
                                   chunksize=cfg["tamano_bloque"]):
                before += int(raw.memory_usage(deep=True).sum()); d = normalizar(raw)
                for c in COLUMNAS:
                    item = bad.setdefault(c, {"corruptos": 0, "conversion_invalida": 0})
                    item["corruptos"] += int(raw[c].str.contains("\ufffd", regex=False, na=False).sum())
                    if c in IDS + FECHAS + NUMERICAS:
                        item["conversion_invalida"] += int((limpiar(raw[c]).notna() & d[c].isna()).sum())
                d["_archivo"] = pd.Series(path.name, index=d.index, dtype="string[pyarrow]")
                d["_registro_archivo"] = np.arange(total+1, total+len(d)+1, dtype="uint32")
                after += int(d.memory_usage(deep=True).sum())
                table = pa.Table.from_pandas(d, preserve_index=False)
                if writer is None: writer = pq.ParquetWriter(temp, table.schema, compression="zstd", use_dictionary=True)
                writer.write_table(table); total += len(d)
            if not total: raise ValueError(f"{path.name}: archivo vacío")
        finally:
            if writer is not None: writer.close()
        temp.replace(dest)
        manifest.append({"archivo": path.name, "bytes_original": path.stat().st_size,
                         "sha256_original": sha256(path), "filas": total, "parquet": dest.name,
                         "sha256_parquet": sha256(dest), "bytes_parquet": dest.stat().st_size,
                         "memoria_texto_bloques_mb": before/1024**2,
                         "memoria_tipificada_bloques_mb": after/1024**2})
        audits.extend({"archivo": path.name, "columna": c, **v} for c,v in bad.items())
        print(f"{path.name}: {total:,} registros; {dest.stat().st_size/1024**2:.1f} MB", flush=True)
    result = {"version": VERSION, "columnas": COLUMNAS, "configuracion": cfg,
              "fecha_lectura_utc": datetime.now(timezone.utc).isoformat(),
              "nota": "Cada registro CSV se conserva; no se deduplican ni unen ofertas.",
              "archivos": manifest, "filas": sum(x["filas"] for x in manifest)}
    guardar_json(ROOT/"datos"/"fuentes.json", result)
    pd.DataFrame(audits).to_csv(ROOT/"datos"/"auditoria_lectura.csv", index=False, encoding="utf-8-sig")
    return pd.DataFrame(manifest)

def leer_juntado():
    info = json.loads((ROOT/"datos"/"fuentes.json").read_text(encoding="utf-8"))
    if info["version"] != VERSION or info["columnas"] != COLUMNAS:
        raise ValueError("Cambió el esquema: vuelva a ejecutar el notebook 1.")
    parts = []
    for entry in info["archivos"]:
        path = ROOT/"datos"/"juntado"/entry["parquet"]
        if sha256(path) != entry["sha256_parquet"]: raise ValueError(f"Parquet modificado: {path.name}")
        parts.append(pd.read_parquet(path))
    d = pd.concat(parts, ignore_index=True)
    if len(d) != info["filas"]: raise AssertionError("El juntado perdió filas.")
    for c in TEXTOS + ["_archivo"]: d[c] = d[c].astype("category")
    return d

def verificar_originales(cfg=None):
    cfg = cfg or configuracion(); info = json.loads((ROOT/"datos"/"fuentes.json").read_text(encoding="utf-8"))
    files = {p.name:p for p in Path(cfg["fuente"]).glob(cfg["patron"])}
    if set(files) != {e["archivo"] for e in info["archivos"]}: raise ValueError("Cambió el conjunto de CSV.")
    for entry in info["archivos"]:
        p = files[entry["archivo"]]
        if p.stat().st_size != entry["bytes_original"] or sha256(p) != entry["sha256_original"]:
            raise ValueError(f"Cambió {p.name}; rehaga el juntado.")
    return True

def auditar_identidad(d):
    """Mide separadores; nunca modifica ni filtra la cohorte."""
    variants = {"linea": ["Codigo", "Codigoitem"], "linea_proveedor": CLAVE_OFERTA,
                "mas_nombre_oferta": CLAVE_OFERTA+["Nombre de la Oferta"],
                "clave_negocio": CLAVE_NEGOCIO,
                "identidad_fisica": ["_archivo", "_registro_archivo"]}
    rows = []
    for name, cols in variants.items():
        valid = d[cols].notna().all(axis=1); sizes = d.loc[valid].groupby(cols, observed=True, dropna=False).size()
        repeated = sizes.gt(1)
        rows.append({"clave": name, "columnas": " + ".join(cols), "filas": len(d),
                     "filas_clave_completa": int(valid.sum()), "claves_distintas": int(len(sizes)),
                     "claves_repetidas": int(repeated.sum()),
                     "filas_en_claves_repetidas": int(sizes[repeated].sum()),
                     "max_repeticiones": int(sizes.max()) if len(sizes) else 0})
    return pd.DataFrame(rows)

def particiones(d, cfg):
    labels = pd.Series("fuera", index=d.index, dtype="string")
    for name,(start,end) in cfg["cortes"].items(): labels.loc[d.FechaPublicacion.ge(start)&d.FechaPublicacion.lt(end)] = name
    if labels.eq("fuera").any(): raise ValueError(f"Hay {labels.eq('fuera').sum():,} filas fuera de los cortes.")
    order = {name:i for i,name in enumerate(cfg["cortes"])}; inverse = {i:name for name,i in order.items()}
    rank = labels.map(order).astype("uint8")
    assigned = rank.groupby(d.CodigoExterno, observed=True).transform("max").map(inverse).astype("category")
    if assigned.groupby(d.CodigoExterno, observed=True).nunique().gt(1).any(): raise AssertionError("Licitación dividida.")
    return assigned, int(labels.ne(assigned.astype("string")).sum())

def ledger_participaciones(raw, keys):
    cols = ["Codigo"]+keys; key_ok = raw[cols].notna().all(axis=1)
    valid = raw.FechaPublicacion.notna() & raw.FechaEnvioOferta.notna() & raw.FechaEnvioOferta.ge(raw.FechaPublicacion)
    source = raw.loc[key_ok, cols+["FechaEnvioOferta"]].assign(_valido=valid.loc[key_ok])
    event = source.groupby(cols, observed=True, sort=False).agg(fecha=("FechaEnvioOferta","max"), valido=("_valido","all")).reset_index()
    event = event.loc[event.valido].drop(columns="valido").reset_index(drop=True); event["n"] = np.uint32(1)
    return event

def consultar_historia(ledger, queries, keys):
    result = pd.Series(0., index=queries.index, dtype="float64")
    valid = queries[keys].notna().all(axis=1) & queries["_consulta"].notna(); q = queries.loc[valid, keys+["_consulta"]].copy()
    if ledger.empty or q.empty: return result
    daily = ledger.groupby(keys+["fecha"], observed=True)["n"].sum().reset_index().sort_values(keys+["fecha"])
    daily["n"] = daily.groupby(keys, observed=True)["n"].cumsum(); q["_indice"] = q.index
    found = pd.merge_asof(q.sort_values("_consulta"), daily.sort_values("fecha"), left_on="_consulta", right_on="fecha",
                          by=keys, direction="backward", allow_exact_matches=False)
    result.loc[found["_indice"]] = found["n"].fillna(0).to_numpy(dtype="float64"); return result

def agregar_historia(d):
    queries = d[["CodigoProveedor", "CodigoUnidad", "CodigoProductoONU"]].copy(); queries["_consulta"] = d.FechaPublicacion
    history = pd.DataFrame(index=d.index)
    for name,keys in PARES.items():
        ledger = ledger_participaciones(d, keys); ledger.to_parquet(DATOS/f"historia_{name}.parquet", index=False)
        count = consultar_historia(ledger, queries, keys); complete = queries[keys+["_consulta"]].notna().all(axis=1)
        history[f"hist_{name}_participaciones_log"] = np.log1p(count.where(complete)).astype("float32")
    return history

def preparar(cfg=None):
    cfg = cfg or configuracion(); DATOS.mkdir(parents=True, exist_ok=True); d = leer_juntado()
    if len(d) != 2_846_181: raise AssertionError(f"Se esperaban 2.846.181 filas; llegaron {len(d):,}.")
    d["_id_fila"] = np.arange(len(d), dtype="uint32")
    d["Seleccionada"] = d["Oferta seleccionada"].astype("string").map({"Seleccionada":1,"No Seleccionada":0}).astype("uint8")
    d["AnioPublicacion"] = d.FechaPublicacion.dt.year.astype("uint16")
    d["MesPublicacion"] = d.FechaPublicacion.dt.month.astype("uint8")
    d["DiaSemanaPublicacion"] = d.FechaPublicacion.dt.dayofweek.astype("uint8")
    d["_particion"], moved = particiones(d, cfg); d = d.join(agregar_historia(d))
    auditar_identidad(d).to_csv(DATOS/"auditoria_identidad.csv", index=False, encoding="utf-8-sig")
    perfil(d).to_csv(DATOS/"perfil_cohorte.csv", index=False, encoding="utf-8-sig")
    pd.crosstab(d.Estado, d["Oferta seleccionada"], dropna=False).to_csv(DATOS/"estado_vs_etiqueta.csv", encoding="utf-8-sig")
    path = DATOS/"cohorte.parquet"; d.to_parquet(path, index=False, compression="zstd")
    counts = {name:{"filas":int(d._particion.eq(name).sum()),
                    "licitaciones":int(d.loc[d._particion.eq(name),"CodigoExterno"].nunique()),
                    "prevalencia":float(d.loc[d._particion.eq(name),"Seleccionada"].mean())} for name in cfg["cortes"]}
    guardar_json(DATOS/"particiones.json", counts)
    guardar_json(DATOS/"auditoria_cohorte.json", {"filas_csv":len(d), "filas_cohorte":len(d),
                 "filas_eliminadas":0, "filas_unidas":0, "filas_movidas_para_no_partir_licitacion":moved,
                 "identidad_fisica":["_archivo","_registro_archivo"], "clave_negocio_auditada":CLAVE_NEGOCIO,
                 "nota":"Las claves repetidas se conservan como filas independientes."})
    provenance = {"version":VERSION, "configuracion":cfg, "cohorte_sha256":sha256(path),
                  "codigo_preparacion_sha256":sha256(Path(__file__)),
                  "fuentes_sha256":sha256(ROOT/"datos"/"fuentes.json"),
                  "historia_sha256":{name:sha256(DATOS/f"historia_{name}.parquet") for name in PARES}}
    guardar_json(DATOS/"preparacion.json", provenance); print("Cohorte preparada sin unir filas:", counts, flush=True); return d

if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("etapa", choices=["juntar","preparar","todo"]); args = parser.parse_args()
    if args.etapa in ["juntar","todo"]: juntar()
    if args.etapa in ["preparar","todo"]: preparar()
