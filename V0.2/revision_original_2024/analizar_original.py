"""Auditoría reproducible: CSV completos, perfiles exactos y muestra para exploración.
Ejecutar desde cualquier directorio con el Python de V0.2/.venv.
"""
from pathlib import Path
from collections import Counter
import json
import argparse
import pandas as pd
import numpy as np

OUT = Path(__file__).resolve().parent

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fuente', default='D:/Licit')
    args = parser.parse_args()
    paths = sorted(Path(args.fuente).glob('lic_2024-*.csv'))
    if not paths:
        raise FileNotFoundError(args.fuente)
    profiles, monthly, samples, tenders = {}, [], [], []
    targets = Counter()
    for path in paths:
        n = 0
        month_target = Counter()
        month_groups = []
        for i, df in enumerate(pd.read_csv(path, sep=';', encoding='cp1252',
                encoding_errors='replace', dtype=str, keep_default_na=False,
                chunksize=12000)):
            n += len(df)
            target = df['Oferta seleccionada'].str.strip()
            month_target.update(target)
            for c in df:
                s = df[c].str.strip()
                missing = s.str.upper().isin(['', 'NA', 'N/A', 'NULL', 'NAN', 'NONE'])
                p = profiles.setdefault(c, {'filas': 0, 'vacios': 0, 'reemplazo_unicode': 0,
                                            'valores': Counter(), 'saturada': False})
                p['filas'] += len(s)
                p['vacios'] += int(missing.sum())
                p['reemplazo_unicode'] += int(s.str.contains('\ufffd', regex=False).sum())
                if not p['saturada']:
                    p['valores'].update(s[~missing])
                    if len(p['valores']) > 3000:
                        p['saturada'] = True
            sample = df.sample(n=min(180, len(df)), random_state=42+i).copy()
            sample['_archivo'] = path.name
            sample['_chunk'] = i
            sample['_peso'] = len(df)/len(sample)
            samples.append(sample)
            g = df[['Codigo','CodigoExterno','FechaPublicacion']].copy()
            g['filas'] = 1
            g['positivas'] = target.str.lower().eq('seleccionada').astype(int)
            month_groups.append(g.groupby(['Codigo','CodigoExterno','FechaPublicacion'],dropna=False)[['filas','positivas']].sum().reset_index())
        groups = pd.concat(month_groups).groupby(['Codigo','CodigoExterno','FechaPublicacion'],dropna=False)[['filas','positivas']].sum().reset_index()
        groups['archivo'] = path.name
        tenders.append(groups)
        targets.update(month_target)
        monthly.append({'archivo': path.name, 'bytes': path.stat().st_size,
                        'filas': n, 'licitaciones_codigo': groups.Codigo.nunique(),
                        'etiquetas': dict(month_target)})
        print(path.name, n, dict(month_target), flush=True)
    rows = []
    for c,p in profiles.items():
        rows.append({'columna': c, 'filas': p['filas'], 'vacios': p['vacios'],
            'porcentaje_vacios': 100*p['vacios']/p['filas'],
            'cardinalidad_observada': len(p['valores']),
            'cardinalidad_truncada': p['saturada'],
            'reemplazo_unicode':p['reemplazo_unicode'],
            'valores_frecuentes': json.dumps(p['valores'].most_common(8),ensure_ascii=False),
            'frecuencias_completas':not p['saturada']})
    pd.DataFrame(rows).to_csv(OUT/'perfil_columnas.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(monthly).to_json(OUT/'resumen_mensual.json',orient='records',force_ascii=False,indent=2)
    pd.concat(samples,ignore_index=True).to_feather(OUT/'muestra_exploratoria.feather')
    pd.concat(tenders,ignore_index=True).to_csv(OUT/'claves_licitaciones.csv',index=False,encoding='utf-8-sig')
    (OUT/'resumen.json').write_text(json.dumps({'filas':sum(x['filas'] for x in monthly),
        'columnas':len(profiles),'etiquetas':dict(targets),'archivos':len(paths),
        'muestra':sum(len(x) for x in samples),
        'metodo_muestra':'180 filas aleatorias por bloque de 12000; semilla 42+indice_bloque; pesos guardados',
        'lectura':'cp1252 con reemplazo de bytes indefinidos (contabilizado por columna), separador ;, cadenas, sin convertir NA'},ensure_ascii=False,indent=2),encoding='utf-8')

if __name__ == '__main__':
    main()
