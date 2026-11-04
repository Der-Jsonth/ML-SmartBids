"""V0.3: ofertas propias ya enviadas, antes de admisibilidad/adjudicación.

Modelo exploratorio: Estado y FechaAdjudicacion del CSV son proxies de madurez,
no una fotografía histórica verificable. Ningún artefacto se promueve automáticamente.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone, timedelta
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
import warnings

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.frozen import FrozenEstimator
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss,
    confusion_matrix, log_loss, precision_recall_curve, precision_score,
    recall_score, roc_auc_score)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

VERSION = '0.3.0'
INGEST_VERSION = 1
ROOT = Path(__file__).resolve().parent
NA = {'', 'NA', 'N/A', 'NULL', 'NAN', 'NONE'}
CURRENCIES = {'Peso Chileno': 'clp', 'Dolar': 'usd', 'Unidad de Fomento': 'uf', 'Euro': 'eur'}
BASE_CAT = ['CodigoUnidad', 'CodigoProductoONU', 'sector', 'Tipo', 'RegionUnidad',
            'Moneda Adquisicion', 'Moneda de la Oferta']
PRODUCT_CAT = ['UnidadMedida', 'Rubro1', 'Rubro2', 'Rubro3']
META = ['Codigo', 'CodigoExterno', 'Codigoitem', 'Correlativo', 'CodigoSucursalProveedor',
        'Estado', 'FechaPublicacion', 'FechaAdjudicacion', 'Oferta seleccionada']
RAW_NUM = ['MontoEstimado', 'Cantidad', 'Cantidad Ofertada', 'Valor Total Ofertado', 'MontoUnitarioOferta']
INPUTS = list(dict.fromkeys(BASE_CAT + PRODUCT_CAT + ['CodigoProveedor'] + RAW_NUM + ['FechaPublicacion']))
READ_COLS = list(dict.fromkeys(META + INPUTS))
FEATURE_SETS = ['base', 'producto', 'proveedor', 'cantidad_precio']
BOUNDARIES = {'train': ('2024-01-01', '2024-06-01'),
              'calibracion': ('2024-06-01', '2024-08-01'),
              'validacion': ('2024-08-01', '2024-10-01'),
              'test': ('2024-10-01', '2025-01-01')}


def save_json(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def clean(s):
    s = s.astype('string').str.strip()
    return s.mask(s.str.upper().isin(NA))


def number(s):
    # El original usa coma decimal, incluso en notación científica; no adivinamos miles.
    return pd.to_numeric(clean(s).str.replace(',', '.', regex=False), errors='coerce').astype(float).replace([np.inf, -np.inf], np.nan)


def dates(s):
    # Los archivos inspeccionados usan ISO; evita reinterpretar fechas ambiguas.
    return pd.to_datetime(clean(s), format='%Y-%m-%d', errors='coerce')


def hash_groups(s):
    # SHA256 estable entre versiones de pandas, sin etiquetas ni orden de filas.
    unique = s.drop_duplicates()
    mapping = {v: int.from_bytes(hashlib.sha256(('smartbids-v03:' + str(v)).encode()).digest()[:8], 'big')
               for v in unique}
    return s.map(mapping).astype('uint64')


def features(raw, variant='cantidad_precio'):
    """Contrato de inferencia: mismos campos crudos y misma preparación que al ajustar.

    Solo usa una lista explícita; ninguna etiqueta/estado/fecha final entra al modelo.
    Importable para cargar joblib, sin transformadores personalizados serializados.
    """
    if variant not in FEATURE_SETS:
        raise ValueError(variant)
    required = BASE_CAT + ['MontoEstimado', 'Cantidad', 'Cantidad Ofertada', 'Valor Total Ofertado', 'FechaPublicacion']
    if variant in ['producto', 'proveedor', 'cantidad_precio']:
        required += PRODUCT_CAT
    if variant in ['proveedor', 'cantidad_precio']:
        required += ['CodigoProveedor']
    if variant == 'cantidad_precio':
        required += ['MontoUnitarioOferta']
    missing = sorted(set(required) - set(raw.columns))
    if missing:
        raise ValueError(f'Faltan campos crudos: {missing}')
    x = pd.DataFrame(index=raw.index)
    cats = BASE_CAT.copy()
    if variant != 'base':
        cats += PRODUCT_CAT
    if variant in ['proveedor', 'cantidad_precio']:
        cats += ['CodigoProveedor']
    for c in cats:
        x[c] = clean(raw[c]).fillna('sin_dato').str.casefold().astype(str)
    for c in ['Cantidad', 'Cantidad Ofertada']:
        v = number(raw[c])
        x[c + '_log'] = np.log1p(v.where(v > 0))
        x[c + '_invalida'] = (v.isna() | v.le(0)).astype(float)
    amounts = [('MontoEstimado', 'Moneda Adquisicion'), ('Valor Total Ofertado', 'Moneda de la Oferta')]
    if variant == 'cantidad_precio':
        amounts.append(('MontoUnitarioOferta', 'Moneda de la Oferta'))
        offered, requested = number(raw['Cantidad Ofertada']), number(raw['Cantidad'])
        ratio = (offered.where(offered > 0) / requested.where(requested > 0)).replace([np.inf, -np.inf], np.nan)
        x['Cobertura_cantidad_log'] = np.log1p(ratio)
    for col, currency_col in amounts:
        v = number(raw[col])
        for currency, suffix in CURRENCIES.items():
            # Cada moneda tiene su propia escala. Una moneda desconocida no se infiere.
            x[col + '_' + suffix + '_log'] = np.log1p(v.where(v.gt(0) & raw[currency_col].eq(currency)))
        x[col + '_invalido'] = (v.isna() | v.le(0)).astype(float)
    pub = dates(raw['FechaPublicacion'])
    x['MesPublicacion'] = pub.dt.month.astype(float)
    x['DiaSemanaPublicacion'] = pub.dt.dayofweek.astype(float)
    return x


def load_original(source, fraction, output):
    """Lee todos los CSV; retiene grupos completos por hash, sin balancear clases."""
    if not 0 < fraction <= 1:
        raise ValueError('La fracción debe estar en (0,1].')
    paths = sorted(Path(source).glob('lic_2024-*.csv'))
    if len(paths) != 12:
        raise ValueError(f'Se requieren los 12 archivos mensuales de 2024; encontrados: {len(paths)}')
    chunks, manifest = [], []
    for path in paths:
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(8*1024*1024), b''):
                digest.update(block)
        total, kept = 0, 0
        states, labels = Counter(), Counter()
        for frame in pd.read_csv(path, sep=';', encoding='cp1252', encoding_errors='replace',
                dtype=str, keep_default_na=False, usecols=READ_COLS, chunksize=25000):
            total += len(frame)
            states.update(frame.Estado)
            labels.update(frame['Oferta seleccionada'])
            ids = clean(frame.Codigo)
            mask = ids.notna() & (hash_groups(ids.fillna('sin_codigo')) / np.float64(2**64) < fraction)
            part = frame.loc[mask].copy()
            part['_archivo'] = path.name
            kept += len(part)
            chunks.append(part)
        manifest.append({'archivo': path.name, 'bytes': path.stat().st_size,
                         'sha256': digest.hexdigest(), 'filas': total, 'retenidas': kept,
                         'estados': dict(states), 'etiquetas': dict(labels)})
        print(f'Leído {path.name}: {total:,}; muestra por licitación: {kept:,}', flush=True)
    raw = pd.concat(chunks, ignore_index=True)
    save_json(output/'fuentes.json', manifest)
    raw.to_parquet(output/'original_muestra.parquet', index=False)
    return raw


def partition(raw, asof):
    """Exclusiones auditadas por licitación; nunca fragmenta grupos por fila."""
    d = raw.copy()
    d['_pub'] = dates(d.FechaPublicacion)
    d['_adj'] = dates(d.FechaAdjudicacion)
    g = d.groupby('Codigo', sort=False)
    stats = g.agg(filas=('Codigo', 'size'), pub=('_pub', 'min'), pub_max=('_pub', 'max'),
                  adj=('_adj', 'max'), adj_min=('_adj', 'min'))
    bad_rows = pd.DataFrame({
        'estado_no_adjudicado': d.Estado.ne('Adjudicada'),
        'etiqueta_invalida': ~d['Oferta seleccionada'].isin(['Seleccionada', 'No Seleccionada']),
        'fecha_invalida': d._pub.isna() | d._adj.isna(),
        'fecha_incoherente': d._adj.lt(d._pub) | d._adj.gt(pd.Timestamp(asof)),
        'texto_corrupto_en_clave': d[['Codigo','CodigoUnidad','CodigoProductoONU','CodigoProveedor']].apply(
            lambda s: s.str.contains('\ufffd', regex=False)).any(axis=1),
    }, index=d.index)
    flags = bad_rows.groupby(d.Codigo).any()
    flags['publicacion_inconsistente'] = stats.pub.ne(stats.pub_max)
    flags['adjudicacion_inconsistente'] = stats.adj.ne(stats.adj_min)
    usable = ~flags.any(axis=1)
    assignment = pd.Series('excluida', index=stats.index, dtype=str)
    for name, (start, end) in BOUNDARIES.items():
        date_limit = pd.Timestamp(end) if name != 'test' else pd.Timestamp(asof) + timedelta(days=1)
        in_window = stats.pub.ge(start) & stats.pub.lt(end)
        known = stats.adj.lt(date_limit)
        assignment.loc[usable & in_window & known] = name
    flags['resultado_no_disponible_al_corte'] = usable & assignment.eq('excluida') & stats.pub.ge('2024-01-01') & stats.pub.lt('2025-01-01')
    audit = stats.join(flags).assign(particion=assignment)
    d['_particion'] = d.Codigo.map(assignment)
    d['_y'] = d['Oferta seleccionada'].eq('Seleccionada').astype('int8')
    parts = {name: np.flatnonzero(d._particion.eq(name)) for name in BOUNDARIES}
    seen = set()
    for name, idx in parts.items():
        groups = set(d.iloc[idx].Codigo)
        if seen & groups:
            raise AssertionError('Licitaciones compartidas entre particiones')
        seen |= groups
        if len(idx) < 100 or d.iloc[idx]._y.nunique() != 2:
            raise ValueError(f'{name}: insuficientes registros o clases; aumentar fracción o revisar cortes')
    return d, parts, audit


def limit_train(d, indices, limit):
    counts = d.iloc[indices].groupby('Codigo').size()
    order = hash_groups(pd.Series(counts.index, index=counts.index)).sort_values().index
    chosen, used = [], 0
    for code in order:
        n = int(counts.loc[code])
        if used + n <= limit:
            chosen.append(code)
            used += n
    result = indices[d.iloc[indices].Codigo.isin(chosen).to_numpy()]
    if len(result) < 100 or d.iloc[result]._y.nunique() != 2:
        raise ValueError('El límite de entrenamiento no admite grupos y clases suficientes')
    return result


def pipeline(x, algorithm):
    cats = x.select_dtypes(include=['object', 'string']).columns.tolist()
    nums = [c for c in x if c not in cats]
    prep = ColumnTransformer([
        ('numeros', Pipeline([('imputar', SimpleImputer(strategy='median', add_indicator=True, keep_empty_features=True)),
                              ('escalar', StandardScaler())]), nums),
        ('categorias', OneHotEncoder(handle_unknown='infrequent_if_exist', min_frequency=10,
                                     max_categories=200, dtype=np.float32), cats),
    ])
    algorithms = {
        'logistica': LogisticRegression(solver='liblinear', max_iter=500, random_state=42),
        'arbol': DecisionTreeClassifier(max_depth=9, min_samples_leaf=60, random_state=42),
        'bosque': RandomForestClassifier(n_estimators=100, max_depth=13, min_samples_leaf=30,
                                         n_jobs=2, random_state=42),
        # No validación aleatoria interna: puede compartir licitaciones y mezclar fechas.
        'mlp': MLPClassifier(hidden_layer_sizes=(24,), max_iter=100, early_stopping=False,
                             random_state=42, batch_size=512, n_iter_no_change=12),
    }
    return Pipeline([('preparacion', prep), ('clasificador', algorithms[algorithm])])


def threshold(y, p, recall_min=.30, min_alerts=200):
    if not 0 < recall_min <= 1 or min_alerts < 1:
        raise ValueError('Recall y alertas inválidos')
    y, p = np.asarray(y), np.asarray(p)
    if set(np.unique(y)) != {0, 1} or not np.isfinite(p).all():
        raise ValueError('Se requieren ambas clases y probabilidades finitas')
    precision, recall, ts = precision_recall_curve(y, p)
    alerts = len(p) - np.searchsorted(np.sort(p), ts, side='left')
    good = np.flatnonzero((recall[:-1] >= recall_min) & (alerts >= min_alerts))
    if not len(good):
        return None
    # Igual criterio que V0.2: precisión, luego recall; regla de empates explícita.
    order = np.lexsort((recall[:-1][good], precision[:-1][good]))
    return float(ts[good[order[-1]]])


def metrics(y, p, t):
    y, p = np.asarray(y), np.asarray(p)
    pred = p >= t
    return {'filas': len(y), 'prevalencia': float(y.mean()),
            'precision': float(precision_score(y, pred, zero_division=0)),
            'recall': float(recall_score(y, pred, zero_division=0)),
            'average_precision': float(average_precision_score(y, p)),
            'roc_auc': float(roc_auc_score(y, p)),
            'brier': float(brier_score_loss(y, p)),
            'log_loss': float(log_loss(y, p, labels=[0, 1])),
            'alertas': int(pred.sum()), 'cobertura': float(pred.mean()),
            'matriz_confusion': confusion_matrix(y, pred, labels=[0, 1]).tolist()}


def cluster_bootstrap(frame, candidates, iterations=300):
    """Remuestrea licitaciones, no filas, con umbrales ya fijados en validación."""
    sums = []
    for name, t in candidates:
        pred = frame[name].ge(t)
        agg = pd.DataFrame({'Codigo': frame.Codigo, 'tp': pred & frame.y.eq(1),
                            'alertas': pred, 'positivas': frame.y.eq(1)}).groupby('Codigo').sum()
        sums.append((name, agg[['tp','alertas','positivas']].to_numpy(dtype=float)))
    rng = np.random.default_rng(42)
    values = {name: [] for name, _ in sums}
    n = len(sums[0][1])
    for _ in range(iterations):
        idx = rng.integers(0, n, size=n)
        for name, array in sums:
            tp, alerts, positives = array[idx].sum(axis=0)
            values[name].append([tp/alerts if alerts else 0, tp/positives if positives else 0])
    out = {}
    for name, v in values.items():
        a = np.array(v)
        out[name] = {'precision_ic95': np.quantile(a[:,0], [.025,.975]).tolist(),
                     'recall_ic95': np.quantile(a[:,1], [.025,.975]).tolist()}
    if len(sums) > 1:
        a = np.array(values[sums[0][0]])[:,0] - np.array(values[sums[1][0]])[:,0]
        out['diferencia_precision_ganador_menos_base_ic95'] = np.quantile(a, [.025,.975]).tolist()
    return out


def predict(artifact, raw):
    if artifact.get('version') != VERSION:
        raise ValueError('Versión de artefacto incompatible')
    x = features(raw, artifact['variante'])
    if x.columns.tolist() != artifact['columnas']:
        raise ValueError('El contrato de variables cambió')
    p = artifact['modelo'].predict_proba(x)[:,1]
    return pd.DataFrame({'probabilidad': p, 'prediccion': (p >= artifact['umbral']).astype(int)}, index=raw.index)


def train(raw, args, output):
    d, parts, audit = partition(raw, args.asof)
    audit.to_csv(output/'auditoria_licitaciones.csv', encoding='utf-8-sig')
    flags = audit.select_dtypes(include='bool').columns
    # Las causas se solapan: no sumar estos recuentos como si fueran exclusiones netas.
    exclusions = {c:{'licitaciones':int(audit[c].sum()), 'filas':int(audit.loc[audit[c],'filas'].sum())} for c in flags}
    exclusions['total_excluidas']={'licitaciones':int(audit.particion.eq('excluida').sum()),
                                  'filas':int(audit.loc[audit.particion.eq('excluida'),'filas'].sum())}
    key=['Codigo','Codigoitem','CodigoProveedor','CodigoSucursalProveedor','Correlativo']
    exclusions['diagnostico_clave_candidata']={'campos':key,'filas_con_clave_repetida':int(raw.duplicated(key,keep=False).sum()),
                                              'accion':'Solo diagnóstico; no se eliminan automáticamente ofertas diferentes'}
    save_json(output/'exclusiones.json',exclusions)
    summary = {k: {'filas': len(idx), 'licitaciones': int(d.iloc[idx].Codigo.nunique()),
                     'prevalencia': float(d.iloc[idx]._y.mean())} for k,idx in parts.items()}
    save_json(output/'particiones.json', summary)
    print('Particiones:', json.dumps(summary), flush=True)
    fit = limit_train(d, parts['train'], args.max_train)
    summary['train']['filas_usadas_ajuste'] = len(fit)
    summary['train']['licitaciones_usadas_ajuste'] = int(d.iloc[fit].Codigo.nunique())
    summary['train']['filas_fuera_por_limite'] = len(parts['train'])-len(fit)
    save_json(output/'particiones.json', summary)
    y = d._y
    val, cal, test = parts['validacion'], parts['calibracion'], parts['test']
    specs = [('logistica', v) for v in FEATURE_SETS] + [(a, 'cantidad_precio') for a in ['arbol','bosque','mlp']]
    rows, fitted = [], {}
    for algorithm, variant in specs:
        name = algorithm + '_' + variant
        start = time.perf_counter()
        x = features(d, variant)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always', ConvergenceWarning)
            base = pipeline(x, algorithm).fit(x.iloc[fit], y.iloc[fit])
        model = CalibratedClassifierCV(FrozenEstimator(base), method='sigmoid').fit(x.iloc[cal], y.iloc[cal])
        p = model.predict_proba(x.iloc[val])[:,1]
        t = threshold(y.iloc[val], p, args.recall, args.min_alerts)
        converged = not any(issubclass(w.category, ConvergenceWarning) for w in caught)
        record = {'candidato': name, 'algoritmo': algorithm, 'variante': variant,
                  'umbral': t, 'convergio': converged, 'elegible': t is not None and converged,
                  'segundos': time.perf_counter()-start,
                  'advertencias': [str(w.message) for w in caught],
                  **metrics(y.iloc[val], p, t if t is not None else .5)}
        rows.append(record)
        fitted[name] = (model, variant, x.columns.tolist())
        save_json(output/'validacion.json', rows)
        print(f'{name}: precisión={record["precision"]:.3f}, recall={record["recall"]:.3f}, elegible={record["elegible"]}', flush=True)
    table = pd.DataFrame(rows)
    eligible = table[table.elegible].sort_values(['precision','recall','brier','candidato'], ascending=[False,False,True,True])
    if eligible.empty:
        raise ValueError('Ningún candidato converge y satisface la política en validación; no se guarda ganador')
    winner = eligible.iloc[0]
    # El registro de decisión se escribe antes de generar probabilidades de test.
    decision = {'candidato': winner.candidato, 'umbral': float(winner.umbral),
                'criterio': 'precisión en validación, recall mínimo y soporte; convergencia requerida',
                'recall_minimo': args.recall, 'min_alertas': args.min_alerts,
                'decidido_antes_de_evaluacion_test': True}
    save_json(output/'decision.json', decision)
    selected_names = list(dict.fromkeys([winner.candidato, 'logistica_base']))
    pred_frame = d.iloc[test][['Codigo','CodigoExterno','Codigoitem','CodigoProveedor','FechaPublicacion','sector','_y']].rename(columns={'_y':'y'}).reset_index(drop=True)
    test_results, ts = {}, []
    for name in selected_names:
        row = table.loc[table.candidato.eq(name)].iloc[0]
        model, variant, columns = fitted[name]
        p = model.predict_proba(features(d.iloc[test], variant))[:,1]
        t = float(row.umbral) if pd.notna(row.umbral) else .5
        pred_frame[name] = p
        test_results[name] = metrics(y.iloc[test], p, t)
        ts.append((name,t))
    dummy = DummyClassifier(strategy='prior').fit(np.zeros((len(fit),1)), y.iloc[fit])
    dummy_p = dummy.predict_proba(np.zeros((len(test),1)))[:,1]
    test_results['dummy_prior'] = metrics(y.iloc[test], dummy_p, .5)
    bootstrap = cluster_bootstrap(pred_frame, ts)
    model, variant, columns = fitted[winner.candidato]
    artifact = {'version':VERSION, 'modelo':model, 'variante':variant, 'columnas':columns,
                'campos_crudos':INPUTS, 'umbral':float(winner.umbral),
                'instante':'Oferta propia enviada; antes de evaluación de admisibilidad',
                'promovido':False, 'requiere_preparacion':'modelo_temporal.features / predict',
                'decision':decision}
    joblib.dump(artifact, output/'modelo_candidato.joblib')
    # Paridad tras serialización y contrato de inferencia probado sobre entradas crudas.
    reloaded = joblib.load(output/'modelo_candidato.joblib')
    example_raw = d.iloc[test[:12]][INPUTS].copy()
    example = predict(reloaded, example_raw)
    np.testing.assert_allclose(example.probabilidad, pred_frame[winner.candidato].iloc[:len(example)])
    example_raw.to_json(output/'ejemplo_entrada.json', orient='records', force_ascii=False, indent=2)
    example.to_csv(output/'ejemplo_predicciones.csv', index=False)
    pred_frame.to_parquet(output/'predicciones_test.parquet', index=False)
    # Diagnósticos de test; nunca se utilizan para elegir candidato o umbral.
    diagnostics = []
    wp = pred_frame[winner.candidato].to_numpy()
    for field, series in [('mes',pd.to_datetime(pred_frame.FechaPublicacion).dt.strftime('%Y-%m')),
                          ('sector',pred_frame.sector),
                          ('proveedor_visto',pred_frame.CodigoProveedor.isin(set(d.iloc[fit].CodigoProveedor)).astype(str))]:
        for value in series.unique():
            mask = series.eq(value).to_numpy()
            yy, pp = pred_frame.y.to_numpy()[mask], wp[mask]
            if len(yy) >= 20 and len(np.unique(yy)) == 2:
                diagnostics.append({'dimension':field, 'valor':str(value), **metrics(yy,pp,float(winner.umbral))})
    save_json(output/'diagnosticos_test.json', diagnostics)
    p = pred_frame[winner.candidato]
    bins = pd.qcut(p, q=10, duplicates='drop')
    calibration = pd.DataFrame({'banda':bins.astype(str),'p':p,'y':pred_frame.y}).groupby('banda').agg(
        filas=('y','size'), predicha=('p','mean'), observada=('y','mean')).reset_index()
    calibration.to_csv(output/'calibracion_test.csv',index=False)
    winner_m = test_results[winner.candidato]
    base_m = test_results['logistica_base']
    delta = bootstrap.get('diferencia_precision_ganador_menos_base_ic95')
    confirmed = bool(delta and delta[0] > 0 and winner_m['recall'] >= args.recall)
    unseen = {}
    train_x, test_x = features(d.iloc[fit],variant), features(d.iloc[test],variant)
    for c in train_x.select_dtypes(include=['object']).columns:
        unseen[c] = float((~test_x[c].isin(set(train_x[c]))).mean())
    report = {'version':VERSION,'decision':decision,'particiones':summary,
        'test':test_results,'bootstrap_por_licitacion':bootstrap,
        'tasas_categorias_no_vistas_test':unseen,
        'mejora_precision_con_ic_y_recall_test':confirmed,
        'promovido':False,'estado':'candidato_exploratorio',
        'limitaciones':[
            'Todo 2024 ya fue explorado: test temporal retrospectivo, no confirmatorio independiente.',
            'Estado y fecha de adjudicación son proxies; faltan snapshots y fecha verificable de resultado efectivo.',
            'Los campos del CSV podrían haber cambiado después del envío; no se acredita disponibilidad histórica.',
            'La población son procesos adjudicados con fechas válidas y resultado disponible al corte, no todas las licitaciones.',
            'Muestra por licitación y límite de ajuste documentados; no se usó sobremuestreo 50/50.',
            'Bootstrap condicionado a la muestra y al modelo ajustado: no captura toda la incertidumbre de entrenamiento.',
            'V0.2 usa otra población y partición: sus métricas no permiten declarar una mejora directa.',
            'No se construyeron tasas históricas de éxito sin verificar fechas efectivas de adjudicación.'
        ]}
    save_json(output/'informe.json',report)
    readme = f'''# Resultado V0.3 — candidato exploratorio

Ganador seleccionado en validación: **{winner.candidato}**. Umbral: {winner.umbral:.6f}.
Política: precisión máxima con recall ≥ {args.recall:.0%} y ≥ {args.min_alerts} alertas en validación.

| Prueba temporal retrospectiva | Precisión | Recall | AP | ROC AUC | Brier |
|---|---:|---:|---:|---:|---:|
| Candidato | {winner_m['precision']:.4f} | {winner_m['recall']:.4f} | {winner_m['average_precision']:.4f} | {winner_m['roc_auc']:.4f} | {winner_m['brier']:.4f} |
| Base con mismo protocolo | {base_m['precision']:.4f} | {base_m['recall']:.4f} | {base_m['average_precision']:.4f} | {base_m['roc_auc']:.4f} | {base_m['brier']:.4f} |

Intervalo bootstrap por licitación de diferencia de precisión candidato−base: {delta}.
Cumple diferencia positiva con IC95 y recall mínimo en test: **{confirmed}**.
Este indicador no acredita generalización futura. No se promovió ni reemplazó V0.2.

## Límites

''' + '\n'.join('- '+x for x in report['limitaciones']) + '\n'
    (output/'RESULTADO.md').write_text(readme,encoding='utf-8')
    print(readme,flush=True)
    return report


def main():
    # Windows puede heredar una consola cp1252 que no admite los signos del informe.
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='backslashreplace')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fuente',default='D:/Licit')
    parser.add_argument('--fraccion',type=float,default=.25)
    parser.add_argument('--max-train',type=int,default=80000)
    parser.add_argument('--recall',type=float,default=.30)
    parser.add_argument('--min-alerts',type=int,default=200)
    parser.add_argument('--asof',default='2026-11-04')
    parser.add_argument('--reusar-datos',action='store_true',help='Reutiliza caché solo si coincide configuración y archivos fuente')
    args = parser.parse_args()
    if not 0 < args.recall <= 1 or args.min_alerts < 1 or args.max_train < 100:
        parser.error('Política o límite inválido')
    # Una ejecución siempre recibe un directorio nuevo: no sobrescribe resultados.
    output = ROOT/'resultados'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output.mkdir(parents=True)
    paths = sorted(Path(args.fuente).glob('lic_2024-*.csv'))
    signature = [{'path':str(p.resolve()),'bytes':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns} for p in paths]
    config = {'version':VERSION,'ingesta_version':INGEST_VERSION,'argumentos':vars(args),'cortes':BOUNDARIES,'fuentes_estadisticas':signature,
              'codigo_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'entorno':{'python':platform.python_version(),'sklearn':sklearn.__version__,'pandas':pd.__version__,'numpy':np.__version__}}
    save_json(output/'configuracion.json',config)
    raw = None
    if args.reusar_datos:
        for candidate in sorted((ROOT/'resultados').iterdir(),reverse=True):
            if candidate == output or not (candidate/'original_muestra.parquet').exists():
                continue
            previous = json.loads((candidate/'configuracion.json').read_text(encoding='utf-8'))
            if previous.get('ingesta_version') == INGEST_VERSION and previous['fuentes_estadisticas'] == signature and previous['argumentos']['fraccion'] == args.fraccion:
                # Verifica contenido, además de tamaño/mtime; registra origen de la caché.
                manifest=json.loads((candidate/'fuentes.json').read_text(encoding='utf-8'))
                expected={m['archivo']:m['sha256'] for m in manifest}
                verified=True
                for p in paths:
                    with p.open('rb') as stream:
                        if hashlib.file_digest(stream,'sha256').hexdigest()!=expected.get(p.name):
                            verified=False
                            break
                if verified:
                    raw=pd.read_parquet(candidate/'original_muestra.parquet')
                    save_json(output/'fuentes.json',manifest)
                    config['cache_origen']=str(candidate)
                    save_json(output/'configuracion.json',config)
                    break
    if raw is None:
        raw = load_original(args.fuente,args.fraccion,output)
    try:
        train(raw,args,output)
        save_json(output/'estado_ejecucion.json',{'estado':'completa'})
    except Exception as exc:
        save_json(output/'error.json',{'tipo':type(exc).__name__,'mensaje':str(exc)})
        raise
    print('Resultados:',output,flush=True)


if __name__ == '__main__':
    main()
