"""Controles de fugas, temporalidad, monedas y paridad de inferencia."""
import tempfile
from pathlib import Path
import unittest
import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from pandas.testing import assert_frame_equal

import modelo_temporal as m


def example(n=120):
    d = pd.DataFrame({c:['dato']*n for c in m.READ_COLS})
    d['Codigo'] = ['proceso']*n
    d['Estado'] = 'Adjudicada'
    d['FechaPublicacion'] = '2024-01-02'
    d['FechaAdjudicacion'] = '2024-01-20'
    d['Oferta seleccionada'] = ['Seleccionada','No Seleccionada']*(n//2)
    for c in m.RAW_NUM:
        d[c] = '100,5'
    d['Moneda Adquisicion'] = 'Peso Chileno'
    d['Moneda de la Oferta'] = 'Peso Chileno'
    return d


def four_parts():
    groups=[]
    for code,pub,adj in [('a','2024-01-02','2024-02-01'),
                         ('b','2024-06-02','2024-07-01'),
                         ('c','2024-08-02','2024-09-01'),
                         ('d','2024-10-02','2024-11-01')]:
        d=example()
        d['Codigo'],d['FechaPublicacion'],d['FechaAdjudicacion']=code,pub,adj
        groups.append(d)
    return pd.concat(groups,ignore_index=True)


class ProtocolTests(unittest.TestCase):
    def test_outcomes_never_enter_features(self):
        d=example()
        original=m.features(d)
        d['Oferta seleccionada']='No Seleccionada'
        d['Estado']='Publicada'
        d['FechaAdjudicacion']='2030-12-31'
        d['NumeroOferentes']='999'
        d['MontoLineaAdjudica']='1000000'
        assert_frame_equal(original,m.features(d))
        self.assertFalse(set(m.META)-{'FechaPublicacion'} & set(original))

    def test_unknown_offer_currency_is_not_clp(self):
        d=example(2)
        d.loc[0,'Moneda de la Oferta']='Moneda revisar'
        d.loc[1,'Moneda de la Oferta']='Dolar'
        x=m.features(d)
        self.assertTrue(x.loc[0,[c for c in x if c.startswith('Valor Total Ofertado_') and c.endswith('_log')]].isna().all())
        self.assertTrue(pd.isna(x.loc[1,'Valor Total Ofertado_clp_log']))
        self.assertAlmostEqual(x.loc[1,'Valor Total Ofertado_usd_log'],np.log1p(100.5))

    def test_missing_and_invalid_quantities_remain_missing(self):
        d=example(4)
        d['Cantidad Ofertada']=['NA','0','-1','2']
        x=m.features(d)
        self.assertTrue(x['Cantidad Ofertada_log'].iloc[:3].isna().all())
        self.assertEqual(x['Cantidad Ofertada_invalida'].tolist(),[1,1,1,0])
        self.assertEqual(len(x),len(d))

    def test_decimal_and_scientific_original(self):
        s=m.number(pd.Series(['1,5e+07','NA','inf','1.234,56','1,25']))
        self.assertEqual(s.iloc[0],15000000)
        self.assertTrue(s.iloc[1:4].isna().all())
        self.assertEqual(s.iloc[-1],1.25)

    def test_hash_sampling_is_per_group_and_order_independent(self):
        s=pd.Series(['uno','dos','uno','tres'])
        h=m.hash_groups(s)
        self.assertEqual(h.iloc[0],h.iloc[2])
        self.assertEqual(h.iloc[1],m.hash_groups(s.iloc[::-1]).loc[1])

    def test_partitions_never_share_tenders(self):
        d,parts,audit=m.partition(four_parts(),'2026-11-04')
        self.assertEqual({k:len(v) for k,v in parts.items()},dict.fromkeys(m.BOUNDARIES,120))
        self.assertEqual(audit.particion.to_dict(),{'a':'train','b':'calibracion','c':'validacion','d':'test'})

    def test_future_result_excludes_whole_tender(self):
        raw=four_parts()
        extra=example()
        extra['Codigo']='tarde'
        extra['FechaAdjudicacion']='2024-07-01'
        _,_,a=m.partition(pd.concat([raw,extra],ignore_index=True),'2026-11-04')
        self.assertEqual(a.loc['tarde','particion'],'excluida')
        self.assertTrue(a.loc['tarde','resultado_no_disponible_al_corte'])

    def test_one_invalid_row_excludes_whole_tender(self):
        raw=four_parts()
        extra=example()
        extra['Codigo']='mixto'
        extra.loc[0,'Estado']='Publicada'
        _,_,a=m.partition(pd.concat([raw,extra],ignore_index=True),'2026-11-04')
        self.assertEqual(a.loc['mixto','particion'],'excluida')
        self.assertTrue(a.loc['mixto','estado_no_adjudicado'])

    def test_threshold_respects_ties_recall_and_min_alerts(self):
        y=np.array([1,0,1,0,0,1])
        p=np.array([.9,.9,.7,.7,.1,.1])
        t=m.threshold(y,p,.6,3)
        self.assertGreaterEqual(int((p>=t).sum()),3)
        self.assertGreaterEqual(y[p>=t].sum()/y.sum(),.6)
        choices=[v for v in np.unique(p) if (p>=v).sum()>=3 and y[p>=v].sum()/y.sum()>=.6]
        self.assertEqual(y[p>=t].mean(),max(y[p>=v].mean() for v in choices))
        self.assertIsNone(m.threshold(y,p,.3,100))

    def test_serialized_inference_uses_raw_contract(self):
        d=example()
        d['Cantidad']=[str(v) for v in range(1,len(d)+1)]
        x=m.features(d,'base')
        y=d['Oferta seleccionada'].eq('Seleccionada').astype(int)
        base=m.pipeline(x,'logistica').fit(x.iloc[:60],y.iloc[:60])
        model=CalibratedClassifierCV(FrozenEstimator(base),method='sigmoid').fit(x.iloc[60:90],y.iloc[60:90])
        a={'version':m.VERSION,'modelo':model,'variante':'base','columnas':x.columns.tolist(),'umbral':.5}
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'m.joblib'
            joblib.dump(a,path)
            loaded=joblib.load(path)
            np.testing.assert_allclose(m.predict(loaded,d.iloc[90:]).probabilidad,
                                       model.predict_proba(x.iloc[90:])[:,1])
        with self.assertRaises(ValueError):
            m.predict(a,d.drop(columns='Cantidad'))


if __name__=='__main__':
    unittest.main()
