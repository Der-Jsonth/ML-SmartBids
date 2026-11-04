import unittest
import pandas as pd
from pandas.testing import assert_frame_equal

import probar_historia as m


def events():
    return pd.DataFrame({
        'Codigo':['anterior','misma','futura'],
        'CodigoProveedor':['p']*3,'CodigoUnidad':['u']*3,'CodigoProductoONU':['onu']*3,
        'envio':pd.to_datetime(['2024-01-01','2024-02-01','2024-11-01']),
        'publicacion':pd.to_datetime(['2024-01-01','2024-02-01','2024-11-01']),
        'adjudicacion':pd.to_datetime(['2024-01-10','2024-02-10','2024-11-10']),
        'ganada':[True,False,True], 'resultado_valido':[True]*3,
        'envio_valido':[True]*3,'fecha_resultado_valida':[True]*3})


def query(date,part='train'):
    return pd.DataFrame({'Codigo':['consulta'],'CodigoProveedor':['p'],
        'CodigoUnidad':['u'],'CodigoProductoONU':['onu'],
        'FechaEnvioOferta':[date],'_particion':[part]})


class HistoryTests(unittest.TestCase):
    def test_same_day_and_future_participation_excluded(self):
        h=m.historical_features(query('2024-02-01'),events(),30)
        self.assertAlmostEqual(h.proveedor_participaciones_log.iloc[0],m.np.log1p(1))

    def test_result_only_after_lag_strictly(self):
        a=m.historical_features(query('2024-02-09'),events(),30)
        b=m.historical_features(query('2024-02-10'),events(),30)
        self.assertEqual(a.proveedor_resultados_log.iloc[0],0)
        self.assertAlmostEqual(b.proveedor_resultados_log.iloc[0],m.np.log1p(1))

    def test_future_labels_never_change_past_features(self):
        e=events()
        a=m.historical_features(query('2024-03-01'),e,30)
        e.loc[e.Codigo.eq('futura'),'ganada']=False
        e.loc[e.Codigo.eq('futura'),'adjudicacion']=pd.Timestamp('2030-01-01')
        assert_frame_equal(a,m.historical_features(query('2024-03-01'),e,30))

    def test_repeated_rows_do_not_multiply_participations(self):
        e=events()
        a=m.historical_features(query('2024-03-01'),e,30)
        assert_frame_equal(a,m.historical_features(query('2024-03-01'),pd.concat([e,e],ignore_index=True),30))

    def test_test_history_frozen_at_october_first(self):
        e=events()
        a=m.historical_features(query('2024-12-01','test'),e,30)
        e.loc[e.Codigo.eq('futura'),'ganada']=False
        assert_frame_equal(a,m.historical_features(query('2024-12-01','test'),e,30))
        self.assertAlmostEqual(a.proveedor_participaciones_log.iloc[0],m.np.log1p(2))

    def test_results_proxy_separated_from_primary_history(self):
        h=m.historical_features(query('2024-03-01'),events(),30)
        self.assertEqual(len([c for c in h if c.endswith('_participaciones_log')]),3)
        self.assertEqual(len(h.columns),12)

    def test_primary_participations_do_not_depend_on_winner_labels(self):
        e=events()
        a=m.historical_features(query('2024-03-01'),e,30)
        e['ganada']=~e.ganada
        e['resultado_valido']=False
        b=m.historical_features(query('2024-03-01'),e,30)
        cols=[c for c in a if c.endswith('_participaciones_log')]
        assert_frame_equal(a[cols],b[cols])

    def test_topk_per_supplier_and_month(self):
        d=pd.DataFrame({'CodigoProveedor':['a']*20+['b']*5,'mes':['2024-10']*25,
            'Codigo':[str(i) for i in range(25)],'Codigoitem':['x']*25,'Correlativo':['1']*25,
            'y':[1]*10+[0]*15,'p':list(range(20,0,-1))+[100]*5})
        r=m.ranking(d,'p')
        self.assertEqual(r[0]['lotes_elegibles'],1)
        self.assertEqual(r[0]['precision_topk'],1)
        self.assertEqual(r[0]['precision_aleatoria_esperada'],.5)
        self.assertEqual(r[1]['precision_topk'],.5)


if __name__=='__main__':
    unittest.main()
