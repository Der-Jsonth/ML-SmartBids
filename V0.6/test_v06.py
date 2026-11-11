"""Pruebas de invariantes de V0.6."""
from __future__ import annotations
import json
from pathlib import Path
import sys
import unittest
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
import preparacion as p
import modelo as m

class V06Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.path=p.DATOS/"cohorte.parquet"
        cls.d=pd.read_parquet(cls.path).head(5000)

    def test_se_conservan_todas_las_filas(self):
        audit=json.loads((p.DATOS/"auditoria_cohorte.json").read_text(encoding="utf-8"))
        self.assertEqual(audit["filas_csv"],2_846_181)
        self.assertEqual(audit["filas_cohorte"],2_846_181)
        self.assertEqual(audit["filas_eliminadas"],0)
        self.assertEqual(audit["filas_unidas"],0)

    def test_identidad_fisica_es_unica(self):
        info=json.loads((p.ROOT/"datos"/"fuentes.json").read_text(encoding="utf-8"))
        for entry in info["archivos"]:
            q=p.ROOT/"datos"/"juntado"/entry["parquet"]
            z=pd.read_parquet(q,columns=["_archivo","_registro_archivo"])
            self.assertFalse(z.duplicated().any())

    def test_auditoria_no_modifica_filas(self):
        before=self.d[["_archivo","_registro_archivo"]].copy()
        report=p.auditar_identidad(self.d)
        self.assertEqual(len(self.d),5000)
        pd.testing.assert_frame_equal(before,self.d[["_archivo","_registro_archivo"]])
        self.assertIn("identidad_fisica",set(report.clave))

    def test_licitacion_no_cruza_particiones(self):
        full=pd.read_parquet(self.path,columns=["CodigoExterno","_particion"])
        self.assertFalse(full.groupby("CodigoExterno",observed=True)._particion.nunique().gt(1).any())

    def test_item_real_distingue_lineas(self):
        d=self.d.iloc[:2].copy()
        d["CodigoExterno"]="X"; d["CodigoProductoONU"]=1; d["Cantidad"]=1
        d["Codigoitem"]=[10,11]
        self.assertEqual(m.clave_linea(d,"proxy").nunique(),1)
        self.assertEqual(m.clave_linea(d,"item").nunique(),2)

    def test_resultados_no_entran_a_variables(self):
        spec=next(s for s in m.ESPECIFICACIONES if s["nombre"]=="cb_completo")
        a=m.variables(self.d,spec)
        changed=self.d.copy(); changed["Seleccionada"]=1-changed["Seleccionada"]
        changed["Estado"]="valor_inventado"; changed["Estado Oferta"]="valor_inventado"
        b=m.variables(changed,spec)
        pd.testing.assert_frame_equal(a,b)

    def test_identidad_fisica_no_es_predictor(self):
        for spec in m.ESPECIFICACIONES:
            cols=set(m.variables(self.d,spec).columns)
            self.assertTrue({"_archivo","_registro_archivo","_id_fila"}.isdisjoint(cols))

    def test_modelo_completo_usa_proveedor_e_historia(self):
        spec=next(s for s in m.ESPECIFICACIONES if s["nombre"]=="cb_completo")
        cols=set(m.variables(self.d,spec))
        self.assertIn("CodigoProveedor",cols)
        self.assertIn("CodigoSucursalProveedor",cols)
        self.assertIn("hist_proveedor_participaciones_log",cols)
        self.assertIn("nombres_oferta_linea",cols)

    def test_umbral_f1_es_un_score_observado(self):
        y=np.array([0,0,1,1]); score=np.array([.1,.3,.4,.9]); threshold=m.umbral_f1(y,score)
        self.assertIn(threshold,score)
        self.assertGreater(m.metricas(y,score,threshold)["f1"],0)

    def test_fuentes_y_cohorte_tienen_hash(self):
        provenance=json.loads((p.DATOS/"preparacion.json").read_text(encoding="utf-8"))
        self.assertEqual(provenance["cohorte_sha256"],p.sha256(self.path))
        self.assertEqual(provenance["fuentes_sha256"],p.sha256(p.ROOT/"datos"/"fuentes.json"))

    def test_comparacion_ampliada_selecciona_solo_por_validacion(self):
        pointer=p.ROOT/"resultados_ampliados"/"ultima_ejecucion.json"
        if not pointer.exists(): self.skipTest("Comparación ampliada aún no ejecutada")
        run=json.loads(pointer.read_text(encoding="utf-8"))["directorio"]
        out=p.ROOT/"resultados_ampliados"/run
        decision=json.loads((out/"seleccion_previa_test.json").read_text(encoding="utf-8"))
        val=pd.read_csv(out/"validacion_ampliada.csv")
        expected=val.sort_values(["f1","average_precision"],ascending=False,kind="stable").iloc[0].candidato
        self.assertEqual(decision["elegido"],expected)
        self.assertEqual({"sgd_logistico","lightgbm_item","xgboost_item"}-set(val.candidato),set())

if __name__=="__main__":
    unittest.main(verbosity=2)
