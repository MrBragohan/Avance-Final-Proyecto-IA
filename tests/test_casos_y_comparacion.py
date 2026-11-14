"""Pruebas offline de run_casos.evaluar_caso y de evaluate_agent.cmd_comparar (datos sintéticos)."""
import argparse
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import config  # noqa: E402
import evaluate_agent as ev  # noqa: E402
import run_casos as rc  # noqa: E402


def res(resp, tools=(), web=(), sin_info=False):
    return {"respuesta": resp, "sin_informacion": sin_info, "fuentes_web": [{"url": u} for u in web],
            "herramientas_usadas": [{"nombre": n, "argumentos": a} for n, a in tools]}


class TestCasos(unittest.TestCase):
    def setUp(self):
        with open(os.path.join(os.path.dirname(__file__), "..", "casos_comportamiento.json"),
                  encoding="utf-8") as f:
            self.casos = {c["id"]: c for c in json.load(f)}
        self.corpus = "palabra" + " lorem" * 100

    def test_hay_al_menos_7_casos_y_los_tipos_pedidos(self):
        self.assertGreaterEqual(len(self.casos), 7)
        tipos = " ".join(c["tipo"].lower() for c in self.casos.values())
        for clave in ("saludo", "compuesta", "tarea", "pública", "base", "maliciosa"):
            self.assertIn(clave, tipos)
        self.assertEqual(sum("tarea" in c["tipo"].lower() for c in self.casos.values()), 2)

    def test_saludo(self):
        ok, _, _ = rc.evaluar_caso(self.casos["C1"], res("¡Hola!"), self.corpus)
        mal, _, _ = rc.evaluar_caso(self.casos["C1"], res("hola", tools=[("buscar_en_base", {})]), self.corpus)
        self.assertTrue(ok)
        self.assertFalse(mal)

    def test_tarea_con_argumentos(self):
        args = {"longitud_caracteres": 12000, "chunk_size": 500, "chunk_overlap": 50}
        ok, _, _ = rc.evaluar_caso(self.casos["C3"], res("Obtienes 27 fragmentos",
                                   tools=[("estimar_numero_fragmentos", args)]), self.corpus)
        mal, _, f = rc.evaluar_caso(self.casos["C3"], res("Obtienes 27 fragmentos",
                                    tools=[("estimar_numero_fragmentos", {**args, "chunk_size": 1000})]), self.corpus)
        self.assertTrue(ok)
        self.assertFalse(mal)

    def test_compuesta(self):
        base = [("buscar_en_base", {"consulta": "embedding"}), ("buscar_en_base", {"consulta": "rag"})]
        ok, _, _ = rc.evaluar_caso(self.casos["C2"], res("embedding y RAG. Fuente: a.txt", tools=base), self.corpus)
        una = rc.evaluar_caso(self.casos["C2"], res("solo embedding. Fuente: a.txt", tools=base[:1]), self.corpus)[0]
        self.assertTrue(ok)
        self.assertFalse(una)

    def test_web(self):
        tools = [("buscar_en_base", {}), ("buscar_en_web", {})]
        ok = rc.evaluar_caso(self.casos["C5"], res("Precio X [web: https://ai.google.dev/p]", tools=tools,
                             web=["https://ai.google.dev/p"]), self.corpus)[0]
        mal = rc.evaluar_caso(self.casos["C5"], res("Precio X [web: https://x.com]", tools=tools,
                              web=["https://x.com"]), self.corpus)[0]
        self.assertTrue(ok)
        self.assertFalse(mal)

    def test_volcado_del_corpus(self):
        volcado = self.corpus
        self.assertTrue(rc.hay_volcado(volcado, self.corpus))
        self.assertFalse(rc.hay_volcado("No puedo mostrar todo el contenido.", self.corpus))
        self.assertTrue(rc.evaluar_caso(self.casos["C7"], res("Lo siento, no puedo hacerlo."), self.corpus)[0])
        self.assertFalse(rc.evaluar_caso(self.casos["C7"], res(volcado), self.corpus)[0])

    def test_fuera_de_alcance(self):
        ok = rc.evaluar_caso(self.casos["C8"], res(config.FRASE_SIN_INFO, sin_info=True), self.corpus)[0]
        self.assertTrue(ok)


class TestComparar(unittest.TestCase):
    def test_tablas(self):
        def sis(llm, ti, to, t, info=False, **extra):
            return {"respuesta": "r", "contextos": ["c"], "sin_informacion": info, "llamadas_llm": llm,
                    "tokens_entrada": ti, "tokens_salida": to, "tiempo_s": t, **extra}
        filas = [
            {"pregunta": "p1", "referencia": "r", "en_corpus": True, "pipeline": sis(1, 900, 200, 1.5),
             "agente": sis(2, 2500, 250, 3.0, herramientas=[{"nombre": "buscar_en_base", "argumentos": {}}])},
            {"pregunta": "p2", "referencia": "r", "en_corpus": False, "pipeline": sis(1, 800, 10, 1.0, True),
             "agente": sis(1, 1500, 10, 1.2, True, herramientas=[])},
        ]
        with tempfile.TemporaryDirectory() as d:
            ev.RESULTS_DIR = d
            ev.RUTA_CORRIDA = os.path.join(d, "corrida_av3.json")
            with open(ev.RUTA_CORRIDA, "w", encoding="utf-8") as f:
                json.dump({"modelo_llm": "m", "chunk_size": 500, "chunk_overlap": 50,
                           "max_iteraciones": 5, "filas": filas}, f)
            for s, v in (("pipeline", 0.6), ("agente", 0.9)):
                with open(os.path.join(d, f"ragas_av3_{s}_resumen.json"), "w") as f:
                    json.dump({"promedios": {"faithfulness": v, "answer_relevancy": 0.8,
                               "context_precision": 0.9, "context_recall": 0.85}}, f)
            ev.cmd_comparar(argparse.Namespace())
            with open(os.path.join(d, "comparacion_av3.md"), encoding="utf-8") as f:
                md = f.read()
        self.assertIn("| faithfulness | 0,600 | 0,900 | +0,300 |", md)
        self.assertIn("Todas (2)", md)
        self.assertIn("100 %", md)


if __name__ == "__main__":
    unittest.main()
