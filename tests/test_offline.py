"""Pruebas SIN red y SIN claves: verifican las reglas que viven en el código.

Se ejecutan con:   python -m unittest discover -s tests -v

Usan un LLM falso (guion de respuestas), un índice falso y un buscador web falso, por lo que
no consumen tokens ni requieren GROQ_API_KEY / TAVILY_API_KEY.
"""
import json
import os
import sys
import tempfile
import types
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import config  # noqa: E402
import agente as ag  # noqa: E402
from herramientas import (EstadoTurno, Recursos, construir_herramientas,  # noqa: E402
                          url_permitida)


# ----------------------------------------------------------------------------- dobles de prueba
class Doc:
    """Doble de langchain Document: Doc("texto", source=...) o Doc(page_content=..., metadata=...)."""

    def __init__(self, texto=None, page_content=None, metadata=None, **meta):
        self.page_content = page_content if page_content is not None else texto
        self.metadata = dict(metadata) if metadata is not None else meta


class IndiceFalso:
    """Imita a Chroma: devuelve SIEMPRE más fragmentos de los que pide la tool."""

    def __init__(self, docs):
        self.docs, self.ultimo_k = docs, None

    def similarity_search_with_relevance_scores(self, consulta, k=4):
        self.ultimo_k = k
        return [(d, 0.9 - 0.01 * i) for i, d in enumerate(self.docs)]  # ignora k a propósito


class WebFalsa:
    def __init__(self, resultados):
        self.resultados, self.llamadas = resultados, []

    def buscar(self, consulta, max_resultados, dominios):
        self.llamadas.append((consulta, max_resultados, list(dominios)))
        return self.resultados


class LLMGuionado:
    """Devuelve respuestas predefinidas y registra con qué tools se le llamó."""

    def __init__(self, guion):
        self.guion, self.llamadas = list(guion), []

    def chat_tools(self, mensajes, sistema=None, tools=None, temperatura=0.2, max_tokens=4096):
        self.llamadas.append({"n_mensajes": len(mensajes), "tools": tools})
        paso = self.guion.pop(0)
        return {"contenido": paso.get("texto", ""), "tool_calls": paso.get("tools", []),
                "tokens_entrada": 100, "tokens_salida": 20}


def llamada(nombre, args, id_="c1"):
    return {"id": id_, "nombre": nombre, "argumentos": json.dumps(args)}


def docs_largos(n=10):
    return [Doc("palabra " * 300 + f"#{i}", source="guia.pdf", page=i + 1, chunk_id=i) for i in range(n)]


def kit(docs=None, web=None):
    estado = EstadoTurno()
    indice = IndiceFalso(docs if docs is not None else docs_largos())
    return construir_herramientas(Recursos(indice, web), estado), estado, indice


# ----------------------------------------------------------------------------- tool: base
class TestBuscarEnBase(unittest.TestCase):
    def test_limite_de_fragmentos_y_longitud_en_codigo(self):
        h, estado, indice = kit()
        salida = h["buscar_en_base"].invoke({"consulta": "embeddings"})
        self.assertEqual(salida.count("<fragmento "), config.MAX_FRAGMENTOS_TOOL)
        self.assertEqual(indice.ultimo_k, config.MAX_FRAGMENTOS_TOOL)
        for f in estado.fuentes_base:
            self.assertLessEqual(len(f["texto"]), config.MAX_CHARS_FRAGMENTO + 1)

    def test_cada_fragmento_trae_documento_y_pagina_o_seccion(self):
        docs = [Doc("texto A", source="guia.pdf", page=3, chunk_id=1),
                Doc("texto B", source="apuntes.txt", seccion="Prompt Engineering", chunk_id=2)]
        h, _, _ = kit(docs)
        salida = h["buscar_en_base"].invoke({"consulta": "x"})
        self.assertIn("guia.pdf, pág. 3", salida)
        self.assertIn("apuntes.txt, sección «Prompt Engineering»", salida)

    def test_el_llm_no_puede_pedir_mas_fragmentos(self):
        h, _, _ = kit()
        salida = h["buscar_en_base"].invoke({"consulta": "x", "k": 100})
        self.assertTrue(salida.startswith("ERROR"))          # parámetro inexistente → error, no se obedece

    def test_consulta_vacia_o_larga(self):
        h, _, _ = kit()
        self.assertTrue(h["buscar_en_base"].invoke({"consulta": "  "}).startswith("ERROR"))
        self.assertTrue(h["buscar_en_base"].invoke({"consulta": "a" * 1000}).startswith("ERROR"))

    def test_docstring_dice_que_hace_y_cuando_usarla(self):
        d = construir_herramientas(Recursos(IndiceFalso([]), None), EstadoTurno())
        for nombre, h in d.items():
            self.assertGreater(len(h.descripcion), 150, nombre)
        self.assertIn("SIEMPRE", d["buscar_en_base"].descripcion)
        self.assertIn("SOLO", d["buscar_en_web"].descripcion)


# ----------------------------------------------------------------------------- tool: web
class TestBuscarEnWeb(unittest.TestCase):
    def resultados(self):
        return [
            {"titulo": "Pricing", "url": "https://ai.google.dev/gemini-api/docs/pricing", "extracto": "x " * 400},
            {"titulo": "Otro", "url": "https://sitio-malo.com/a", "extracto": "no permitido"},
            {"titulo": "Docs", "url": "https://console.groq.com/docs", "extracto": "ok"},
            {"titulo": "Docs2", "url": "https://openai.com/api/pricing", "extracto": "ok"},
            {"titulo": "Docs3", "url": "https://arxiv.org/abs/1", "extracto": "ok"},
        ]

    def test_primero_la_base(self):
        h, _, _ = kit(web=WebFalsa(self.resultados()))
        self.assertTrue(h["buscar_en_web"].invoke({"consulta": "precio gemini"}).startswith("BLOQUEADO"))

    def test_lista_fija_maximo_3_extracto_corto_y_origen_web(self):
        web = WebFalsa(self.resultados())
        h, estado, _ = kit(web=web)
        h["buscar_en_base"].invoke({"consulta": "x"})
        salida = h["buscar_en_web"].invoke({"consulta": "precio gemini"})
        self.assertEqual(web.llamadas[0][2], config.SITIOS_WEB_PERMITIDOS)    # lista del código
        self.assertEqual(web.llamadas[0][1], config.MAX_RESULTADOS_WEB)
        self.assertEqual(salida.count("<resultado "), 3)
        self.assertNotIn("sitio-malo.com", salida)                             # 2.ª barrera
        self.assertEqual(salida.count('origen="web"'), 3 + 1)                  # 3 resultados + cabecera
        for f in estado.fuentes_web:
            self.assertLessEqual(len(f["extracto"]), config.MAX_CHARS_EXTRACTO_WEB + 1)

    def test_el_llm_no_puede_cambiar_los_sitios(self):
        h, _, _ = kit(web=WebFalsa([]))
        h["buscar_en_base"].invoke({"consulta": "x"})
        salida = h["buscar_en_web"].invoke({"consulta": "x", "include_domains": ["evil.com"]})
        self.assertTrue(salida.startswith("ERROR"))

    def test_bloquea_texto_de_la_base_en_la_consulta_web(self):
        texto_base = "Un embedding es una representación numérica de un texto imagen u otro dato en un espacio"
        h, _, _ = kit([Doc(texto_base, source="a.txt", chunk_id=1)], web=WebFalsa([]))
        h["buscar_en_base"].invoke({"consulta": "embedding"})
        salida = h["buscar_en_web"].invoke(
            {"consulta": "una representación numérica de un texto imagen u otro dato"})
        self.assertTrue(salida.startswith("BLOQUEADO"))
        ok = h["buscar_en_web"].invoke({"consulta": "precio api gemini por millón de tokens"})
        self.assertFalse(ok.startswith("BLOQUEADO"))

    def test_sin_clave_no_rompe(self):
        h, _, _ = kit(web=None)
        h["buscar_en_base"].invoke({"consulta": "x"})
        self.assertTrue(h["buscar_en_web"].invoke({"consulta": "x"}).startswith("NO_DISPONIBLE"))

    def test_url_permitida(self):
        self.assertTrue(url_permitida("https://ai.google.dev/x"))
        self.assertTrue(url_permitida("https://docs.anthropic.com/x"))
        self.assertFalse(url_permitida("https://ai.google.dev.evil.com/x"))
        self.assertFalse(url_permitida("https://notopenai.com/x"))


# ----------------------------------------------------------------------------- tools del dominio
class TestToolsDominio(unittest.TestCase):
    def test_fragmentos(self):
        h, _, _ = kit()
        s = h["estimar_numero_fragmentos"].invoke(
            {"longitud_caracteres": 12000, "chunk_size": 500, "chunk_overlap": 50})
        self.assertIn("27 fragmentos", s)                         # 1 + ceil(11500/450) = 27
        self.assertIn("1 fragmentos", h["estimar_numero_fragmentos"].invoke({"longitud_caracteres": 300}))
        self.assertIn("ERROR", h["estimar_numero_fragmentos"].invoke(
            {"longitud_caracteres": 100, "chunk_size": 50, "chunk_overlap": 50}))

    def test_costo(self):
        h, _, _ = kit()
        s = h["estimar_costo_llm"].invoke({
            "tokens_entrada": 2_000_000, "tokens_salida": 500_000,
            "precio_entrada_por_millon": 0.15, "precio_salida_por_millon": 0.60})
        self.assertIn("Costo total = 0.6000 USD", s)
        s2 = h["estimar_costo_llm"].invoke({
            "tokens_entrada": "1000", "tokens_salida": "500",           # el LLM a veces manda strings
            "precio_entrada_por_millon": "1", "precio_salida_por_millon": "2", "numero_solicitudes": 10})
        self.assertIn("Costo total = 0.0200 USD", s2)
        self.assertIn("ERROR", h["estimar_costo_llm"].invoke({"tokens_entrada": 1}))   # faltan datos


# ----------------------------------------------------------------------------- bucle del agente
class TestBucle(unittest.TestCase):
    def agente(self, guion, web=None, max_it=5, log=None):
        llm = LLMGuionado(guion)
        a = ag.Agente(IndiceFalso(docs_largos()), llm=llm, buscador_web=web,
                      max_iteraciones=max_it, log_path=log if log is not None else "")
        return a, llm

    def test_saludo_sin_tools(self):
        a, llm = self.agente([{"texto": "¡Hola! ¿Qué tema repasamos?"}])
        r = a.responder("Hola, buenos días")
        self.assertEqual(r["herramientas_usadas"], [])
        self.assertEqual(r["llamadas_llm"], 1)
        self.assertEqual(len(llm.llamadas[0]["tools"]), 4)          # tenía tools disponibles y no las usó

    def test_pregunta_compuesta_dos_busquedas(self):
        a, _ = self.agente([
            {"tools": [llamada("buscar_en_base", {"consulta": "embedding"}, "a"),
                       llamada("buscar_en_base", {"consulta": "RAG"}, "b")]},
            {"texto": "## Explicación ... ## Fuente - Base: guia.pdf, pág. 1"},
        ])
        r = a.responder("¿Qué es un embedding y qué es RAG?")
        self.assertEqual([t["nombre"] for t in r["herramientas_usadas"]], ["buscar_en_base"] * 2)
        self.assertEqual(r["llamadas_llm"], 2)
        self.assertEqual(r["tokens_entrada"], 200)

    def test_tool_de_dominio_con_argumentos_del_llm(self):
        a, llm = self.agente([
            {"tools": [llamada("estimar_numero_fragmentos",
                               {"longitud_caracteres": 12000, "chunk_size": 500, "chunk_overlap": 50})]},
            {"texto": "Obtienes 27 fragmentos."},
        ])
        r = a.responder("¿cuántos chunks salen de 12000 caracteres?")
        self.assertEqual(r["herramientas_usadas"][0]["argumentos"]["longitud_caracteres"], 12000)
        self.assertIn("27 fragmentos", r["herramientas_usadas"][0]["resultado"])

    def test_limite_de_iteraciones(self):
        # El LLM insiste en buscar siempre; en la última vuelta no recibe tools y debe responder.
        guion = [{"tools": [llamada("buscar_en_base", {"consulta": f"q{i}"}, f"c{i}")]} for i in range(3)]
        guion.append({"texto": "Respuesta final con lo que hay."})
        a, llm = self.agente(guion, max_it=4)
        r = a.responder("pregunta que nunca termina")
        self.assertEqual(r["llamadas_llm"], 4)
        self.assertIsNone(llm.llamadas[-1]["tools"])                 # última vuelta sin tools
        self.assertTrue(r["limite_alcanzado"])
        self.assertEqual(r["respuesta"], "Respuesta final con lo que hay.")

    def test_limite_si_el_llm_sigue_pidiendo_tools_sin_texto(self):
        guion = [{"tools": [llamada("buscar_en_base", {"consulta": "q"}, f"c{i}")]} for i in range(3)]
        a, _ = self.agente(guion, max_it=3)
        r = a.responder("x")
        self.assertEqual(r["llamadas_llm"], 3)
        self.assertEqual(r["respuesta"], config.FRASE_SIN_INFO)
        self.assertTrue(r["sin_informacion"] and r["limite_alcanzado"])

    def test_tool_inexistente_y_json_malo_no_rompen(self):
        a, _ = self.agente([
            {"tools": [{"id": "a", "nombre": "borrar_todo", "argumentos": "{}"},
                       {"id": "b", "nombre": "buscar_en_base", "argumentos": "{no es json"}]},
            {"texto": "Lo siento, no pude."},
        ])
        r = a.responder("x")
        self.assertIn("no existe", r["herramientas_usadas"][0]["resultado"])
        self.assertIn("JSON", r["herramientas_usadas"][1]["resultado"])

    def test_web_sin_base_primero_se_bloquea_dentro_del_bucle(self):
        web = WebFalsa([{"titulo": "t", "url": "https://openai.com/x", "extracto": "e"}])
        a, _ = self.agente([
            {"tools": [llamada("buscar_en_web", {"consulta": "precio"})]},
            {"texto": config.FRASE_SIN_INFO},
        ], web=web)
        r = a.responder("¿precio actual?")
        self.assertTrue(r["herramientas_usadas"][0]["resultado"].startswith("BLOQUEADO"))
        self.assertEqual(web.llamadas, [])

    def test_fuentes_y_contextos_vienen_de_las_tools(self):
        web = WebFalsa([{"titulo": "Pricing", "url": "https://ai.google.dev/p", "extracto": "USD 0.30"}])
        a, _ = self.agente([
            {"tools": [llamada("buscar_en_base", {"consulta": "precio"}, "a")]},
            {"tools": [llamada("buscar_en_web", {"consulta": "precio gemini api"}, "b")]},
            {"texto": "Desde la web: ... [web: https://ai.google.dev/p]"},
        ], web=web)
        r = a.responder("¿precio actual de la API de Gemini?")
        self.assertEqual(len(r["fuentes_base"]), config.MAX_FRAGMENTOS_TOOL)
        self.assertEqual(r["fuentes_web"][0]["origen"], "web")
        self.assertEqual(len(r["contextos"]), config.MAX_FRAGMENTOS_TOOL + 1)

    def test_historial_se_pasa_como_texto(self):
        a, llm = self.agente([{"texto": "ok"}])
        a.responder("¿y eso?", historial=[{"role": "user", "content": "hola"},
                                           {"role": "assistant", "content": "hola!"}])
        self.assertEqual(llm.llamadas[0]["n_mensajes"], 3)

    def test_log_de_uso_de_tools(self):
        with tempfile.TemporaryDirectory() as d:
            ruta = os.path.join(d, "logs", "uso.jsonl")
            a, _ = self.agente([
                {"tools": [llamada("buscar_en_base", {"consulta": "rag"})]}, {"texto": "listo"}], log=ruta)
            a.responder("¿qué es rag?")
            a2, _ = self.agente([{"texto": "hola"}], log=ruta)
            a2.responder("hola")
            with open(ruta, encoding="utf-8") as f:
                lineas = [json.loads(l) for l in f]
        self.assertEqual(len(lineas), 2)
        self.assertEqual(lineas[0]["herramientas"][0]["nombre"], "buscar_en_base")
        self.assertEqual(lineas[0]["herramientas"][0]["argumentos"], {"consulta": "rag"})
        self.assertEqual(lineas[0]["llamadas_llm"], 2)
        self.assertEqual(lineas[1]["herramientas"], [])


# ----------------------------------------------------------------------------- ingesta (txt por tema)
class TestIngestaPorTema(unittest.TestCase):
    def test_separar_por_tema(self):
        # langchain no se necesita para esta prueba: se sustituye el módulo Document por un doble
        for nombre in ("langchain_core", "langchain_core.documents", "langchain_text_splitters"):
            sys.modules.setdefault(nombre, types.ModuleType(nombre))
        sys.modules["langchain_core.documents"].Document = Doc
        sys.modules["langchain_text_splitters"].RecursiveCharacterTextSplitter = object
        import importlib
        ingest = importlib.import_module("ingest")
        crudo = Doc("[TEMA: Uno]\nTexto uno.\n\n[TEMA: Dos (x)]\nTexto dos.", source="k.txt")
        sal = ingest._separar_por_tema([crudo])
        self.assertEqual([d.metadata["seccion"] for d in sal], ["Uno", "Dos (x)"])
        self.assertTrue(sal[0].page_content.startswith("[TEMA: Uno]"))
        self.assertNotIn("Texto dos", sal[0].page_content)
        sin_tema = ingest._separar_por_tema([Doc("sin encabezados")])
        self.assertEqual(len(sin_tema), 1)


if __name__ == "__main__":
    unittest.main()
