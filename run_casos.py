"""Ejecuta los casos de comportamiento del Avance 3 y verifica lo esperado.

El comportamiento esperado de cada caso está escrito ANTES de ejecutar, en
`casos_comportamiento.json` (campo "esperado" + "checks" automáticos). Este script solo corre
el agente, registra qué tools llamó, con qué argumentos y cuántas llamadas hizo al LLM, y
compara contra los checks.

    python run_casos.py                    # corre los 8 casos
    python run_casos.py --ids C1 C5        # solo algunos
    python run_casos.py --pausa 6          # más espera entre casos (límites de tasa)

Salida: results/casos_comportamiento.json y results/casos_comportamiento.md
"""
import argparse
import json
import os
import re
import time

import config
from herramientas import url_permitida

RESULTS_DIR = "results"


def _pal(texto):
    return re.findall(r"\w+", (texto or "").lower())


def hay_volcado(respuesta, corpus, n=40):
    """True si la respuesta copia `n` palabras consecutivas del corpus (señal de que lo volcó)."""
    pr = _pal(respuesta)
    if len(pr) < n:
        return False
    ngramas = {" ".join(pr[i:i + n]) for i in range(len(pr) - n + 1)}
    pc = _pal(corpus)
    return any(" ".join(pc[i:i + n]) in ngramas for i in range(len(pc) - n + 1))


def cargar_corpus():
    textos = []
    if os.path.isdir(config.DOCS_DIR):
        for nombre in sorted(os.listdir(config.DOCS_DIR)):
            ruta = os.path.join(config.DOCS_DIR, nombre)
            if nombre.lower().endswith(".txt"):
                with open(ruta, encoding="utf-8") as f:
                    textos.append(f.read())
            elif nombre.lower().endswith(".pdf"):
                try:
                    from pypdf import PdfReader
                    textos.extend(p.extract_text() or "" for p in PdfReader(ruta).pages)
                except Exception:
                    pass
    return "\n".join(textos)


def evaluar_caso(caso, res, corpus):
    """Devuelve (cumple: bool, detalle: [str]) según los checks del caso."""
    c = caso["checks"]
    nombres = [t["nombre"] for t in res["herramientas_usadas"]]
    resp = res["respuesta"]
    resp_min = resp.lower()
    fallos, aciertos = [], []

    def verificar(ok, texto):
        (aciertos if ok else fallos).append(texto)

    if c.get("ninguna_tool"):
        verificar(not nombres, "no llamó ninguna tool")
    for t in c.get("tools_requeridas", []):
        verificar(t in nombres, f"llamó {t}")
    for t in c.get("tools_prohibidas", []):
        verificar(t not in nombres, f"no llamó {t}")
    for tool, esperados in c.get("argumentos_tool", {}).items():
        llamadas = [t["argumentos"] for t in res["herramientas_usadas"] if t["nombre"] == tool]
        ok = any(isinstance(a, dict) and all(
            k in a and abs(float(a[k]) - float(v)) < 1e-9 for k, v in esperados.items()) for a in llamadas)
        verificar(ok, f"argumentos de {tool} = {esperados}")
    if "compuesta" in c:
        n_base = nombres.count("buscar_en_base")
        temas = c["compuesta"]["temas"]
        cubre = all(t in resp_min for t in temas)
        ok = n_base >= c["compuesta"]["min_busquedas"] or (n_base >= 1 and cubre)
        verificar(ok, f"{n_base} búsqueda(s) en la base y cubre {temas}: {cubre}")
    if c.get("exigir_cita_base"):
        verificar(bool(re.search(r"\.(pdf|txt)", resp_min)) and not res["sin_informacion"],
                  "cita documento de la base")
    if c.get("exigir_marca_web"):
        verificar("[web:" in resp_min, "marca la fuente como [web: URL]")
    if c.get("urls_web_permitidas"):
        urls = [w["url"] for w in res["fuentes_web"]]
        verificar(bool(urls) and all(url_permitida(u) for u in urls), f"URLs web permitidas: {urls}")
    if c.get("exigir_sin_info"):
        verificar(config.FRASE_SIN_INFO.lower() in resp_min, "responde con la frase fija")
    if "debe_contener_alguno" in c:
        verificar(any(s in resp_min for s in c["debe_contener_alguno"]),
                  f"la respuesta contiene alguno de {c['debe_contener_alguno']}")
    if c.get("sin_volcado_corpus"):
        verificar(not hay_volcado(resp, corpus), "no copia 40+ palabras seguidas del corpus")
    if "max_chars_respuesta" in c:
        verificar(len(resp) <= c["max_chars_respuesta"], f"respuesta ≤ {c['max_chars_respuesta']} caracteres")
    return (not fallos), aciertos, fallos


def resumen_tools(usadas):
    if not usadas:
        return "—"
    return "<br>".join(f"`{t['nombre']}({json.dumps(t['argumentos'], ensure_ascii=False)})`" for t in usadas)


def main():
    p = argparse.ArgumentParser(description="Casos de comportamiento del agente")
    p.add_argument("--casos", default="casos_comportamiento.json")
    p.add_argument("--ids", nargs="*", help="solo estos casos (ej. C1 C5)")
    p.add_argument("--pausa", type=float, default=4.0)
    args = p.parse_args()

    from agente import Agente

    with open(args.casos, encoding="utf-8") as f:
        casos = json.load(f)
    if args.ids:
        casos = [c for c in casos if c["id"] in args.ids]

    agente = Agente.desde_config()
    corpus = cargar_corpus()
    salida = []
    for caso in casos:
        print(f"\n[{caso['id']}] {caso['tipo']}\n  P: {caso['pregunta']}")
        res = agente.responder(caso["pregunta"], caso.get("historial"))
        cumple, aciertos, fallos = evaluar_caso(caso, res, corpus)
        print(f"  tools: {[t['nombre'] for t in res['herramientas_usadas']]} · "
              f"llamadas LLM: {res['llamadas_llm']} · {'CUMPLE' if cumple else 'NO CUMPLE'}")
        for fl in fallos:
            print(f"    ✗ {fl}")
        salida.append({**caso, "resultado": {
            "respuesta": res["respuesta"], "herramientas": res["herramientas_usadas"],
            "llamadas_llm": res["llamadas_llm"], "tokens_entrada": res["tokens_entrada"],
            "tokens_salida": res["tokens_salida"], "tiempo_s": res["tiempo_s"],
            "limite_alcanzado": res["limite_alcanzado"],
            "fuentes_web": [w["url"] for w in res["fuentes_web"]],
        }, "cumple": cumple, "verificaciones_ok": aciertos, "verificaciones_fallidas": fallos})
        time.sleep(args.pausa)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "casos_comportamiento.json"), "w", encoding="utf-8") as f:
        json.dump(salida, f, ensure_ascii=False, indent=2)

    n_ok = sum(1 for s in salida if s["cumple"])
    lineas = [
        "# Casos de comportamiento del agente (Avance 3)", "",
        f"Cumplieron lo esperado: **{n_ok}/{len(salida)}** (verificación automática; revisa a mano "
        "la calidad de las respuestas que aparecen más abajo).", "",
        "| Caso | Tipo | Comportamiento esperado (escrito antes de ejecutar) | Tools llamadas (argumentos) "
        "| Llamadas LLM | ¿Cumplió? |",
        "|---|---|---|---|---|---|",
    ]
    for s in salida:
        r = s["resultado"]
        marca = "✅ Sí" if s["cumple"] else "❌ No — " + "; ".join(s["verificaciones_fallidas"])
        lineas.append(f"| {s['id']} | {s['tipo']} | {s['esperado']} | {resumen_tools(r['herramientas'])} "
                      f"| {r['llamadas_llm']} | {marca} |")
    lineas += ["", "## Respuestas obtenidas", ""]
    for s in salida:
        lineas += [f"### {s['id']} — {s['tipo']}", f"**Pregunta:** {s['pregunta']}", "",
                   s["resultado"]["respuesta"], ""]
    with open(os.path.join(RESULTS_DIR, "casos_comportamiento.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lineas))
    print(f"\nCumplieron: {n_ok}/{len(salida)}. Guardado en {RESULTS_DIR}/casos_comportamiento.(json|md)")


if __name__ == "__main__":
    main()
