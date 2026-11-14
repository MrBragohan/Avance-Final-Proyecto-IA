"""Evaluación del AGENTE frente al PIPELINE FIJO del Avance 2 (Avance 3, sección 3).

Usa el MISMO conjunto de evaluación (eval_dataset.json: 20 preguntas, 16 en el corpus + 4 fuera),
el mismo índice, el mismo LLM generador y el mismo juez de Ragas, para que la comparación sea justa.

Tres pasos independientes (así un error 429 de la API no obliga a repetirlo todo):

  1) python evaluate_agent.py correr
        Ejecuta el pipeline fijo y el agente sobre las 20 preguntas y guarda en
        results/corrida_av3.json: respuesta, contextos recuperados, tools llamadas, nº de llamadas
        al LLM, tokens y tiempo de cada sistema.

  2) python evaluate_agent.py ragas [--sistemas pipeline agente] [--metricas faithfulness ...]
        Calcula faithfulness, answer_relevancy, context_precision y context_recall (Ragas) sobre las
        preguntas EN el corpus, a partir de la corrida guardada (no vuelve a llamar al generador).
        Con --metricas recalcula solo esas columnas (útil para completar celdas vacías por 429).

  3) python evaluate_agent.py comparar
        Genera results/comparacion_av3.md (+ .csv y .png): tabla Ragas pipeline vs agente,
        control de rechazo/alucinaciones y comparación de costo (llamadas, tokens, tiempo).

Los retrieved_contexts del agente son los fragmentos que devolvieron sus tools (base y, si las usó,
extractos web).
"""
import argparse
import json
import os
import sys
import time
import unittest.mock as mock

import config

# Parche conocido de ragas 0.4.x con langchain-community >= 0.4 (igual que en evaluate_ragas.py)
sys.modules["langchain_community.chat_models.vertexai"] = mock.MagicMock()
sys.modules["langchain_community.llms.vertexai"] = mock.MagicMock()

METRICAS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
RESULTS_DIR = "results"
RUTA_CORRIDA = os.path.join(RESULTS_DIR, "corrida_av3.json")
SISTEMAS = ["pipeline", "agente"]
NOMBRES = {"pipeline": "Pipeline fijo (Avance 2)", "agente": "Agente (Avance 3)"}


# ---------------------------------------------------------------------------
# 1) CORRER: genera las respuestas de ambos sistemas
# ---------------------------------------------------------------------------
def cmd_correr(args):
    from agente import Agente
    from rag_pipeline import RAGPipeline

    with open(args.dataset, encoding="utf-8") as f:
        items = json.load(f)
    if len(items) < 15:
        print(f"[!] El dataset tiene {len(items)} preguntas; el requisito pide mínimo 15.")

    pipeline = RAGPipeline(reconstruir=args.reconstruir)
    agente = Agente(pipeline.vector_store)          # mismo índice y embeddings; su propio cliente LLM
    print(f"Web del agente: {'activa' if agente.buscador_web else 'DESACTIVADA (sin TAVILY_API_KEY)'}")

    filas = []
    for i, item in enumerate(items, 1):
        pregunta = item["pregunta"]

        # --- pipeline fijo ---
        antes = dict(pipeline.llm.uso)
        t0 = time.perf_counter()
        rp = pipeline.responder(pregunta)
        t_pipe = time.perf_counter() - t0
        despues = pipeline.llm.uso
        pip = {
            "respuesta": rp["respuesta"],
            "contextos": rp["contextos"],
            "sin_informacion": rp["sin_informacion"],
            "llamadas_llm": despues["llamadas"] - antes["llamadas"],
            "tokens_entrada": despues["tokens_entrada"] - antes["tokens_entrada"],
            "tokens_salida": despues["tokens_salida"] - antes["tokens_salida"],
            "tiempo_s": round(t_pipe, 3),
        }
        time.sleep(args.pausa)

        # --- agente ---
        ra = agente.responder(pregunta)
        ag = {
            "respuesta": ra["respuesta"],
            "contextos": ra["contextos"],
            "sin_informacion": ra["sin_informacion"],
            "llamadas_llm": ra["llamadas_llm"],
            "tokens_entrada": ra["tokens_entrada"],
            "tokens_salida": ra["tokens_salida"],
            "tiempo_s": ra["tiempo_s"],
            "herramientas": [{"nombre": t["nombre"], "argumentos": t["argumentos"]}
                             for t in ra["herramientas_usadas"]],
            "uso_web": bool(ra["fuentes_web"]),
            "limite_alcanzado": ra["limite_alcanzado"],
        }
        filas.append({"pregunta": pregunta, "referencia": item["referencia"],
                      "en_corpus": item["en_corpus"], "pipeline": pip, "agente": ag})
        print(f"  [{i:02d}/{len(items)}] {'(en corpus)' if item['en_corpus'] else '(FUERA)   '} "
              f"{pregunta[:52]:<52} pipe: {pip['llamadas_llm']} LLM · agente: {ag['llamadas_llm']} LLM, "
              f"tools={[t['nombre'] for t in ag['herramientas']]}")
        time.sleep(args.pausa)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(RUTA_CORRIDA, "w", encoding="utf-8") as f:
        json.dump({"modelo_llm": config.LLM_MODEL, "chunk_size": config.CHUNK_SIZE,
                   "chunk_overlap": config.CHUNK_OVERLAP, "max_iteraciones": config.MAX_ITERACIONES,
                   "filas": filas}, f, ensure_ascii=False, indent=2)
    print(f"\nCorrida guardada en {RUTA_CORRIDA}. Siguiente paso: python evaluate_agent.py ragas")


# ---------------------------------------------------------------------------
# 2) RAGAS: métricas sobre las preguntas en el corpus
# ---------------------------------------------------------------------------
def cargar_corrida():
    if not os.path.exists(RUTA_CORRIDA):
        raise SystemExit(f"No existe {RUTA_CORRIDA}. Primero ejecuta: python evaluate_agent.py correr")
    with open(RUTA_CORRIDA, encoding="utf-8") as f:
        return json.load(f)


def cmd_ragas(args):
    import pandas as pd
    from ragas import EvaluationDataset, evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import (answer_relevancy, context_precision, context_recall,
                               faithfulness)
    from ragas.run_config import RunConfig

    from ingest import crear_embeddings
    from llm import crear_juez_langchain

    try:                      # la API de Groq solo admite n=1 (ver nota del README del Avance 2)
        answer_relevancy.strictness = 1
    except Exception:
        pass
    objetos = {"faithfulness": faithfulness, "answer_relevancy": answer_relevancy,
               "context_precision": context_precision, "context_recall": context_recall}
    metricas = args.metricas or METRICAS

    corrida = cargar_corrida()
    filas = [f for f in corrida["filas"] if f["en_corpus"]]
    llm_juez = LangchainLLMWrapper(crear_juez_langchain())
    emb_juez = LangchainEmbeddingsWrapper(crear_embeddings())

    for sistema in args.sistemas:
        print(f"\n== Ragas sobre {NOMBRES[sistema]} ({len(filas)} preguntas en el corpus) ==")
        registros = [{"user_input": f["pregunta"], "retrieved_contexts": f[sistema]["contextos"],
                      "response": f[sistema]["respuesta"], "reference": f["referencia"]} for f in filas]
        res = evaluate(
            dataset=EvaluationDataset.from_list(registros),
            metrics=[objetos[m] for m in metricas],
            llm=llm_juez, embeddings=emb_juez,
            run_config=RunConfig(max_workers=2, timeout=240, max_retries=6),
        )
        df = res.to_pandas()

        ruta_csv = os.path.join(RESULTS_DIR, f"ragas_av3_{sistema}.csv")
        if args.metricas and os.path.exists(ruta_csv):      # recálculo parcial: conserva el resto
            previo = pd.read_csv(ruta_csv, encoding="utf-8-sig")
            for m in metricas:
                if m in df.columns:
                    previo[m] = df[m].values
            df = previo
        df.to_csv(ruta_csv, index=False, encoding="utf-8-sig")

        resumen = {"sistema": sistema, "n_preguntas": len(filas), "modelo_llm": corrida["modelo_llm"],
                   "promedios": {}, "valores_validos": {}}
        for m in METRICAS:
            if m in df.columns:
                resumen["promedios"][m] = round(float(df[m].mean()), 4)
                resumen["valores_validos"][m] = int(df[m].notna().sum())
        with open(os.path.join(RESULTS_DIR, f"ragas_av3_{sistema}_resumen.json"), "w",
                  encoding="utf-8") as f:
            json.dump(resumen, f, ensure_ascii=False, indent=2)
        print(json.dumps(resumen["promedios"], ensure_ascii=False))
        for m, v in resumen["valores_validos"].items():
            if v < len(filas):
                print(f"  [!] {m}: solo {v}/{len(filas)} valores válidos (¿error 429?). "
                      f"Recalcula con: python evaluate_agent.py ragas --sistemas {sistema} --metricas {m}")


# ---------------------------------------------------------------------------
# 3) COMPARAR: tablas para el README
# ---------------------------------------------------------------------------
def prom(valores):
    valores = list(valores)
    return sum(valores) / len(valores) if valores else float("nan")


def fmt(x, dec=3):
    return "—" if x is None or x != x else f"{x:.{dec}f}".replace(".", ",")


def pct(x):
    return "—" if x is None or x != x else f"{100 * x:.0f} %"


def delta(a, b, dec=3):
    return "—" if any(v is None or v != v for v in (a, b)) else f"{b - a:+.{dec}f}".replace(".", ",")


def cmd_comparar(args):
    corrida = cargar_corrida()
    filas = corrida["filas"]
    en = [f for f in filas if f["en_corpus"]]
    fuera = [f for f in filas if not f["en_corpus"]]

    resumenes = {}
    for s in SISTEMAS:
        ruta = os.path.join(RESULTS_DIR, f"ragas_av3_{s}_resumen.json")
        resumenes[s] = json.load(open(ruta, encoding="utf-8"))["promedios"] if os.path.exists(ruta) else {}
        if not resumenes[s]:
            print(f"[!] Falta {ruta}: ejecuta 'ragas --sistemas {s}' para llenar esas columnas.")

    L = ["# Comparación: pipeline fijo (Avance 2) vs agente (Avance 3)", "",
         f"LLM: `{corrida['modelo_llm']}` · chunk {corrida['chunk_size']}/{corrida['chunk_overlap']} · "
         f"límite de iteraciones del agente: {corrida['max_iteraciones']} · "
         f"{len(filas)} preguntas ({len(en)} en el corpus + {len(fuera)} fuera).", "",
         "## 1. Métricas de Ragas (preguntas en el corpus)", "",
         "| Métrica | Pipeline fijo | Agente | Variación |", "|---|---|---|---|"]
    filas_csv = []
    for m in METRICAS:
        a, b = resumenes["pipeline"].get(m), resumenes["agente"].get(m)
        L.append(f"| {m} | {fmt(a)} | {fmt(b)} | {delta(a, b)} |")
        filas_csv.append({"metrica": m, "pipeline": a, "agente": b})

    # --- control de alucinaciones ---
    def rechazo_correcto(sistema):
        return prom(1.0 if f[sistema]["sin_informacion"] else 0.0 for f in fuera)

    def falso_rechazo(sistema):
        return prom(1.0 if f[sistema]["sin_informacion"] else 0.0 for f in en)

    web_fuera = prom(1.0 if f["agente"].get("uso_web") else 0.0 for f in fuera)
    L += ["", "## 2. Control de alucinaciones y uso de la web", "",
          "| Indicador | Pipeline fijo | Agente |", "|---|---|---|",
          f"| Rechazo con la frase fija en las {len(fuera)} preguntas fuera del corpus "
          f"| {pct(rechazo_correcto('pipeline'))} | {pct(rechazo_correcto('agente'))} |",
          f"| Falso rechazo en las {len(en)} preguntas del corpus "
          f"| {pct(falso_rechazo('pipeline'))} | {pct(falso_rechazo('agente'))} |",
          f"| Preguntas fuera del corpus que el agente respondió con la web | — | {pct(web_fuera)} |",
          "", "_Si el agente respondió con la web una pregunta fuera del corpus (p. ej. el precio de la "
          "API de Gemini), no es un error: la política lo permite. Revisa a mano esas respuestas._"]

    # --- costo ---
    L += ["", "## 3. Costo (mismas preguntas)", "",
          "| Grupo | Sistema | Llamadas al LLM (prom.) | Tokens de entrada (prom.) | Tokens de salida (prom.) "
          "| Tiempo (s, prom.) |", "|---|---|---|---|---|---|"]
    for etiqueta, grupo in (("Todas", filas), ("En corpus", en), ("Fuera de corpus", fuera)):
        for s in SISTEMAS:
            L.append(f"| {etiqueta} ({len(grupo)}) | {NOMBRES[s]} "
                     f"| {fmt(prom(f[s]['llamadas_llm'] for f in grupo), 2)} "
                     f"| {fmt(prom(f[s]['tokens_entrada'] for f in grupo), 0)} "
                     f"| {fmt(prom(f[s]['tokens_salida'] for f in grupo), 0)} "
                     f"| {fmt(prom(f[s]['tiempo_s'] for f in grupo), 2)} |")
    L += ["", "## 4. Uso de herramientas del agente", "",
          f"- Promedio de tools por pregunta: {fmt(prom(len(f['agente']['herramientas']) for f in filas), 2)}",
          f"- Preguntas sin ninguna tool: {sum(1 for f in filas if not f['agente']['herramientas'])}/{len(filas)}",
          f"- Preguntas con límite de iteraciones alcanzado: "
          f"{sum(1 for f in filas if f['agente'].get('limite_alcanzado'))}/{len(filas)}", ""]

    detalle = ["## 5. Detalle por pregunta (agente)", "",
               "| # | Pregunta | Tools | Llamadas LLM | Tokens (ent./sal.) | Tiempo (s) |", "|---|---|---|---|---|---|"]
    for i, f in enumerate(filas, 1):
        a = f["agente"]
        tools = ", ".join(t["nombre"] for t in a["herramientas"]) or "—"
        detalle.append(f"| {i} | {f['pregunta'][:60]} | {tools} | {a['llamadas_llm']} "
                       f"| {a['tokens_entrada']}/{a['tokens_salida']} | {a['tiempo_s']} |")
    L += detalle

    with open(os.path.join(RESULTS_DIR, "comparacion_av3.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    import pandas as pd
    pd.DataFrame(filas_csv).to_csv(os.path.join(RESULTS_DIR, "comparacion_av3.csv"), index=False)
    print("\n".join(L))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        validas = [m for m in METRICAS if resumenes["pipeline"].get(m) is not None
                   and resumenes["agente"].get(m) is not None]
        if validas:
            x, ancho = range(len(validas)), 0.38
            fig, ax = plt.subplots(figsize=(9, 5))
            ax.bar([i - ancho / 2 for i in x], [resumenes["pipeline"][m] for m in validas], ancho,
                   label=NOMBRES["pipeline"])
            ax.bar([i + ancho / 2 for i in x], [resumenes["agente"][m] for m in validas], ancho,
                   label=NOMBRES["agente"])
            ax.set_xticks(list(x))
            ax.set_xticklabels(validas, rotation=15)
            ax.set_ylim(0, 1.05)
            ax.set_ylabel("Puntaje (0-1)")
            ax.set_title("Ragas: pipeline fijo vs agente")
            ax.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(RESULTS_DIR, "comparacion_av3.png"), dpi=150)
            print(f"\nGráfico: {RESULTS_DIR}/comparacion_av3.png")
    except ImportError:
        pass


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Evaluación del agente vs pipeline fijo")
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("correr", help="ejecuta pipeline y agente sobre el dataset")
    c.add_argument("--dataset", default="eval_dataset.json")
    c.add_argument("--pausa", type=float, default=4.0, help="segundos entre llamadas (sube si ves 429)")
    c.add_argument("--reconstruir", action="store_true", help="reconstruye el índice vectorial")
    c.set_defaults(fn=cmd_correr)

    r = sub.add_parser("ragas", help="calcula las métricas de Ragas de la corrida guardada")
    r.add_argument("--sistemas", nargs="+", choices=SISTEMAS, default=SISTEMAS)
    r.add_argument("--metricas", nargs="+", choices=METRICAS, help="recalcula solo estas métricas")
    r.set_defaults(fn=cmd_ragas)

    k = sub.add_parser("comparar", help="genera las tablas de comparación")
    k.set_defaults(fn=cmd_comparar)

    args = p.parse_args()
    args.fn(args)
