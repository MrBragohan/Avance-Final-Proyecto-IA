"""Evaluación del pipeline RAG con Ragas.

Métricas Ragas: faithfulness, answer_relevancy, context_precision, context_recall.
Métrica propia: tasa de rechazo correcto (control de alucinaciones en preguntas fuera del corpus).

Ejemplos:
    # 1) Línea base
    python evaluate_ragas.py --tag baseline --chunk-size 500 --overlap 50 --top-k 4

    # 2) Iteración de mejora (cambia UN parámetro, o los que justifiques)
    python evaluate_ragas.py --tag mejora --chunk-size 1000 --overlap 150 --top-k 4

    # 3) Comparar antes / después (tabla + gráfico)
    python evaluate_ragas.py --comparar baseline mejora
"""
import argparse
import json
import os
import sys
import time
import unittest.mock as mock

import pandas as pd

import config

# --- Parche para un bug conocido de ragas 0.4.x con langchain-community >= 0.4 ---
# ragas importa clases de VertexAI que ya no existen en langchain-community y falla al
# importar con "No module named 'langchain_community.chat_models.vertexai'".
# No usamos VertexAI, así que se reemplazan esos módulos por objetos vacíos ANTES de importar ragas.
sys.modules["langchain_community.chat_models.vertexai"] = mock.MagicMock()
sys.modules["langchain_community.llms.vertexai"] = mock.MagicMock()

METRICAS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
RESULTS_DIR = "results"


# ---------------------------------------------------------------------------
def ejecutar_pipeline(pipeline, items, pausa):
    """Corre el RAG sobre cada pregunta (sin historial) y arma los registros para Ragas."""
    registros, extras = [], []
    for i, item in enumerate(items, 1):
        res = pipeline.responder(item["pregunta"])
        registros.append(
            {
                "user_input": item["pregunta"],
                "retrieved_contexts": res["contextos"],
                "response": res["respuesta"],
                "reference": item["referencia"],
            }
        )
        extras.append(
            {"en_corpus": item["en_corpus"], "sin_informacion": res["sin_informacion"]}
        )
        print(f"  [{i:02d}/{len(items)}] {'(en corpus)' if item['en_corpus'] else '(FUERA)   '} "
              f"{item['pregunta'][:60]}")
        time.sleep(pausa)  # evita límites de tasa de la API
    return registros, extras


def resumir(df, tag, args, n):
    """Promedios globales, solo en-corpus y tasa de rechazo correcto."""
    cols = [c for c in METRICAS if c in df.columns]
    en = df[df["en_corpus"]]
    fuera = df[~df["en_corpus"]]
    return {
        "tag": tag,
        "chunk_size": args.chunk_size,
        "chunk_overlap": args.overlap,
        "top_k": args.top_k,
        "n_preguntas": n,
        "global": {c: round(float(df[c].mean()), 4) for c in cols},
        "en_corpus": {c: round(float(en[c].mean()), 4) for c in cols},
        "tasa_rechazo_correcto": round(float(fuera["sin_informacion"].mean()), 4)
        if len(fuera) else None,
        "tasa_falso_rechazo": round(float(en["sin_informacion"].mean()), 4)
        if len(en) else None,
    }


def evaluar(args):
    from ragas import EvaluationDataset, evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import (answer_relevancy, context_precision, context_recall,
                               faithfulness)
    from ragas.run_config import RunConfig

    from llm import crear_juez_langchain
    from rag_pipeline import RAGPipeline

    with open(args.dataset, encoding="utf-8") as f:
        items = json.load(f)
    if len(items) < 15:
        print(f"[!] El dataset tiene {len(items)} preguntas; el requisito pide mínimo 15.")

    print(f"\n== Configuración '{args.tag}': chunk={args.chunk_size} overlap={args.overlap} "
          f"top_k={args.top_k} ==")
    pipeline = RAGPipeline(args.chunk_size, args.overlap, args.top_k, args.reconstruir)

    print("\nEjecutando el pipeline RAG sobre el conjunto de evaluación...")
    registros, extras = ejecutar_pipeline(pipeline, items, args.pausa)

    # LLM "juez" (el mismo proveedor configurado) y embeddings locales para answer_relevancy
    llm_juez = LangchainLLMWrapper(crear_juez_langchain())
    emb_juez = LangchainEmbeddingsWrapper(pipeline.embeddings)

    print("\nEvaluando con Ragas (consume llamadas adicionales al LLM, puede tardar)...")
    resultados = evaluate(
        dataset=EvaluationDataset.from_list(registros),
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=llm_juez,
        embeddings=emb_juez,
        run_config=RunConfig(max_workers=2, timeout=240, max_retries=6),
    )

    df = resultados.to_pandas()
    df["en_corpus"] = [e["en_corpus"] for e in extras]
    df["sin_informacion"] = [e["sin_informacion"] for e in extras]

    os.makedirs(RESULTS_DIR, exist_ok=True)
    df.to_csv(os.path.join(RESULTS_DIR, f"ragas_{args.tag}.csv"), index=False, encoding="utf-8-sig")

    resumen = resumir(df, args.tag, args, len(items))
    with open(os.path.join(RESULTS_DIR, f"ragas_{args.tag}_resumen.json"), "w",
              encoding="utf-8") as f:
        json.dump(resumen, f, ensure_ascii=False, indent=2)

    mostrar_resumen(resumen, df)


def mostrar_resumen(resumen, df):
    cols = [c for c in METRICAS if c in df.columns]
    print("\n" + "=" * 70)
    print(f"RESULTADOS — {resumen['tag']}  (chunk={resumen['chunk_size']}, "
          f"overlap={resumen['chunk_overlap']}, top_k={resumen['top_k']})")
    print("=" * 70)
    detalle = df[["user_input"] + cols].copy()
    detalle["user_input"] = detalle["user_input"].str[:45]
    print(detalle.round(3).to_string(index=False))
    print("\nPromedio GLOBAL        :", resumen["global"])
    print("Promedio SOLO en corpus:", resumen["en_corpus"])
    print(f"Rechazo correcto fuera del corpus: {resumen['tasa_rechazo_correcto']}")
    print(f"Falsos rechazos dentro del corpus : {resumen['tasa_falso_rechazo']}")
    print(f"\nArchivos guardados en {RESULTS_DIR}/")


# ---------------------------------------------------------------------------
def comparar(tag_a, tag_b):
    """Tabla y gráfico antes/después a partir de los resúmenes guardados."""
    resumenes = []
    for tag in (tag_a, tag_b):
        ruta = os.path.join(RESULTS_DIR, f"ragas_{tag}_resumen.json")
        if not os.path.exists(ruta):
            raise SystemExit(
                f"No existe {ruta}. Primero ejecuta la evaluación con: "
                f"python evaluate_ragas.py --tag {tag} ..."
            )
        with open(ruta, encoding="utf-8") as f:
            resumenes.append(json.load(f))
    a, b = resumenes

    filas = []
    for m in METRICAS:
        va, vb = a["global"].get(m), b["global"].get(m)
        filas.append({"metrica": m, tag_a: va, tag_b: vb,
                      "variacion": round(vb - va, 4) if va is not None and vb is not None else None})
    filas.append({"metrica": "rechazo_correcto", tag_a: a["tasa_rechazo_correcto"],
                  tag_b: b["tasa_rechazo_correcto"],
                  "variacion": round(b["tasa_rechazo_correcto"] - a["tasa_rechazo_correcto"], 4)})
    tabla = pd.DataFrame(filas)

    print("\nParámetros:")
    for r in (a, b):
        print(f"  {r['tag']:<12} chunk={r['chunk_size']} overlap={r['chunk_overlap']} "
              f"top_k={r['top_k']}")
    print("\n" + tabla.to_string(index=False))

    tabla.to_csv(os.path.join(RESULTS_DIR, f"comparacion_{tag_a}_vs_{tag_b}.csv"), index=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    graf = tabla.dropna(subset=[tag_a, tag_b])
    x = range(len(graf))
    ancho = 0.38
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar([i - ancho / 2 for i in x], graf[tag_a], ancho, label=tag_a)
    ax.bar([i + ancho / 2 for i in x], graf[tag_b], ancho, label=tag_b)
    ax.set_xticks(list(x))
    ax.set_xticklabels(graf["metrica"], rotation=15)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Puntaje (0-1)")
    ax.set_title("Evaluación Ragas: antes vs. después de la mejora")
    ax.legend()
    plt.tight_layout()
    ruta_png = os.path.join(RESULTS_DIR, f"comparacion_{tag_a}_vs_{tag_b}.png")
    plt.savefig(ruta_png, dpi=150)
    print(f"\nGráfico guardado en {ruta_png}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Evaluación RAG con Ragas")
    p.add_argument("--tag", help="Nombre de la corrida (ej. baseline, mejora)")
    p.add_argument("--chunk-size", type=int, default=config.CHUNK_SIZE)
    p.add_argument("--overlap", type=int, default=config.CHUNK_OVERLAP)
    p.add_argument("--top-k", type=int, default=config.TOP_K)
    p.add_argument("--dataset", default="eval_dataset.json")
    p.add_argument("--pausa", type=float, default=4.0, help="segundos entre preguntas (sube el valor si ves errores 429)")
    p.add_argument("--reconstruir", action="store_true", help="reconstruye el índice")
    p.add_argument("--comparar", nargs=2, metavar=("ANTES", "DESPUES"))
    args = p.parse_args()

    if args.comparar:
        comparar(*args.comparar)
    elif args.tag:
        evaluar(args)
    else:
        p.error("Indica --tag NOMBRE para evaluar o --comparar ANTES DESPUES.")