"""Interfaz de chat del Tutor Académico — AGENTE con herramientas (Avance 3).

Ejecutar en local:   streamlit run app.py

Novedades frente al Avance 2:
  - Usa el agente (decide cuándo buscar en la base, calcular o buscar en la web).
  - Cada respuesta muestra qué tools usó el agente, con qué argumentos y cuántas llamadas al LLM.
  - Las fuentes se muestran separadas por origen: base (documento + página/sección) y web (título + URL).
"""
import json

import streamlit as st

import config
from agente import Agente

st.set_page_config(page_title="Tutor IA (Agente RAG)", page_icon="🤖", layout="centered")


@st.cache_resource(show_spinner="Cargando modelo de embeddings, índice vectorial y agente...")
def cargar_agente():
    # Se crea una sola vez por servidor: embeddings locales + índice Chroma + cliente LLM + buscador web
    return Agente.desde_config()


def mostrar_herramientas(m):
    """Sección desplegable: tools usadas, argumentos y costo de la pregunta."""
    usadas = m.get("herramientas") or []
    titulo = f"🛠️ Herramientas usadas ({len(usadas)})" if usadas else "🛠️ Sin herramientas (respuesta directa)"
    with st.expander(titulo):
        for i, t in enumerate(usadas, 1):
            st.markdown(f"**{i}. `{t['nombre']}`**")
            st.code(json.dumps(t["argumentos"], ensure_ascii=False, indent=2), language="json")
            st.caption(f"Resultado: {t['resultado'][:160].replace(chr(10), ' ')}…  ·  {t['duracion_s']} s")
        met = m.get("metricas", {})
        st.caption(
            f"Llamadas al LLM: **{met.get('llamadas_llm', '?')}** · tokens: "
            f"{met.get('tokens_entrada', '?')} entrada / {met.get('tokens_salida', '?')} salida · "
            f"tiempo: {met.get('tiempo_s', '?')} s"
            + (" · ⚠️ se alcanzó el límite de iteraciones" if met.get("limite_alcanzado") else "")
        )


def mostrar_fuentes(m):
    """Fuentes separadas por origen: base de conocimientos vs. web."""
    base, web = m.get("fuentes_base") or [], m.get("fuentes_web") or []
    if base:
        with st.expander(f"📚 Fuentes de la base de conocimientos ({len(base)})"):
            for i, f in enumerate(base, 1):
                ubicacion = f"pág. {f['pagina']}" if f["pagina"] != "n/a" else f"sección «{f['seccion']}»"
                st.markdown(f"**{i}. {f['fuente']}** · {ubicacion} · similitud {f['score']:.2f}")
                st.caption(f["texto"])
    if web:
        with st.expander(f"🌐 Fuentes web ({len(web)})"):
            for i, w in enumerate(web, 1):
                st.markdown(f"**{i}. [{w['titulo']}]({w['url']})** · origen: web")
                st.caption(w["extracto"])
    if not base and not web:
        if m.get("sin_info"):
            st.info("La información no está ni en la base de conocimientos ni en los sitios web permitidos.")
        else:
            st.caption("Esta respuesta no usó fuentes (conversación o cálculo).")


def mostrar_detalle(m):
    mostrar_herramientas(m)
    mostrar_fuentes(m)


st.title("🤖 Tutor Académico — Agente RAG")
st.caption(
    "Agente con herramientas: decide cuándo consultar el material de estudio, cuándo hacer un "
    "cálculo y, solo si la base no tiene la respuesta, cuándo buscar en sitios web oficiales."
)

if not config.API_KEY:
    st.error(
        f"No se encontró {config.NOMBRE_VARIABLE_CLAVE}. Configúrala en el archivo .env (local) "
        "o en los Secrets de la plataforma de despliegue."
    )
    st.stop()
if config.LLM_PROVIDER != "groq":
    st.error("El agente usa el tool calling de Groq: configura LLM_PROVIDER = \"groq\" y GROQ_API_KEY.")
    st.stop()

try:
    agente = cargar_agente()
except Exception as e:
    st.error(f"No se pudo inicializar el sistema: {e}")
    st.stop()

if "mensajes" not in st.session_state:
    st.session_state.mensajes = []

# ------------------------------------------------------------------ barra lateral
with st.sidebar:
    st.header("⚙️ Sistema")
    info = agente.info()
    st.markdown(
        f"- **LLM:** `{info['modelo_llm']}` ({info['proveedor_llm']})\n"
        f"- **Embeddings (local):** `{info['modelo_embeddings']}`\n"
        f"- **Chunk:** {info['chunk_size']} car. / solape {info['chunk_overlap']}\n"
        f"- **Máx. fragmentos por búsqueda:** {info['max_fragmentos_tool']}\n"
        f"- **Máx. iteraciones por pregunta:** {info['max_iteraciones']}\n"
        f"- **Búsqueda web:** {'✅ activa' if info['busqueda_web'] else '⚠️ desactivada (falta TAVILY_API_KEY)'}"
    )
    with st.expander("Sitios web permitidos"):
        st.markdown("\n".join(f"- {s}" for s in info["sitios_permitidos"]))
    st.divider()
    if st.button("🗑️ Nueva conversación", use_container_width=True):
        st.session_state.mensajes = []
        st.rerun()
    st.caption(
        "Privacidad: el corpus y el índice vectorial permanecen en este servidor. Al modelo solo se "
        "le envían los fragmentos que devuelve cada búsqueda. Si se usa la web, solo sale la "
        "consulta corta del agente (nunca texto de la base) y únicamente hacia los sitios permitidos."
    )

# ------------------------------------------------------------------ historial en pantalla
for m in st.session_state.mensajes:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m["role"] == "assistant":
            mostrar_detalle(m)

# ------------------------------------------------------------------ nuevo turno
pregunta = st.chat_input("Escribe tu pregunta sobre el material de estudio...")

if pregunta:
    # El historial se toma ANTES de agregar el turno actual (solo texto, sin trazas de tools)
    historial = [{"role": m["role"], "content": m["content"]} for m in st.session_state.mensajes]

    with st.chat_message("user"):
        st.markdown(pregunta)

    with st.chat_message("assistant"):
        with st.spinner("Pensando y decidiendo qué herramientas usar..."):
            try:
                res = agente.responder(pregunta, historial)
            except Exception as e:
                st.error(f"Error al procesar la solicitud: {e}")
                st.stop()

        st.markdown(res["respuesta"])
        registro = {
            "role": "assistant",
            "content": res["respuesta"],
            "herramientas": res["herramientas_usadas"],
            # si el agente rechazó la pregunta, no se muestran fuentes (como en el Avance 2)
            "fuentes_base": [] if res["sin_informacion"] else res["fuentes_base"],
            "fuentes_web": [] if res["sin_informacion"] else res["fuentes_web"],
            "sin_info": res["sin_informacion"],
            "metricas": {k: res[k] for k in
                         ("llamadas_llm", "tokens_entrada", "tokens_salida", "tiempo_s", "limite_alcanzado")},
        }
        mostrar_detalle(registro)

    st.session_state.mensajes.append({"role": "user", "content": pregunta})
    st.session_state.mensajes.append(registro)
