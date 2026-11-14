"""Configuración central: parámetros del pipeline RAG, modelos y secretos.

Las claves NUNCA se escriben en el código ni en el repositorio:
- En local se leen del archivo .env (ignorado por git).
- En la nube (Streamlit Community Cloud, etc.) se leen de los Secrets de la plataforma.
"""
import os
from dotenv import load_dotenv

load_dotenv()  # Carga variables desde .env (solo en local)


def get_secret(nombre, por_defecto=None):
    """Busca primero en variables de entorno y luego en st.secrets (Streamlit Cloud)."""
    valor = os.getenv(nombre)
    if valor:
        return valor
    try:
        import streamlit as st
        return st.secrets.get(nombre, por_defecto)
    except Exception:
        return por_defecto


# --- Credenciales y modelo generativo (proveedor intercambiable: Groq o Gemini) ---
GROQ_API_KEY = get_secret("GROQ_API_KEY")
GENAI_API_KEY = get_secret("GENAI_API_KEY")

# Si no se indica LLM_PROVIDER, se usa Groq cuando hay GROQ_API_KEY y Gemini en caso contrario
LLM_PROVIDER = (get_secret("LLM_PROVIDER") or ("groq" if GROQ_API_KEY else "gemini")).lower()
GROQ_MODEL = get_secret("GROQ_MODEL", "openai/gpt-oss-120b")
GEMINI_MODEL = get_secret("GEMINI_MODEL", "gemini-3-flash-preview")

LLM_MODEL = GROQ_MODEL if LLM_PROVIDER == "groq" else GEMINI_MODEL
API_KEY = GROQ_API_KEY if LLM_PROVIDER == "groq" else GENAI_API_KEY
NOMBRE_VARIABLE_CLAVE = "GROQ_API_KEY" if LLM_PROVIDER == "groq" else "GENAI_API_KEY"

# --- Búsqueda web (Avance 3): la clave va SIEMPRE en variables de entorno / Secrets ---
TAVILY_API_KEY = get_secret("TAVILY_API_KEY")

# --- Embeddings (se ejecutan LOCALMENTE en CPU: el corpus no sale de tu equipo/servidor) ---
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

# --- Corpus y base vectorial ---
DOCS_DIR = "docs"
CHROMA_BASE_DIR = "chroma_db"
COLLECTION_NAME = "tutor_ia"

# --- Parámetros del pipeline (línea base; se modifican en la iteración de mejora) ---
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K = 4
LIMPIAR_ENCABEZADOS = True  # quita encabezados/pies de página repetidos en cada página

# --- Conversación ---
MAX_HISTORIAL_TURNOS = 6  # nº de turnos (pregunta+respuesta) que se reenvían al modelo

# Frase fija que el asistente debe devolver cuando la respuesta no está en el corpus
FRASE_SIN_INFO = (
    "No tengo información suficiente en el material de estudio "
    "para responder esto con precisión."
)


# ===========================================================================
# AVANCE 3 — Agente con herramientas
# Estos límites se aplican EN EL CÓDIGO de las tools (herramientas.py): el LLM no puede
# cambiarlos ni saltárselos, aunque el usuario se lo pida en la pregunta.
# ===========================================================================

# --- Bucle del agente ---
MAX_ITERACIONES = 5          # máx. llamadas al LLM por pregunta (la última se hace SIN tools)
AGENTE_TEMPERATURA = 0.2     # igual que el pipeline fijo del Avance 2, para comparar en igualdad
LOG_TOOLS_PATH = "logs/uso_tools.jsonl"   # un registro JSON por pregunta

# --- Tool buscar_en_base ---
MAX_FRAGMENTOS_TOOL = 4      # máx. fragmentos devueltos por llamada
MAX_CHARS_FRAGMENTO = 600    # longitud máx. de cada fragmento (se recorta si la supera)
MAX_CHARS_CONSULTA = 300     # longitud máx. de la consulta que escribe el LLM
SCORE_MINIMO = 0.0           # similitud mínima [0-1]; 0 = desactivado (calíbralo con Ragas si quieres)

# --- Tool buscar_en_web (política "primero la base") ---
MAX_RESULTADOS_WEB = 3       # máx. resultados por búsqueda
MAX_CHARS_EXTRACTO_WEB = 300 # longitud máx. del extracto de cada resultado
MAX_CHARS_CONSULTA_WEB = 200 # consulta web corta: evita "colar" texto de la base en el buscador
N_PALABRAS_FUGA = 6          # una consulta web que copie 6+ palabras seguidas de la base se bloquea

# Lista cerrada de sitios (el LLM NO puede cambiarla). Sitios oficiales o de referencia del
# dominio "Creación de Aplicaciones con IA". Se pasa a Tavily (include_domains) y además se
# vuelve a filtrar en el código por si el buscador devolviera otro dominio.
SITIOS_WEB_PERMITIDOS = [
    "ai.google.dev",        # documentación y precios oficiales de Gemini
    "openai.com",           # documentación oficial de OpenAI
    "anthropic.com",        # documentación oficial de Anthropic
    "claude.com",           # documentación oficial de Claude
    "groq.com",             # documentación y límites oficiales de Groq (proveedor del LLM)
    "huggingface.co",       # modelos y documentación (embeddings, Sentence Transformers)
    "langchain.com",        # documentación oficial de LangChain
    "trychroma.com",        # documentación oficial de Chroma (base vectorial usada)
    "docs.ragas.io",        # documentación oficial de Ragas (evaluación)
    "arxiv.org",            # artículos académicos
    "deeplearning.ai",      # material educativo de referencia
    "promptingguide.ai",    # guía de referencia de prompt engineering
]
