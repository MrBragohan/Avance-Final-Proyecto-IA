"""Herramientas (tools) del agente — Avance 3.

Principio de diseño: las REGLAS DURAS viven en el código, no en el prompt. El prompt le dice al
modelo cuándo conviene usar cada tool, pero lo que NO puede hacer (devolver más de N fragmentos,
buscar fuera de la lista de sitios, pegar la base en el buscador web) lo impide este archivo.

Herramientas:
  1. buscar_en_base            — recuperación del Avance 2 convertida en tool (fragmentos + cita).
  2. buscar_en_web             — búsqueda web controlada (Tavily), política "primero la base".
  3. estimar_numero_fragmentos — tarea del dominio: cuántos chunks genera un texto (cálculo exacto).
  4. estimar_costo_llm         — tarea del dominio: costo en USD según tokens y precios (cálculo exacto).

Cada herramienta es un objeto `Herramienta` con `.invoke(dict)` (igual que en LangChain), de modo
que se puede probar SIN el LLM (ver probar_tools.py).

Las tools se construyen POR PREGUNTA con `construir_herramientas(recursos, estado)`: `recursos`
(índice vectorial, buscador web) es compartido y de solo lectura; `estado` guarda lo ocurrido en
ESA pregunta (fuentes devueltas, si ya se consultó la base). Así dos usuarios simultáneos en la
app no se mezclan.
"""
import inspect
import math
import re
from dataclasses import dataclass, field
from typing import Callable
from urllib.parse import urlparse

import config


# ---------------------------------------------------------------------------
# Infraestructura
# ---------------------------------------------------------------------------
@dataclass
class Herramienta:
    """Una tool: nombre, descripción (su docstring), esquema JSON de parámetros y función."""

    nombre: str
    descripcion: str
    parametros: dict
    funcion: Callable[..., str]

    def invoke(self, argumentos=None) -> str:
        """Ejecuta la tool con un dict de argumentos. Los errores se devuelven como texto
        (así el LLM puede corregirse) en lugar de romper el bucle del agente."""
        argumentos = argumentos or {}
        if not isinstance(argumentos, dict):
            return "ERROR: los argumentos deben ser un objeto JSON."
        try:
            return str(self.funcion(**argumentos))
        except TypeError as e:
            return f"ERROR: argumentos inválidos para {self.nombre}: {e}"
        except Exception as e:  # noqa: BLE001 — la tool nunca debe tumbar al agente
            return f"ERROR: {self.nombre} falló: {type(e).__name__}: {e}"

    def esquema(self) -> dict:
        """Esquema en formato de function calling (OpenAI / Groq)."""
        return {
            "type": "function",
            "function": {
                "name": self.nombre,
                "description": self.descripcion,
                "parameters": self.parametros,
            },
        }


@dataclass
class EstadoTurno:
    """Lo que ocurrió durante UNA pregunta (lo usan las tools y lo lee el agente)."""

    fuentes_base: list = field(default_factory=list)   # [{fuente, pagina, seccion, chunk_id, score, texto}]
    fuentes_web: list = field(default_factory=list)    # [{titulo, url, extracto, origen}]
    contextos: list = field(default_factory=list)      # textos devueltos por las tools (para Ragas)
    busquedas_base: int = 0                            # nº de llamadas a buscar_en_base
    texto_base_devuelto: list = field(default_factory=list)  # para detectar fugas hacia la web


@dataclass
class Recursos:
    """Objetos pesados y compartidos (de solo lectura) que usan las tools."""

    vector_store: object            # Chroma (langchain) con similarity_search_with_relevance_scores
    buscador_web: object = None     # objeto con .buscar(consulta, max_resultados, dominios) o None


def _palabras(texto):
    return re.findall(r"\w+", (texto or "").lower())


def _limpiar_atributo(valor):
    """Evita que comillas o saltos de línea rompan las etiquetas XML del resultado."""
    return str(valor).replace('"', "'").replace("\n", " ").strip()


def etiqueta_fuente(meta):
    """Cómo se cita un fragmento: 'archivo.pdf, pág. 3' o 'archivo.txt, sección «Tema»'."""
    fuente = meta.get("source") or meta.get("fuente") or "desconocido"
    if meta.get("page") not in (None, "n/a") or meta.get("pagina") not in (None, "n/a"):
        pagina = meta.get("page", meta.get("pagina"))
        return f"{fuente}, pág. {pagina}"
    if meta.get("seccion"):
        return f"{fuente}, sección «{meta['seccion']}»"
    return fuente


def url_permitida(url, dominios=None):
    """True si el host de la URL es (o es subdominio de) alguno de los sitios permitidos."""
    dominios = dominios or config.SITIOS_WEB_PERMITIDOS
    host = (urlparse(url or "").hostname or "").lower()
    return any(host == d or host.endswith("." + d) for d in dominios)


def _copia_texto_de_la_base(consulta, textos_base, n=None):
    """True si la consulta contiene `n` palabras seguidas copiadas de fragmentos de la base."""
    n = n or config.N_PALABRAS_FUGA
    pal = _palabras(consulta)
    if len(pal) < n:
        return False
    ngramas = {" ".join(pal[i:i + n]) for i in range(len(pal) - n + 1)}
    for texto in textos_base:
        pb = _palabras(texto)
        for i in range(len(pb) - n + 1):
            if " ".join(pb[i:i + n]) in ngramas:
                return True
    return False


class BuscadorTavily:
    """Adaptador mínimo de Tavily. Devuelve [{titulo, url, extracto}]."""

    def __init__(self, api_key):
        from tavily import TavilyClient

        self._cliente = TavilyClient(api_key=api_key)

    def buscar(self, consulta, max_resultados, dominios):
        resp = self._cliente.search(
            query=consulta,
            max_results=max_resultados,
            include_domains=list(dominios),   # lista fija: la define config.py, no el LLM
            search_depth="basic",
        )
        return [
            {"titulo": r.get("title", ""), "url": r.get("url", ""), "extracto": r.get("content", "")}
            for r in resp.get("results", [])
        ]


def crear_buscador_web():
    """Devuelve el buscador web si hay TAVILY_API_KEY; si no, None (la tool avisa y sigue)."""
    if not config.TAVILY_API_KEY:
        return None
    try:
        return BuscadorTavily(config.TAVILY_API_KEY)
    except Exception:  # paquete no instalado o clave inválida: la app sigue sin web
        return None


# ---------------------------------------------------------------------------
# Construcción de las 4 tools
# ---------------------------------------------------------------------------
def construir_herramientas(recursos: Recursos, estado: EstadoTurno) -> dict:
    """Crea las tools ligadas a los recursos compartidos y al estado de la pregunta actual."""

    # ---------------- 1. buscar_en_base ----------------
    def buscar_en_base(consulta: str) -> str:
        """Busca en la BASE DE CONOCIMIENTOS (apuntes y guías de clase de la asignatura
        «Creación de Aplicaciones con IA») los fragmentos más relevantes para una consulta.

        Úsala SIEMPRE que la pregunta trate de la materia: LLMs, prompt engineering, Zero/One/
        Few-Shot, delimitadores, RAG, embeddings y bases vectoriales, agentes, function calling,
        fine-tuning, alucinaciones, temperatura, evaluación, multimodalidad, despliegue, ética
        y sesgos, configuración del sistema y roles de chat.
        NO la uses para saludos, agradecimientos ni conversación casual, ni para hacer cálculos.
        Si la pregunta tiene varios temas, llámala una vez por tema con una consulta corta y
        autónoma (sin pronombres como «eso»).
        Devuelve pocos fragmentos; cada uno trae documento y página o sección para poder citarlo.
        """
        consulta = (consulta or "").strip()
        if not consulta:
            return "ERROR: la consulta está vacía."
        if len(consulta) > config.MAX_CHARS_CONSULTA:
            return (f"ERROR: consulta demasiado larga (máx. {config.MAX_CHARS_CONSULTA} caracteres). "
                    "Reformúlala de forma corta.")

        estado.busquedas_base += 1
        resultados = recursos.vector_store.similarity_search_with_relevance_scores(
            consulta, k=config.MAX_FRAGMENTOS_TOOL          # límite escrito en código
        )
        if config.SCORE_MINIMO > 0:                          # 0 = filtro desactivado
            resultados = [(d, s) for d, s in resultados if s >= config.SCORE_MINIMO]
        resultados = resultados[: config.MAX_FRAGMENTOS_TOOL]
        if not resultados:
            return "SIN_RESULTADOS: la base de conocimientos no devolvió fragmentos para esta consulta."

        partes = []
        for i, (d, score) in enumerate(resultados, 1):
            texto = d.page_content.strip().replace("</fragmento>", "")
            if len(texto) > config.MAX_CHARS_FRAGMENTO:      # límite de longitud escrito en código
                texto = texto[: config.MAX_CHARS_FRAGMENTO].rstrip() + "…"
            meta = d.metadata
            cita = etiqueta_fuente(meta)
            partes.append(
                f'<fragmento id="{i}" origen="base" cita="{_limpiar_atributo(cita)}">\n{texto}\n</fragmento>'
            )
            estado.contextos.append(texto)
            estado.texto_base_devuelto.append(texto)
            registro = {
                "fuente": meta.get("source", "desconocido"),
                "pagina": meta.get("page", "n/a"),
                "seccion": meta.get("seccion", ""),
                "chunk_id": meta.get("chunk_id", "?"),
                "score": float(score),
                "texto": texto,
            }
            if not any(
                f["fuente"] == registro["fuente"] and f["chunk_id"] == registro["chunk_id"]
                and f["pagina"] == registro["pagina"] and f["seccion"] == registro["seccion"]
                for f in estado.fuentes_base
            ):
                estado.fuentes_base.append(registro)

        return (
            f'<resultado_base consulta="{_limpiar_atributo(consulta)}" fragmentos="{len(partes)}">\n'
            + "\n".join(partes)
            + "\n</resultado_base>"
        )

    # ---------------- 2. buscar_en_web ----------------
    def buscar_en_web(consulta: str) -> str:
        """Busca información PÚBLICA o ACTUAL en la web, restringida a una lista cerrada de
        sitios oficiales (documentación de proveedores de IA, frameworks y artículos académicos).

        Úsala SOLO si se cumplen las DOS condiciones: (1) ya llamaste a buscar_en_base en esta
        pregunta y la base NO contiene la respuesta; (2) la pregunta trata de información pública
        o que cambia con el tiempo sobre IA o LLMs (precios, versiones, límites, documentación
        oficial). NO la uses si la base ya responde, para saludos, para temas ajenos a la IA ni
        para cálculos. Escribe una consulta corta con tus propias palabras: nunca copies texto
        de la base de conocimientos ni datos personales del usuario.
        Devuelve como máximo 3 resultados (título, URL y extracto corto), marcados con origen web.
        """
        consulta = (consulta or "").strip()
        if not consulta:
            return "ERROR: la consulta está vacía."
        if recursos.buscador_web is None:
            return ("NO_DISPONIBLE: la búsqueda web no está configurada (falta TAVILY_API_KEY). "
                    "Responde solo con lo que haya en la base.")
        if estado.busquedas_base == 0:
            return ("BLOQUEADO: política «primero la base». Llama antes a buscar_en_base y usa la "
                    "web solo si la base no tiene la respuesta.")
        if len(consulta) > config.MAX_CHARS_CONSULTA_WEB:
            return (f"BLOQUEADO: consulta demasiado larga (máx. {config.MAX_CHARS_CONSULTA_WEB} "
                    "caracteres). Usa una consulta corta con tus propias palabras.")
        if _copia_texto_de_la_base(consulta, estado.texto_base_devuelto):
            return ("BLOQUEADO: la consulta contiene texto copiado de la base de conocimientos. "
                    "Reescríbela con tus propias palabras, sin pegar contenido de la base.")

        crudos = recursos.buscador_web.buscar(
            consulta, config.MAX_RESULTADOS_WEB, config.SITIOS_WEB_PERMITIDOS
        )
        # Segunda barrera: aunque el buscador devolviera otro dominio, aquí se descarta.
        crudos = [r for r in crudos if url_permitida(r.get("url"))][: config.MAX_RESULTADOS_WEB]
        if not crudos:
            return "SIN_RESULTADOS: no hay resultados en los sitios permitidos para esta consulta."

        partes = []
        for i, r in enumerate(crudos, 1):
            extracto = " ".join((r.get("extracto") or "").split())
            if len(extracto) > config.MAX_CHARS_EXTRACTO_WEB:
                extracto = extracto[: config.MAX_CHARS_EXTRACTO_WEB].rstrip() + "…"
            titulo = (r.get("titulo") or r["url"]).strip()
            partes.append(
                f'<resultado id="{i}" origen="web" titulo="{_limpiar_atributo(titulo)}" '
                f'url="{_limpiar_atributo(r["url"])}">\n{extracto}\n</resultado>'
            )
            estado.contextos.append(extracto)
            if not any(f["url"] == r["url"] for f in estado.fuentes_web):
                estado.fuentes_web.append(
                    {"titulo": titulo, "url": r["url"], "extracto": extracto, "origen": "web"}
                )
        return (
            f'<resultado_web consulta="{_limpiar_atributo(consulta)}" origen="web" '
            f'resultados="{len(partes)}">\n' + "\n".join(partes) + "\n</resultado_web>"
        )

    # ---------------- 3. estimar_numero_fragmentos ----------------
    def estimar_numero_fragmentos(longitud_caracteres: int, chunk_size: int = None,
                                  chunk_overlap: int = None) -> str:
        """Calcula cuántos fragmentos (chunks) genera un texto de N caracteres al dividirlo con un
        tamaño de fragmento y un solape dados (ventana deslizante). Es un cálculo exacto.

        Úsala cuando el usuario pregunte cuántos chunks/fragmentos resultan de un documento o texto
        de cierta longitud. Extrae de la pregunta: longitud en caracteres, chunk_size y
        chunk_overlap. Si el usuario no da chunk_size o chunk_overlap, omítelos (se usan los del
        proyecto: 500 y 50). No hagas esta cuenta de memoria.
        El resultado es una estimación por ventana fija; un splitter real que corta por párrafos
        puede producir algunos fragmentos más.
        """
        longitud = int(float(longitud_caracteres))
        tam = int(float(chunk_size)) if chunk_size is not None else config.CHUNK_SIZE
        solape = int(float(chunk_overlap)) if chunk_overlap is not None else config.CHUNK_OVERLAP
        if longitud < 0 or tam <= 0 or solape < 0:
            return "ERROR: longitud, chunk_size y chunk_overlap deben ser positivos."
        if solape >= tam:
            return "ERROR: chunk_overlap debe ser menor que chunk_size."
        paso = tam - solape
        n = 0 if longitud == 0 else (1 if longitud <= tam else 1 + math.ceil((longitud - tam) / paso))
        return (
            f"Texto de {longitud} caracteres, chunk_size={tam}, chunk_overlap={solape} "
            f"(avance por fragmento = {paso}).\n"
            f"Fórmula: 1 + ceil(({longitud} - {tam}) / {paso}) cuando la longitud supera chunk_size.\n"
            f"Resultado: {n} fragmentos (estimación por ventana fija)."
        )

    # ---------------- 4. estimar_costo_llm ----------------
    def estimar_costo_llm(tokens_entrada: int, tokens_salida: int,
                          precio_entrada_por_millon: float, precio_salida_por_millon: float,
                          numero_solicitudes: int = 1) -> str:
        """Calcula el costo en USD de usar un LLM por API a partir de los tokens y los precios por
        millón de tokens. Es un cálculo exacto.

        Úsala cuando el usuario pida estimar o calcular costos de tokens/llamadas a un modelo.
        Extrae de la pregunta: tokens_entrada y tokens_salida POR SOLICITUD, los dos precios por
        millón de tokens (entrada y salida) y numero_solicitudes (1 si los tokens ya son el total).
        NO inventes precios: si el usuario no los da, pídeselos o, si son públicos y actuales,
        sigue la política de la web (primero la base). No hagas esta cuenta de memoria.
        """
        t_in, t_out = float(tokens_entrada), float(tokens_salida)
        p_in, p_out = float(precio_entrada_por_millon), float(precio_salida_por_millon)
        n = int(float(numero_solicitudes))
        if min(t_in, t_out, p_in, p_out) < 0 or n < 1:
            return "ERROR: tokens y precios no pueden ser negativos y numero_solicitudes debe ser ≥ 1."
        costo_in = t_in / 1_000_000 * p_in * n
        costo_out = t_out / 1_000_000 * p_out * n
        total = costo_in + costo_out
        return (
            f"Solicitudes: {n}. Tokens por solicitud: entrada={t_in:,.0f}, salida={t_out:,.0f}.\n"
            f"Costo de entrada = {t_in * n:,.0f} / 1,000,000 × {p_in} = {costo_in:.4f} USD\n"
            f"Costo de salida  = {t_out * n:,.0f} / 1,000,000 × {p_out} = {costo_out:.4f} USD\n"
            f"Costo total = {total:.4f} USD"
        )

    # ---------------- Registro de esquemas ----------------
    cadena = {"type": "string"}
    entero = {"type": "integer"}
    numero = {"type": "number"}
    definiciones = [
        (buscar_en_base, {
            "type": "object",
            "properties": {"consulta": {**cadena, "description":
                "Consulta corta y autónoma con los términos clave del tema a buscar."}},
            "required": ["consulta"],
        }),
        (buscar_en_web, {
            "type": "object",
            "properties": {"consulta": {**cadena, "description":
                "Consulta web corta, en tus propias palabras (sin texto copiado de la base)."}},
            "required": ["consulta"],
        }),
        (estimar_numero_fragmentos, {
            "type": "object",
            "properties": {
                "longitud_caracteres": {**entero, "description": "Longitud del texto en caracteres."},
                "chunk_size": {**entero, "description": "Tamaño máximo de cada fragmento (opcional)."},
                "chunk_overlap": {**entero, "description": "Solape entre fragmentos (opcional)."},
            },
            "required": ["longitud_caracteres"],
        }),
        (estimar_costo_llm, {
            "type": "object",
            "properties": {
                "tokens_entrada": {**entero, "description": "Tokens de entrada por solicitud."},
                "tokens_salida": {**entero, "description": "Tokens de salida por solicitud."},
                "precio_entrada_por_millon": {**numero, "description": "USD por millón de tokens de entrada."},
                "precio_salida_por_millon": {**numero, "description": "USD por millón de tokens de salida."},
                "numero_solicitudes": {**entero, "description": "Nº de solicitudes (1 si los tokens ya son el total)."},
            },
            "required": ["tokens_entrada", "tokens_salida",
                         "precio_entrada_por_millon", "precio_salida_por_millon"],
        }),
    ]
    return {
        f.__name__: Herramienta(f.__name__, inspect.getdoc(f), esquema, f)
        for f, esquema in definiciones
    }
