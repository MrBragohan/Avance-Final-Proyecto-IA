"""Ingesta del corpus: extracción de texto, chunking, embeddings y base vectorial (Chroma).

Uso desde la terminal:
    python ingest.py                       # construye el índice con los parámetros de config.py
    python ingest.py --reconstruir         # borra y vuelve a crear el índice
    python ingest.py --chunk-size 1000 --overlap 150
"""
import argparse
import collections
import os
import re
import shutil

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config

EXTENSIONES = {".pdf", ".txt", ".docx", ".html", ".htm"}


# ---------------------------------------------------------------------------
# 1. INGESTA: extracción de texto (PDF, DOCX, TXT, HTML)
# ---------------------------------------------------------------------------
def cargar_archivo(ruta):
    """Extrae el texto de un archivo y devuelve Documents con metadatos mínimos y seguros."""
    ext = os.path.splitext(ruta)[1].lower()

    if ext == ".pdf":
        from langchain_community.document_loaders import PyPDFLoader
        crudos = PyPDFLoader(ruta).load()
    elif ext == ".txt":
        from langchain_community.document_loaders import TextLoader
        crudos = TextLoader(ruta, encoding="utf-8").load()
        crudos = _separar_por_tema(crudos)
    elif ext == ".docx":
        from langchain_community.document_loaders import Docx2txtLoader
        crudos = Docx2txtLoader(ruta).load()
    elif ext in (".html", ".htm"):
        from langchain_community.document_loaders import BSHTMLLoader
        crudos = BSHTMLLoader(ruta, open_encoding="utf-8").load()
    else:
        return []

    nombre = os.path.basename(ruta)
    documentos = []
    for d in crudos:
        # Solo metadatos simples (Chroma no acepta None ni estructuras anidadas).
        # 'source', 'page' y 'seccion' son los que permiten CITAR la fuente de cada respuesta.
        meta = {"source": nombre}
        if "page" in d.metadata:
            meta["page"] = int(d.metadata["page"]) + 1  # PyPDF cuenta desde 0
        if d.metadata.get("seccion"):
            meta["seccion"] = d.metadata["seccion"]
        if d.page_content.strip():
            documentos.append(Document(page_content=d.page_content, metadata=meta))
    return documentos


_RE_TEMA = re.compile(r"(?m)^\[TEMA:\s*(.+)\]\s*$")


def _separar_por_tema(crudos):
    """AVANCE 3: los apuntes .txt no tienen páginas; se dividen por sus encabezados [TEMA: ...]
    y el nombre del tema queda como metadato `seccion`, para poder citar "documento, sección".
    Si el texto no tiene encabezados, se devuelve igual que antes."""
    salida = []
    for d in crudos:
        marcas = list(_RE_TEMA.finditer(d.page_content))
        if not marcas:
            salida.append(d)
            continue
        for i, m in enumerate(marcas):
            fin = marcas[i + 1].start() if i + 1 < len(marcas) else len(d.page_content)
            texto = d.page_content[m.start():fin].strip()
            salida.append(Document(page_content=texto, metadata={"seccion": m.group(1).strip()}))
    return salida


def quitar_encabezados_repetidos(paginas, min_paginas=5, min_ratio=0.8, min_largo=25):
    """Elimina encabezados/pies de página repetidos (dirección, NIT, teléfono, etc.).

    Una línea se considera "ruido de plantilla" si, ignorando dígitos y espacios, aparece en
    al menos `min_ratio` (80 %) de las páginas del documento y tiene `min_largo` caracteres o más.
    El largo mínimo protege líneas cortas legítimas que se repiten, como los títulos
    "Artículo 12:" de un reglamento. Sin esta limpieza, el texto de plantilla se cuela en
    todos los fragmentos y diluye la similitud en la recuperación.
    """
    if len(paginas) < min_paginas:
        return paginas

    def clave(linea):
        return re.sub(r"\d+", "#", re.sub(r"\s+", " ", linea.strip().lower()))

    conteo = collections.Counter()
    for d in paginas:
        conteo.update({clave(l) for l in d.page_content.splitlines() if len(l.strip()) >= min_largo})
    ruido = {k for k, c in conteo.items() if c >= min_ratio * len(paginas)}
    if not ruido:
        return paginas

    limpias = []
    for d in paginas:
        lineas = [l for l in d.page_content.splitlines() if clave(l) not in ruido]
        texto = "\n".join(lineas).strip()
        if texto:
            limpias.append(Document(page_content=texto, metadata=d.metadata))
    print(f"  [ingesta] limpieza: {len(ruido)} línea(s) de encabezado/pie repetidas eliminadas")
    return limpias


def cargar_documentos(docs_dir=None):
    """Carga todos los archivos soportados de la carpeta del corpus."""
    docs_dir = docs_dir or config.DOCS_DIR
    if not os.path.isdir(docs_dir):
        raise FileNotFoundError(f"No existe la carpeta del corpus: '{docs_dir}/'")

    documentos = []
    archivos = sorted(
        f for f in os.listdir(docs_dir) if os.path.splitext(f)[1].lower() in EXTENSIONES
    )
    for nombre in archivos:
        paginas = cargar_archivo(os.path.join(docs_dir, nombre))
        if config.LIMPIAR_ENCABEZADOS:
            paginas = quitar_encabezados_repetidos(paginas)
        documentos.extend(paginas)
        print(f"  [ingesta] {nombre}: {len(paginas)} página(s)/sección(es) cargadas")

    if not documentos:
        raise ValueError(
            f"La carpeta '{docs_dir}/' no contiene archivos válidos {sorted(EXTENSIONES)}."
        )
    return documentos


# ---------------------------------------------------------------------------
# 2. CHUNKING
# ---------------------------------------------------------------------------
def dividir_documentos(documentos, chunk_size, chunk_overlap):
    """Fragmenta con criterio jerárquico: párrafo > línea > oración > palabra.

    RecursiveCharacterTextSplitter intenta cortar primero en saltos de párrafo y solo
    si el fragmento sigue siendo demasiado grande baja al siguiente separador. Así se
    respetan al máximo los límites naturales del texto.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documentos)

    # chunk_id incremental por documento: permite citar "documento + fragmento #n"
    contadores = {}
    for c in chunks:
        fuente = c.metadata["source"]
        contadores[fuente] = contadores.get(fuente, 0) + 1
        c.metadata["chunk_id"] = contadores[fuente]
    return chunks


# ---------------------------------------------------------------------------
# 3. VECTORIZACIÓN
# ---------------------------------------------------------------------------
def crear_embeddings():
    """Embeddings locales (CPU). Ningún texto del corpus se envía a servicios externos."""
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


# ---------------------------------------------------------------------------
# 4. BASE VECTORIAL PERSISTENTE (Chroma, similitud coseno)
# ---------------------------------------------------------------------------
def ruta_indice(chunk_size, chunk_overlap):
    """Un directorio por configuración: permite comparar iteraciones sin pisar el índice."""
    # sufijo _v3: el índice del Avance 3 tiene el metadato extra `seccion`
    return os.path.join(config.CHROMA_BASE_DIR, f"cs{chunk_size}_ov{chunk_overlap}_v3")


def obtener_indice(embeddings, chunk_size, chunk_overlap, reconstruir=False):
    """Carga el índice persistido o lo construye si no existe."""
    from langchain_chroma import Chroma

    persist_dir = ruta_indice(chunk_size, chunk_overlap)
    kwargs = dict(
        persist_directory=persist_dir,
        collection_name=config.COLLECTION_NAME,
        collection_metadata={"hnsw:space": "cosine"},
    )

    if reconstruir and os.path.exists(persist_dir):
        shutil.rmtree(persist_dir)

    if os.path.exists(persist_dir):
        store = Chroma(embedding_function=embeddings, **kwargs)
        if store._collection.count() > 0:
            return store

    print(f"[ingesta] Construyendo índice en {persist_dir}/ ...")
    documentos = cargar_documentos()
    chunks = dividir_documentos(documentos, chunk_size, chunk_overlap)
    print(f"[ingesta] {len(documentos)} páginas/secciones -> {len(chunks)} fragmentos")

    return Chroma.from_documents(documents=chunks, embedding=embeddings, **kwargs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Construye el índice vectorial del corpus")
    parser.add_argument("--chunk-size", type=int, default=config.CHUNK_SIZE)
    parser.add_argument("--overlap", type=int, default=config.CHUNK_OVERLAP)
    parser.add_argument("--reconstruir", action="store_true")
    args = parser.parse_args()

    emb = crear_embeddings()
    store = obtener_indice(emb, args.chunk_size, args.overlap, args.reconstruir)
    print(f"[OK] Índice listo con {store._collection.count()} fragmentos "
          f"(chunk_size={args.chunk_size}, overlap={args.overlap}).")
