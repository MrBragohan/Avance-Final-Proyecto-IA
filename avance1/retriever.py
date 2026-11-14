import re


def cargar_base_conocimiento(ruta_archivo):
    """
    Carga la base de conocimiento y la divide en fragmentos (chunks)
    usando el delimitador [TEMA: ...] como separador.
    """
    with open(ruta_archivo, "r", encoding="utf-8") as f:
        contenido = f.read()

    bloques = contenido.split("[TEMA:")
    chunks = []

    for bloque in bloques:
        bloque = bloque.strip()
        if bloque:
            chunks.append("[TEMA:" + bloque)

    return chunks


def extraer_palabras(texto):
    """
    Extrae solo palabras (sin signos de puntuación) en minúsculas,
    usando una expresión regular, para que la comparación de palabras
    clave no falle por signos como '?', '¿', '.', ','.
    """
    return set(re.findall(r"\w+", texto.lower()))


def recuperar_contexto(pregunta, chunks, top_k=2):
    """
    Recuperador simple basado en coincidencia de palabras clave.
    Calcula cuántas palabras de la pregunta aparecen en cada chunk
    y devuelve los 'top_k' fragmentos más relevantes.

    Nota: en una versión avanzada, esto se reemplazaría por embeddings
    y similitud coseno (búsqueda semántica real).
    """
    palabras_pregunta = extraer_palabras(pregunta)

    puntuaciones = []
    for chunk in chunks:
        palabras_chunk = extraer_palabras(chunk)
        interseccion = palabras_pregunta.intersection(palabras_chunk)
        puntuaciones.append((len(interseccion), chunk))

    # Ordenar por relevancia (mayor coincidencia primero)
    puntuaciones.sort(key=lambda x: x[0], reverse=True)

    mejores_chunks = [chunk for score, chunk in puntuaciones[:top_k] if score > 0]

    # Si no hay coincidencias, devolvemos los primeros chunks como respaldo
    if not mejores_chunks:
        mejores_chunks = chunks[:top_k]

    return mejores_chunks
