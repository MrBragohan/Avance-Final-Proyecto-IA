"""Prueba de las tools SIN el LLM (requisito 1.4 del Avance 3).

Llama a cada herramienta con `.invoke({...})`, igual que lo haría el agente, y muestra lo que
devuelve. No necesita GROQ_API_KEY (los embeddings son locales). La búsqueda web solo se ejecuta
si hay TAVILY_API_KEY.

    python probar_tools.py                 # imprime y guarda results/prueba_tools.txt
"""
import io
import os
import sys
from contextlib import redirect_stdout

import config
from herramientas import EstadoTurno, Recursos, construir_herramientas, crear_buscador_web


def mostrar(titulo, herramienta, argumentos):
    print("\n" + "=" * 78)
    print(f"{titulo}\n>>> {herramienta.nombre}.invoke({argumentos})")
    print("-" * 78)
    print(herramienta.invoke(argumentos))


def main():
    from ingest import crear_embeddings, obtener_indice

    indice = obtener_indice(crear_embeddings(), config.CHUNK_SIZE, config.CHUNK_OVERLAP)
    buscador = crear_buscador_web()
    estado = EstadoTurno()
    h = construir_herramientas(Recursos(indice, buscador), estado)

    print("LÍMITES ESCRITOS EN EL CÓDIGO (config.py):")
    print(f"  máx. fragmentos por llamada = {config.MAX_FRAGMENTOS_TOOL}, "
          f"máx. caracteres por fragmento = {config.MAX_CHARS_FRAGMENTO}")
    print(f"  máx. resultados web = {config.MAX_RESULTADOS_WEB}, "
          f"extracto web = {config.MAX_CHARS_EXTRACTO_WEB} caracteres")
    print(f"  sitios permitidos = {config.SITIOS_WEB_PERMITIDOS}")

    mostrar("1. Búsqueda en la base (tema cubierto)", h["buscar_en_base"],
            {"consulta": "qué es un embedding"})
    mostrar("2. Búsqueda en la base (tema NO cubierto: devuelve lo más cercano; el LLM decide)",
            h["buscar_en_base"], {"consulta": "capital de Francia"})
    mostrar("3. Intento de pedir más fragmentos (parámetro que no existe)", h["buscar_en_base"],
            {"consulta": "RAG", "k": 100})
    mostrar("4. Tarea del dominio: número de fragmentos", h["estimar_numero_fragmentos"],
            {"longitud_caracteres": 12000, "chunk_size": 500, "chunk_overlap": 50})
    mostrar("5. Tarea del dominio: costo de tokens", h["estimar_costo_llm"],
            {"tokens_entrada": 2_000_000, "tokens_salida": 500_000,
             "precio_entrada_por_millon": 0.15, "precio_salida_por_millon": 0.60})
    mostrar("6. Argumentos inválidos (se devuelve un ERROR legible, no una excepción)",
            h["estimar_numero_fragmentos"], {"longitud_caracteres": 100, "chunk_size": 50, "chunk_overlap": 80})

    # La política «primero la base» se comprueba con un estado limpio
    limpio = construir_herramientas(Recursos(indice, buscador), EstadoTurno())
    mostrar("7. Web SIN haber consultado antes la base → bloqueada por el código",
            limpio["buscar_en_web"], {"consulta": "precio API Gemini por millón de tokens"})

    if buscador is None:
        print("\n[!] Sin TAVILY_API_KEY: se omite la prueba real de búsqueda web.")
    else:
        mostrar("8. Web después de consultar la base (sitios permitidos, máx. 3 resultados)",
                h["buscar_en_web"], {"consulta": "Gemini API pricing per million tokens"})


if __name__ == "__main__":
    buffer = io.StringIO()

    class Tee:
        def write(self, s):
            sys.__stdout__.write(s)
            buffer.write(s)

        def flush(self):
            sys.__stdout__.flush()

    with redirect_stdout(Tee()):
        main()
    os.makedirs("results", exist_ok=True)
    with open(os.path.join("results", "prueba_tools.txt"), "w", encoding="utf-8") as f:
        f.write(buffer.getvalue())
    print("\nGuardado en results/prueba_tools.txt")
