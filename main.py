import os
from dotenv import load_dotenv
from google import genai
from google.genai import types
from retriever import cargar_base_conocimiento, recuperar_contexto


load_dotenv()  # Load environment variables from .env file

API_KEY = os.getenv("GENAI_API_KEY")

# Inicializar el cliente
client = genai.Client(api_key=API_KEY)

MODEL = "gemini-3-flash-preview"

# --------------------------------------------------------------------------
# SYSTEM PROMPT
# --------------------------------------------------------------------------
SYSTEM_INSTRUCTION = """Eres un Tutor Académico Personalizado especializado en Creación de
Aplicaciones con IA, dirigido a estudiantes de Ingeniería de Sistemas.

Tu tono es paciente, claro y motivador. Explicas los conceptos con ejemplos sencillos antes
de profundizar en la teoría.

Reglas de uso del contexto:
1. Recibirás fragmentos de material de estudio delimitados con etiquetas <contexto></contexto>.
2. Debes responder ÚNICAMENTE basándote en la información de ese contexto.
3. Si la pregunta del estudiante no puede responderse con el contexto proporcionado, responde:
   "No tengo información suficiente en el material de estudio para responder esto con precisión."
4. Nunca inventes información que no esté en el contexto.

Formato de salida obligatorio (Markdown):
## Explicación
(Explicación clara y breve del concepto)

## Ejemplo
(Un ejemplo práctico o analogía)

## Fuente
(Menciona de qué tema del material de estudio proviene la información)
"""

# --------------------------------------------------------------------------
# FEW-SHOT
# --------------------------------------------------------------------------
EJEMPLO_FEW_SHOT_USER = """<contexto>
[TEMA: Prompt Engineering]
El Prompt Engineering es la disciplina de diseñar instrucciones efectivas para obtener el
mejor resultado posible de un modelo de lenguaje...
</contexto>

<pregunta>
¿Qué es el Prompt Engineering?
</pregunta>"""

EJEMPLO_FEW_SHOT_MODEL = """## Explicación
El Prompt Engineering es la disciplina de diseñar instrucciones claras y efectivas para
guiar a un modelo de lenguaje hacia el resultado deseado, usando pilares como tarea,
contexto, persona, formato, tono y ejemplos.

## Ejemplo
En vez de escribir "escribe sobre perros", un buen prompt sería: "Eres un veterinario.
Escribe un párrafo breve, en tono educativo, sobre los cuidados básicos de un cachorro."

## Fuente
Tema: Prompt Engineering"""

configuration = types.GenerateContentConfig(
    max_output_tokens=2048,
    temperature=0.4,
    system_instruction=SYSTEM_INSTRUCTION
)

chat = client.chats.create(
    model=MODEL,
    config=configuration,
    history=[
        types.Content(role="user", parts=[types.Part(text=EJEMPLO_FEW_SHOT_USER)]),
        types.Content(role="model", parts=[types.Part(text=EJEMPLO_FEW_SHOT_MODEL)]),
    ]
)

# Cargar base de conocimiento
chunks = cargar_base_conocimiento("knowledge_base.txt")

print("--- Tutor Académico de Creación de Aplicaciones con IA (RAG) ---")
print("(Escribe 'salir' para terminar)\n")

while True:
    pregunta_usuario = input("Estudiante: ")

    if pregunta_usuario.lower() in ["salir", "exit", "quit"]:
        print("Tutor: ¡Hasta la próxima! Sigue estudiando. 📚")
        break

    try:
        contexto_relevante = recuperar_contexto(pregunta_usuario, chunks, top_k=2)
        contexto_texto = "\n\n".join(contexto_relevante)

        prompt = f"""<contexto>
{contexto_texto}
</contexto>

<pregunta>
{pregunta_usuario}
</pregunta>"""

        response = chat.send_message(prompt)

        print(f"\nTutor:\n{response.text}\n")

    except Exception as e:
        print(f"Error al procesar la solicitud: {e}")