"""Capa mínima de acceso al LLM: Groq o Gemini con la misma interfaz.

El resto del proyecto solo conoce `LLMClient`; así se puede cambiar de proveedor con la
variable de entorno LLM_PROVIDER ("groq" o "gemini") sin tocar el pipeline.

  - generar(...)    : texto simple (pipeline fijo del Avance 2). Groq o Gemini.
  - chat_tools(...) : conversación con llamadas a herramientas (agente del Avance 3).
                      Usa el tool calling nativo de Groq (API compatible con OpenAI).

Los mensajes se pasan como lista de dicts: {"role": "user" | "assistant" | "tool", ...}.
`self.uso` acumula llamadas y tokens (se usa en la comparación de costos pipeline vs agente).
"""
import time

import config

# Esperas (segundos) entre reintentos: cubre errores 503 (alta demanda) y 429 (límite de tasa)
_ESPERAS = [2, 5, 10]


class LLMClient:
    def __init__(self, proveedor=None):
        self.proveedor = (proveedor or config.LLM_PROVIDER).lower()
        self.modelo = config.LLM_MODEL
        self.uso = {"llamadas": 0, "tokens_entrada": 0, "tokens_salida": 0}

        if self.proveedor == "groq":
            from groq import Groq
            self._cliente = Groq(api_key=config.GROQ_API_KEY)
        elif self.proveedor == "gemini":
            from google import genai
            self._cliente = genai.Client(api_key=config.GENAI_API_KEY)
        else:
            raise ValueError(
                f"LLM_PROVIDER='{self.proveedor}' no es válido. Usa 'groq' o 'gemini'."
            )

    # ------------------------------------------------------------------ utilidades
    def _reintentar(self, fn):
        """Ejecuta fn() con reintentos ante fallos transitorios (503, 429, tool_use_failed...)."""
        for intento in range(len(_ESPERAS) + 1):
            try:
                return fn()
            except Exception:
                if intento == len(_ESPERAS):
                    raise
                time.sleep(_ESPERAS[intento])

    def _registrar_uso(self, tokens_entrada, tokens_salida):
        self.uso["llamadas"] += 1
        self.uso["tokens_entrada"] += int(tokens_entrada or 0)
        self.uso["tokens_salida"] += int(tokens_salida or 0)

    # ------------------------------------------------------------------ texto simple
    def generar(self, mensajes, sistema=None, temperatura=0.2, max_tokens=4096):
        """Devuelve el texto de la respuesta, con reintentos ante fallos transitorios."""
        return self._reintentar(
            lambda: self._generar_una_vez(mensajes, sistema, temperatura, max_tokens)
        )

    def _generar_una_vez(self, mensajes, sistema, temperatura, max_tokens):
        if self.proveedor == "groq":
            msgs = ([{"role": "system", "content": sistema}] if sistema else []) + list(mensajes)
            resp = self._cliente.chat.completions.create(
                model=self.modelo,
                messages=msgs,
                temperature=temperatura,
                max_completion_tokens=max_tokens,
            )
            uso = getattr(resp, "usage", None)
            self._registrar_uso(getattr(uso, "prompt_tokens", 0), getattr(uso, "completion_tokens", 0))
            return (resp.choices[0].message.content or "").strip()

        # Gemini
        from google.genai import types
        contents = [
            types.Content(
                role="model" if m["role"] == "assistant" else "user",
                parts=[types.Part(text=m["content"])],
            )
            for m in mensajes
        ]
        cfg = types.GenerateContentConfig(
            system_instruction=sistema,
            temperature=temperatura,
            max_output_tokens=max_tokens,
        )
        resp = self._cliente.models.generate_content(
            model=self.modelo, contents=contents, config=cfg
        )
        meta = getattr(resp, "usage_metadata", None)
        self._registrar_uso(
            getattr(meta, "prompt_token_count", 0), getattr(meta, "candidates_token_count", 0)
        )
        return (resp.text or "").strip()

    # ------------------------------------------------------------------ tool calling
    def chat_tools(self, mensajes, sistema=None, tools=None, temperatura=0.2, max_tokens=4096):
        """Una llamada al LLM con herramientas (tool calling).

        mensajes : historial en formato OpenAI (user / assistant con tool_calls / tool).
        tools    : lista de esquemas {"type": "function", "function": {...}}; None = sin tools
                   (se usa en la última iteración para obligar al modelo a responder).

        Devuelve: {"contenido": str, "tool_calls": [{"id", "nombre", "argumentos"(str JSON)}],
                   "tokens_entrada": int, "tokens_salida": int}
        """
        if self.proveedor != "groq":
            raise NotImplementedError(
                "El agente del Avance 3 usa el tool calling de Groq. Configura LLM_PROVIDER=groq "
                "y GROQ_API_KEY (el pipeline fijo sigue funcionando con Gemini)."
            )
        return self._reintentar(
            lambda: self._chat_tools_groq(mensajes, sistema, tools, temperatura, max_tokens)
        )

    def _chat_tools_groq(self, mensajes, sistema, tools, temperatura, max_tokens):
        msgs = ([{"role": "system", "content": sistema}] if sistema else []) + list(mensajes)
        kwargs = dict(
            model=self.modelo,
            messages=msgs,
            temperature=temperatura,
            max_completion_tokens=max_tokens,
        )
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        resp = self._cliente.chat.completions.create(**kwargs)
        msg = resp.choices[0].message
        uso = getattr(resp, "usage", None)
        t_in = getattr(uso, "prompt_tokens", 0) or 0
        t_out = getattr(uso, "completion_tokens", 0) or 0
        self._registrar_uso(t_in, t_out)
        llamadas = [
            {
                "id": tc.id,
                "nombre": tc.function.name,
                "argumentos": tc.function.arguments or "{}",
            }
            for tc in (getattr(msg, "tool_calls", None) or [])
        ]
        return {
            "contenido": (msg.content or "").strip(),
            "tool_calls": llamadas,
            "tokens_entrada": t_in,
            "tokens_salida": t_out,
        }


def crear_juez_langchain():
    """Chat model de LangChain que usa Ragas como LLM 'juez' (solo para evaluar)."""
    if config.LLM_PROVIDER == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(model=config.GROQ_MODEL, temperature=0, api_key=config.GROQ_API_KEY)

    from langchain_google_genai import ChatGoogleGenerativeAI
    return ChatGoogleGenerativeAI(
        model=config.GEMINI_MODEL, google_api_key=config.GENAI_API_KEY, temperature=0
    )
