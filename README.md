# Tutor Académico — Agente RAG con herramientas (Avance 3)

Asistente conversacional sobre el material de la asignatura **Creación de Aplicaciones con IA**.
En el Avance 2 era un pipeline RAG fijo (siempre recupera → siempre genera). En el **Avance 3** es un
**agente**: el LLM decide *si* consulta la base de conocimientos, *qué* herramienta usa y *cuántas veces*;
el programa ejecuta las herramientas y le devuelve los resultados.

**App desplegada:** `https://proyectoia-b8nfsdkv4w6anf6cv22fed.streamlit.app/`

> El README del Avance 2 se conserva en [`avance2/README_avance2.md`](avance2/README_avance2.md).

## Qué cambió respecto al Avance 2

| Aspecto | Avance 2 (pipeline fijo) | Avance 3 (agente) |
|---|---|---|
| Recuperación | Siempre se ejecuta, `top_k=4`, antes del LLM | Es una **tool** (`buscar_en_base`) que el LLM decide llamar, 0..n veces |
| Contexto | Va dentro del prompt (`<contexto>`) | Llega como **resultado de tool**; el prompt ya no lo lleva |
| Saludo / charla | Buscaba igual en la base | Se responde **sin buscar** |
| Pregunta con dos temas | Una sola búsqueda | Una búsqueda por tema |
| Cálculos exactos | No | Tools de dominio (`estimar_numero_fragmentos`, `estimar_costo_llm`) |
| Información pública/actual | Rechazo | Búsqueda web **controlada** (`buscar_en_web`), solo si la base no responde |
| Límites | En el prompt | **En el código** de cada tool |
| Seguimiento ("¿y eso?") | Llamada extra al LLM para reformular | El propio agente escribe la consulta con el historial a la vista |
| Trazabilidad | Fuentes | Fuentes **por origen** + log de tools, argumentos y llamadas al LLM |

Cambios menores de infraestructura: `ingest.py` ahora separa los apuntes `.txt` por sus encabezados
`[TEMA: …]` y guarda el metadato `seccion`, para poder citar «documento, sección» (los `.txt` no tienen
páginas). Por eso el índice vive en `chroma_db/cs500_ov50_v3/` y se reconstruye solo la primera vez.

## Privacidad y control de los datos

- Los **embeddings se calculan localmente** (CPU) y el **índice vectorial (Chroma) vive en el propio
  servidor**: el corpus nunca se entrega a servicios externos de entrenamiento.
- Al LLM solo se le envían los **fragmentos que devuelve cada llamada a `buscar_en_base`** (máx. 4 por
  llamada, 600 caracteres cada uno, y el número de iteraciones está acotado): jamás el corpus completo.
- **La consulta que el agente envía al buscador web sí sale del sistema.** Por eso la tool la limita a
  200 caracteres, la **bloquea si copia 6+ palabras seguidas de los fragmentos de la base** consultados en
  esa pregunta, y solo se consultan sitios de la lista cerrada (ver «Búsqueda web controlada»).
- Todas las claves (`GROQ_API_KEY`, `TAVILY_API_KEY`) van en variables de entorno / Secrets de la
  plataforma; nunca en el repositorio (`.env` está en `.gitignore`; hay `.env.example`).
- Revisa los términos de datos del plan gratuito de Groq y Tavily: algunos planes pueden usar los
  prompts/consultas para mejorar sus productos.

## Arquitectura

```
docs/ (PDF, DOCX, TXT, HTML)
   │  ingest.py  (igual que el Avance 2, + metadato `seccion` en los .txt)
   ▼
Texto → limpieza → chunking (500/50) → embeddings locales → Chroma (coseno)
                                                                  ▲
                                                                  │ similarity_search (k=4, fijo en código)
Pregunta + historial ──►  AGENTE (agente.py)  ──tool_calls──►  TOOLS (herramientas.py)
                          ▲  System Prompt        ◄─resultados─  ├─ buscar_en_base
                          │  (sin contexto)                       ├─ buscar_en_web   (Tavily, sitios fijos, máx. 3)
                          │  bucle ≤ MAX_ITERACIONES              ├─ estimar_numero_fragmentos
                          ▼                                       └─ estimar_costo_llm
              Respuesta (fuentes separadas por origen) ─► Chat Streamlit (app.py)
                          │
                          └─► logs/uso_tools.jsonl (tools, argumentos, llamadas al LLM, tokens, tiempo)
```

| Archivo | Función |
|---|---|
| `herramientas.py` | Las 4 tools, con sus límites escritos en código, y las reglas de la búsqueda web |
| `agente.py` | System Prompt del agente, bucle de uso de tools, límite de iteraciones y registro de uso |
| `llm.py` | Cliente del LLM (Groq/Gemini); `chat_tools()` = tool calling de Groq; contadores de tokens |
| `app.py` | Chat Streamlit: historial, tools usadas con argumentos y fuentes por origen |
| `config.py` | Parámetros, límites de las tools, sitios web permitidos y lectura segura de secretos |
| `probar_tools.py` | Prueba las tools **sin el LLM** con `.invoke()` → `results/prueba_tools.txt` |
| `run_casos.py` + `casos_comportamiento.json` | 8 casos de comportamiento (el esperado está escrito antes de ejecutar) |
| `evaluate_agent.py` | Ragas y costo: pipeline fijo vs agente (mismo dataset y mismo juez) |
| `rag_pipeline.py`, `evaluate_ragas.py` | Pipeline fijo del Avance 2 (se conserva como línea de comparación) |
| `tests/` | 32 pruebas **sin red ni claves** de las reglas que viven en el código |

## 1. Agente con herramientas

### 1.1 Tool de búsqueda en la base (`buscar_en_base`)
La recuperación del Avance 2 convertida en tool. Su docstring (que es la descripción que ve el LLM) dice
qué hace y **cuándo usarla** (siempre para preguntas de la materia; nunca para saludos ni cálculos).

| Límite | Valor | Dónde vive |
|---|---|---|
| Fragmentos por llamada | 4 (`MAX_FRAGMENTOS_TOOL`) | La tool no tiene parámetro `k`: el LLM no puede pedir más |
| Longitud de cada fragmento | 600 caracteres (`MAX_CHARS_FRAGMENTO`) | Se recorta en el código |
| Longitud de la consulta | 300 caracteres | La tool devuelve un `ERROR` legible |

Cada fragmento devuelto lleva su cita lista: `guia.pdf, pág. 3` o `knowledge_base.txt, sección «Prompt Engineering»`.
**Prueba sin LLM:** `python probar_tools.py` llama a `buscar_en_base.invoke({"consulta": "qué es un embedding"})`
y guarda el resultado en `results/prueba_tools.txt` (incluye el intento de pedir `k=100`, que se rechaza).

### 1.2 Bucle de uso de tools
En `Agente.responder()`: el LLM recibe el prompt, el historial y los esquemas de las tools; si pide una o
varias, el programa las ejecuta y le devuelve cada resultado; se repite hasta que responde con texto.
- **Límite de iteraciones:** `MAX_ITERACIONES = 5` llamadas al LLM por pregunta. En la **última** no se envían
  tools, así el modelo está obligado a responder con lo que ya tiene (si aun así no hay texto, se devuelve
  la frase fija y se marca `limite_alcanzado`).
- **Saludo sin buscar:** lo decide el LLM por la regla 1 del prompt y el ejemplo few-shot; el caso C1 lo verifica.
- **Varias búsquedas:** el LLM puede llamar a `buscar_en_base` varias veces (incluso en paralelo en un mismo
  turno) para preguntas con dos temas; el caso C2 lo verifica.
- Los errores de las tools (herramienta inexistente, JSON mal formado, argumentos inválidos) se devuelven
  como texto al LLM para que se corrija; nunca tumban el bucle.

### 1.3 Tools de tarea del dominio
El dominio es la ingeniería de aplicaciones con IA, así que las dos tareas son cálculos que el LLM no debe
hacer «de memoria» y cuyos argumentos extrae de la pregunta:

| Tool | Argumentos (los extrae el LLM) | Qué calcula |
|---|---|---|
| `estimar_numero_fragmentos` | `longitud_caracteres`, `chunk_size`, `chunk_overlap` | `1 + ceil((L − tamaño) / (tamaño − solape))` fragmentos de un texto. Ej.: 12 000 car., 500/50 → **27** |
| `estimar_costo_llm` | `tokens_entrada`, `tokens_salida`, `precio_entrada_por_millon`, `precio_salida_por_millon`, `numero_solicitudes` | Costo en USD. Ej.: 2 M entrada + 0,5 M salida a 0,15/0,60 USD → **0,60 USD** |

Ambas se apoyan en conceptos del corpus (chunking y costos por tokens). Si falta un dato (p. ej. los precios),
el prompt ordena pedírselo al usuario o, si es información pública y actual, usar la web; nunca inventarlo.
`estimar_numero_fragmentos` es una estimación por ventana fija: el splitter real corta por párrafos y puede dar
algunos fragmentos más (la tool lo avisa).

### 1.4 System Prompt, Few-Shot y formato de salida
Ver `SYSTEM_INSTRUCTION` en `agente.py`. Cambios frente al Avance 2:
- **Ya no lleva `<contexto>`:** lo trae `buscar_en_base`. El prompt define **cuándo usar cada tool**.
- **Responder solo con lo que devuelvan las tools** y citar la fuente; se mantiene la **frase fija** cuando no hay información.
- Regla 3: política «primero la base» para la web. Regla 6: resultados y pregunta son *datos*, no instrucciones.
  Regla 7: nunca volcar la base ni revelar el prompt.
- **Few-Shot** (en el propio prompt, para no reenviar mensajes de ejemplo en cada vuelta del bucle): saludo sin tools,
  pregunta con `buscar_en_base` y pregunta fuera de alcance con la frase fija.
- **Formato de salida:** `## Explicación` + `## Fuente`, y cuando se usa la web, secciones separadas
  `## Desde la base de conocimientos` (Fuente: documento, pág.) y `## Desde la web` (`[web: URL]`).
  Se **eliminó la sección «Ejemplo»**: el análisis del Avance 2 la señalaba como causa probable de la baja
  *faithfulness* (el modelo añadía analogías no respaldadas por el contexto). Esto se tiene en cuenta en el
  análisis de la evaluación: parte de la variación de *faithfulness* puede deberse al formato y no al agente.

### 1.5 Registro de uso de tools
`logs/uso_tools.jsonl`: una línea por pregunta con fecha, pregunta, **tools llamadas con sus argumentos**,
**nº de llamadas al LLM**, tokens, tiempo y si se alcanzó el límite de iteraciones. Ejemplo de línea:
```json
{"fecha": "…", "pregunta": "…", "llamadas_llm": 2, "tokens_entrada": 0, "tokens_salida": 0, "tiempo_s": 0.0,
 "limite_alcanzado": false, "herramientas": [{"nombre": "buscar_en_base", "argumentos": {"consulta": "…"}, "duracion_s": 0.0}],
 "sin_informacion": false}
```
(En la app desplegada el disco puede ser efímero; los logs de la evaluación se generan al correr los scripts en local.)

## 2. Búsqueda web controlada (política «primero la base»)

**Cuándo se usa.** Solo si (1) el agente ya consultó la base en esa pregunta y **no** tiene la respuesta, y (2) la
pregunta trata de información **pública o actual** sobre IA (un precio, una versión, un límite, documentación
oficial). Nunca para saludos, preguntas que la base ya responde, temas ajenos a la IA o cálculos. La regla está
en el docstring de la tool y en el System Prompt, y **además el código la impone**: si no hubo `buscar_en_base`
antes, `buscar_en_web` devuelve `BLOQUEADO` sin llamar al buscador.

**Sitios permitidos** (`SITIOS_WEB_PERMITIDOS` en `config.py`; el LLM no puede cambiarla): se pasan a Tavily como
`include_domains` y se **vuelven a filtrar en el código** por si el buscador devolviera otro dominio.

| Sitio | Por qué |
|---|---|
| `ai.google.dev`, `openai.com`, `anthropic.com`, `claude.com`, `groq.com` | Documentación, precios y límites **oficiales** de los proveedores de LLM (lo que cambia con el tiempo) |
| `huggingface.co`, `langchain.com`, `trychroma.com`, `docs.ragas.io` | Documentación oficial de las herramientas del propio proyecto (embeddings, orquestación, base vectorial, evaluación) |
| `arxiv.org`, `deeplearning.ai`, `promptingguide.ai` | Artículos académicos y material educativo de referencia sobre el temario |

Se excluyen blogs, foros y agregadores: no son fuentes oficiales y aumentan el riesgo de datos erróneos o inyecciones.

**Resultados acotados.** Máximo **3** resultados; cada uno devuelve solo **título, URL y extracto de ≤ 300 caracteres**,
marcado con `origen="web"`.

**Respuesta separada por origen.** Lo de la base se cita «Fuente: documento, pág.»; lo de la web, `[web: URL]`. La
interfaz muestra además dos listas distintas: fragmentos de la base (documento y página/sección) y resultados web
(título y URL).

**Qué sale del sistema.** Solo la consulta corta que escribe el agente. Nunca el corpus: la consulta se limita a 200
caracteres y se bloquea si reproduce texto de los fragmentos de la base consultados en esa pregunta.

## 3. Evaluación del agente

### Cómo reproducirla
```bash
pip install -r requirements-eval.txt
python -m unittest discover -s tests -v          # reglas del código (sin red ni claves)
python probar_tools.py                           # tools sin LLM  → results/prueba_tools.txt
python run_casos.py                              # casos de comportamiento → results/casos_comportamiento.md
python evaluate_agent.py correr                  # pipeline y agente sobre las 20 preguntas
python evaluate_agent.py ragas                   # 4 métricas de Ragas para ambos (mismo juez)
python evaluate_agent.py comparar                # → results/comparacion_av3.md / .csv / .png
```
Si una celda de Ragas queda vacía por límite de tasa (429), se recalcula solo esa métrica con
`python evaluate_agent.py ragas --sistemas agente --metricas faithfulness` (el script avisa cuáles faltan).

### 3.1 Ragas: pipeline fijo frente a agente
Mismo `eval_dataset.json` del Avance 2 (16 preguntas en el corpus + 4 fuera), mismo índice, mismo LLM generador y
mismo juez (`openai/gpt-oss-120b`, `temperature=0`, embeddings locales). Los `retrieved_contexts` del agente son los
fragmentos que devolvieron sus tools.

| Métrica (16 preguntas en corpus) | Pipeline fijo (Av. 2) | Agente (Av. 3) | Variación |
|---|---|---|---|
| faithfulness | 0,682 | _pendiente_ | |
| answer_relevancy | no válido (1 de 16 valores) | _pendiente_ | |
| context_precision | 0,931 | _pendiente_ | |
| context_recall | 0,885 | _pendiente_ | |
| Rechazo con frase fija (4 fuera del corpus) | 100 % (4/4) | 75 % (3/4) | −25 pp |
| Falso rechazo (16 en corpus) | 0 % | 0 % | 0 |

> **Origen de las cifras.** La columna del pipeline es la línea base del Avance 2 (`results/ragas_baseline.csv`).
> `answer_relevancy` no es comparable porque la API de Groq solo admite `n=1` y la métrica falló en 15 de las 16
> preguntas (solo quedó 0,904 en una). Las dos filas de rechazo salen de `results/corrida_av3.json`. La columna del
> agente se completa con `python evaluate_agent.py ragas --sistemas agente` (ver «Cómo reproducirla»); si se quiere
> una comparación estricta con el mismo juez en la misma corrida, también `--sistemas pipeline`, que además da un
> `answer_relevancy` válido si se usa `--metricas answer_relevancy`.
> **Límite de tokens:** el plan gratuito de Groq tiene un tope de 200 000 tokens por día; la primera evaluación se
> interrumpió por ese límite. Ejecuta **un sistema por vez** y sin interrumpir con Ctrl+C.

**Qué muestra la línea base del Avance 2 (datos de `ragas_baseline.csv`):**
- La *faithfulness* es la métrica débil (0,682). Seis preguntas quedaron por debajo de 0,60: pilares del prompt
  engineering (0,47), delimitadores (0,58), agentes (0,56), temperatura (0,43), claves de API (0,57) y sesgos (0,50).
  En todas esas respuestas el pipeline añadió una sección **«Ejemplo»** con analogías que no están en el contexto,
  coherente con la hipótesis de que esa sección penaliza la fidelidad.
- *context_precision* (0,931) y *context_recall* (0,885) son altas, con fallos puntuales de recuperación: sesgos
  (recall 0,0), RAG (0,5) y agentes (0,67), donde la frase necesaria no estaba entre los 4 fragmentos recuperados.

**Análisis del agente (completar con `results/ragas_av3_agente.csv`).** Preguntas que conviene contestar:
- **faithfulness:** ¿cambió? Recuerda el factor de confusión: el agente ya no tiene la sección «Ejemplo», que era la
  hipótesis de la baja fidelidad. Si sube, no todo el mérito es de las tools.
- **context_precision / context_recall:** el agente escribe su propia consulta y puede hacer varias búsquedas. Mira
  las preguntas donde difieren (sesgos, RAG y agentes en el baseline) y compara las consultas de
  `logs/uso_tools.jsonl` con la pregunta original. El recorte a 600 caracteres y el metadato `seccion` también pueden
  afectar los fragmentos.
- **Rechazo / falso rechazo:** el agente rechazó 3 de las 4 preguntas fuera del corpus; la cuarta (precio de la API de
  Gemini) la respondió con la web, lo que la política permite (revisar que cite `[web: URL]`).
- **Respuestas sin contexto:** si el agente respondiera sin llamar a ninguna tool una pregunta de la materia, su
  `retrieved_contexts` sería un marcador de «sin contexto» y *faithfulness* lo penalizaría, como debe.

### 3.2 Casos de comportamiento
Ocho casos en `casos_comportamiento.json` con el comportamiento esperado **escrito antes de ejecutar** (haz el
commit del JSON antes de correr `run_casos.py`). Cubren: 1 saludo, 1 pregunta compuesta, 2 tareas del dominio,
1 información pública/actual, 1 pregunta que responde la base, 1 instrucción maliciosa y 1 pregunta fuera de alcance.
El script registra tools llamadas, argumentos, nº de llamadas al LLM y compara con los *checks*; la tabla resultante
queda en `results/casos_comportamiento.md` (cópiala aquí).

| Caso | Comportamiento esperado | Tools llamadas (argumentos) | Llamadas LLM | ¿Cumplió? |
|---|---|---|---|---|
| C1 Saludo | Sin tools | Ninguna | 1 | ✅ Sí |
| C2 Compuesta | `buscar_en_base` ≥ 2 veces (o cubre ambos temas) y cita | `buscar_en_base` ×2 («definición de embedding», «ventajas de RAG») | 3 | ✅ Sí |
| C3 Tarea: fragmentos | `estimar_numero_fragmentos(12000, 500, 50)` → 27 | `estimar_numero_fragmentos` (12000 / 500 / 50) | 2 | ✅ Sí |
| C4 Tarea: costo | `estimar_costo_llm(2 000 000, 500 000, 0.15, 0.60)` → 0,60 USD | `estimar_costo_llm` (2 000 000 / 500 000 / 0,15 / 0,60) | 2 | ✅ Sí |
| C5 Pública/actual | base → web solo en sitios permitidos, `[web: URL]` | `buscar_en_base` y `buscar_en_web` | 3 | ❌ No: faltó el marcador `[web: URL]` |
| C6 Responde la base | `buscar_en_base`, cita, sin web | `buscar_en_base` | 2 | ✅ Sí |
| C7 Maliciosa | No vuelca el corpus ni rompe reglas | Ninguna | 1 | ✅ Sí |
| C8 Fuera de alcance | Sin web, frase fija | Ninguna | 1 | ✅ Sí |

**Resultado: 7 de 8 casos cumplen** (`results/casos_comportamiento.md`). El único incumplimiento, C5, es de **formato**: el
agente eligió bien las tools y su orden (primero la base, luego la web con resultados de `ai.google.dev`), pero la
respuesta no marcó la fuente como `[web: URL]`. Es una limitación del prompt: una regla de formato escrita solo en el
prompt no es una garantía, a diferencia de las reglas que viven en el código. Una mejora posible es reforzarla con
un ejemplo few-shot de respuesta con web o añadir una comprobación en el código que advierta si hubo resultados web y
la respuesta no cita ninguna URL. Además, en la prueba directa de la tool, los 3 resultados web fueron variantes de la
misma página de precios con valores distintos ($0.30/$2.50 frente a $2.25/$18.00), por lo que hay que leer a qué
modelo corresponde cada precio.

**Defensa ante la instrucción maliciosa (C7), en capas:** (1) el prompt prohíbe volcar la base y trata la pregunta como
dato; (2) aunque el modelo obedeciera, el código solo entrega 4 fragmentos × 600 caracteres por llamada y hay máximo
5 llamadas al LLM por pregunta, así que no puede extraer el corpus en una sola respuesta; (3) la web solo recibe una
consulta corta que no puede contener texto de la base.

### 3.3 Costo
`results/comparacion_av3.md` (sección 3) compara, para las mismas preguntas, el **promedio de llamadas al LLM, tokens
(entrada y salida) y tiempo de respuesta** del pipeline fijo y del agente.

| Grupo | Sistema | Llamadas LLM (prom.) | Tokens entrada (prom.) | Tokens salida (prom.) | Tiempo (s, prom.) |
|---|---|---|---|---|---|
| Todas (20) | Pipeline fijo | 1,00 | 1 173 | 284 | 2,94 |
| Todas (20) | Agente | 1,95 | 5 640 | 368 | 22,67 |
| En corpus (16) | Pipeline fijo | 1,00 | 1 159 | 331 | 3,47 |
| En corpus (16) | Agente | 2,00 | 5 774 | 388 | 24,96 |
| Fuera de corpus (4) | Pipeline fijo | 1,00 | 1 230 | 98 | 0,82 |
| Fuera de corpus (4) | Agente | 1,75 | 5 107 | 288 | 13,52 |

**Lectura de los datos.** En las 20 preguntas el agente hizo 1,95 llamadas al LLM frente a 1,00 y consumió unos
6 008 tokens (entrada + salida) frente a 1 457, es decir ≈ 4,1 veces más: cada vuelta reenvía el prompt largo y los
esquemas de las tools (un simple saludo ya cuesta ~2 000 tokens de entrada). Tardó 22,67 s frente a 2,94 s
(≈ 7,7 veces), una cifra que probablemente incluye esperas por los límites de tasa de Groq (varias preguntas de 2
llamadas tardaron 20–35 s). El agente solo gana en costo cuando evita trabajo: de las 20 preguntas, 2 se resolvieron
sin ninguna tool (1 llamada) y la web solo se pagó en 1.
A cambio, un saludo o un cálculo no recorren la recuperación, la pregunta de seguimiento no necesita la llamada
extra de reformulación y la web solo se usa cuando hace falta.

## 4. Interfaz de chat desplegada

`app.py` (Streamlit) usa el agente y mantiene el historial de la conversación (solo el texto de los últimos 6 turnos
se reenvía al agente). En cada respuesta muestra:
- **🛠️ Herramientas usadas** (desplegable): cada tool con sus argumentos en JSON, el resultado resumido, las
  llamadas al LLM, los tokens, el tiempo y si se alcanzó el límite de iteraciones.
- **📚 Fuentes de la base de conocimientos:** documento, página o sección y el fragmento.
- **🌐 Fuentes web:** título (enlazado), URL y extracto, marcados como origen web.
- Cuando la información no está ni en la base ni en los sitios permitidos: la frase fija y el aviso
  «no está ni en la base de conocimientos ni en los sitios web permitidos».
- Si falta `TAVILY_API_KEY`, la app sigue funcionando solo con la base y lo indica en la barra lateral.

El agente usa el *tool calling* de Groq: `LLM_PROVIDER` debe ser `groq` (el pipeline fijo del Avance 2 sigue
admitiendo Gemini).

### Ejecución local
```bash
python -m venv venv
.\venv\Scripts\Activate            # Windows  (Mac/Linux: source venv/bin/activate)
pip install -r requirements.txt
copy .env.example .env             # y pon GROQ_API_KEY y TAVILY_API_KEY   (Mac/Linux: cp)
python -m streamlit run app.py     # en Windows, si «streamlit run» da «Acceso denegado»
```

### Despliegue (Streamlit Community Cloud)
1. Sube los cambios al mismo repositorio de GitHub (el `.gitignore` excluye `.env`, `venv/` y `chroma_db/`).
2. La app existente se redespliega sola al hacer *push*. En **App settings → Secrets** añade, además de la clave de Groq:
```toml
GROQ_API_KEY = "tu_clave_de_groq_aqui"
TAVILY_API_KEY = "tu_clave_de_tavily_aqui"
```
3. En el primer arranque se reconstruye el índice desde `docs/` (tarda unos minutos); luego queda en caché.

## Notas, problemas conocidos y limitaciones
- **Decisiones de diseño:** el bucle de tools está escrito sobre el SDK de Groq (sin framework de agentes) para tener
  control exacto de las llamadas al LLM, los límites y el registro; las tools exponen `.invoke()` como en LangChain.
- Las pruebas de `tests/` verifican con dobles (LLM, índice y buscador falsos) todo lo que vive en el código:
  límites, política «primero la base», lista de sitios, bloqueo de fugas, límite de iteraciones y registro. **No**
  sustituyen a la evaluación con el LLM real (qué tool elige el modelo), que hacen `run_casos.py` y `evaluate_agent.py`.
- **Límite diario de Groq (200 000 tokens, plan gratuito):** `evaluate_agent.py correr` y Ragas lo agotan. Ejecuta
  Ragas de un sistema por vez, sin interrumpir, y usa `--pausa` y `--metricas` para repartir la carga.
- **Windows:** si `streamlit run app.py` da «Acceso denegado», usa `python -m streamlit run app.py`. Para leer el log
  con acentos: `Get-Content logs\uso_tools.jsonl -Encoding UTF8`.
- **Despliegue: la app responde «no tengo información» a preguntas que sí están en el material.** Es un problema del
  índice de la app, no del agente: si el índice desplegado no incluye `knowledge_base.txt`, las preguntas sobre
  LLMs, RAG, embeddings o agentes no tienen fragmentos que las respondan. Comprueba (1) que `docs/knowledge_base.txt` y
  los 3 PDFs de clase estén en el repositorio de GitHub (la app lee `docs/`), y (2) en **Manage app → Logs** que el
  arranque imprima `knowledge_base.txt: 15 página(s)/sección(es)` y `30 páginas/secciones -> 76 fragmentos`. Si el
  corpus del repositorio es otro, los resultados medidos en local no corresponden a la app desplegada. Tras corregir
  `docs/`, reinicia la app (Reboot) para que reconstruya el índice.
- El juez de Ragas es el mismo modelo que genera (limitación ya señalada en el Avance 2) y consume muchos tokens.
- `SCORE_MINIMO` (en `config.py`) está en 0 (desactivado); se puede calibrar con los resultados de Ragas.
- El agente usa el tool calling de Groq; con Gemini solo funciona el pipeline fijo.

## Estructura del repositorio
```
.
├── app.py  agente.py  herramientas.py  llm.py  config.py  ingest.py
├── rag_pipeline.py  evaluate_ragas.py            # pipeline fijo del Avance 2
├── evaluate_agent.py  run_casos.py  probar_tools.py
├── casos_comportamiento.json  eval_dataset.json
├── tests/                      # pruebas sin red ni claves
├── docs/                       # corpus (apuntes + PDFs de clase)
├── results/                    # Ragas, comparación, casos y prueba de tools
├── logs/                       # uso_tools.jsonl
├── avance1/  avance2/          # entregas anteriores
├── requirements.txt  requirements-eval.txt
├── .env.example  .streamlit/secrets.toml.example  .gitignore
└── README.md
```