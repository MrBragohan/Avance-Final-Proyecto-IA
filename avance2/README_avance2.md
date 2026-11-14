# Tutor Académico Personalizado (RAG) — Creación de Aplicaciones con IA

Asistente conversacional que responde preguntas sobre el material de la asignatura
**Creación de Aplicaciones con IA** usando una arquitectura **RAG** (Retrieval-Augmented
Generation). Responde únicamente con la información del corpus, cita el documento y fragmento
que respalda cada respuesta y reconoce cuándo la información no está disponible.

**App desplegada:** `https://proyectoia-b8nfsdkv4w6anf6cv22fed.streamlit.app/`

## Privacidad y control de los datos
 
- Los **embeddings se calculan localmente** (CPU) y el **índice vectorial (Chroma) vive en el
  propio servidor**: el corpus completo nunca se envía a ningún servicio externo.
- Al LLM (Groq o Gemini, configurable) solo se le envían, en cada consulta, los **`top_k` fragmentos
  recuperados** más la pregunta y el historial reciente; nunca el corpus completo.
- Las claves se gestionan con variables de entorno (`.env` en local) o Secrets de la plataforma
  de despliegue. **Nunca** se incluyen en el repositorio.
- Nota: revisa los términos de uso de datos del plan de la API que utilices (en el nivel gratuito
  de algunas APIs el proveedor puede usar los prompts para mejorar sus productos). Si el corpus es
  sensible, usa un plan/proveedor con política de no entrenamiento.
## Arquitectura
 
```
docs/ (PDF, DOCX, TXT, HTML)
   │  ingest.py
   ▼
Extracción de texto → Limpieza de encabezados → Chunking → Embeddings locales → Chroma (coseno)
                                                                                    │
Pregunta ─► Reformulación (si hay historial) ─► Búsqueda top_k ─────────────────────┘
                                                            ▼
              Prompt aumentado (<contexto> + <pregunta> + historial + few-shot)
                                                            ▼
                               LLM ─► Respuesta + fuentes citadas ─► Chat (Streamlit)
```
 
| Archivo | Función |
|---|---|
| `config.py` | Parámetros del pipeline, modelos, juez de Ragas y lectura segura de secretos |
| `ingest.py` | Ingesta multiformato, limpieza, chunking, embeddings e índice Chroma persistente |
| `llm.py` | Acceso al LLM con proveedor intercambiable (Groq o Gemini) |
| `rag_pipeline.py` | Recuperación, reformulación de seguimiento, System Prompt (variantes), Few-Shot y generación |
| `app.py` | Interfaz de chat (Streamlit) con historial y fuentes |
| `evaluate_ragas.py` | Evaluación con Ragas, control de alucinaciones y comparación antes/después |
| `eval_dataset.json` | 20 preguntas con ground truth (16 en el corpus + 4 fuera del corpus) |
| `results/` | CSV, JSON y gráficos de las evaluaciones |
| `avance1/` | Código del Avance 1 (prompt + few-shot con recuperador por palabras clave) |
 
## Decisiones técnicas
 
### 1. Selección de documentos
Corpus de 4 archivos (16 secciones/páginas en total):
 
| Archivo | Contenido | Secciones |
|---|---|---|
| `knowledge_base.txt` | Apuntes de la asignatura: 15 temas (LLMs, Prompt Engineering, Few-Shot, delimitadores, RAG, embeddings, agentes, function calling, fine-tuning, alucinaciones, temperatura, evaluación, multimodalidad, despliegue y ética) | 1 |
| `Introducción_Ingeniería_Prompt.pdf` | Guía de clase: introducción a la ingeniería de prompts | 6 págs. |
| `5 Shot_Prompting.pdf` | Guía de clase: técnicas de Zero/One/Few-Shot Prompting | 3 págs. |
| `6 System_configuration_Chat_Roles.pdf` | Guía de clase: configuración del sistema y roles en chat | 6 págs. |
 
**Pertinencia:** es el material que un estudiante de la asignatura consultaría. Los apuntes dan
cobertura amplia de la materia y las guías en PDF profundizan en las técnicas de prompting
trabajadas en clase, lo que permite citar páginas concretas. El corpus es acotado y verificable,
lo que permite construir un ground truth fiable para la evaluación.
 
### 2. Ingesta
`PyPDFLoader` (PDF, conserva nº de página), `TextLoader` (TXT), `Docx2txtLoader` (DOCX) y
`BSHTMLLoader` (HTML). Cada sección conserva solo metadatos simples (`source`, `page`) para
poder citarla.
 
**Limpieza de encabezados y pies de página:** muchos PDFs repiten en cada página una plantilla
(institución, dirección, versión…) que se cuela en los fragmentos y diluye la similitud. Se eliminan
las líneas de ≥25 caracteres que aparecen (ignorando dígitos) en ≥80 % de las páginas del documento.
El largo mínimo y el umbral alto protegen líneas legítimas que se repiten (por ejemplo, títulos
como "Artículo N:" en un reglamento). En este corpus la limpieza eliminó 1 línea repetida en dos de
los PDFs. Se puede desactivar con `LIMPIAR_ENCABEZADOS = False` en `config.py`.
 
### 3. Chunking
`RecursiveCharacterTextSplitter` con separadores jerárquicos `["\n\n", "\n", ". ", " ", ""]`:
corta primero por párrafo y solo baja de nivel si el fragmento es demasiado grande, respetando
los límites naturales del texto.
- **Parámetros:** `chunk_size=500`, `chunk_overlap=50` (≈10 %). Fragmentos pequeños y precisos,
  compatibles con el límite de entrada del modelo de embeddings, que se solapan lo justo para no
  perder el hilo entre fragmentos consecutivos.
- **Resultado:** las 16 secciones generan **76 fragmentos**.
- Cada fragmento recibe un `chunk_id` por documento para citar "documento + fragmento #n".
### 4. Vectorización
`paraphrase-multilingual-MiniLM-L12-v2` (Sentence Transformers, 384 dimensiones):
multilingüe (buen desempeño en español), ligero (~120 MB), corre en CPU y **local**, lo que
mantiene el corpus bajo nuestro control y evita costos por embeddings. Vectores normalizados.
 
### 5. Base de datos vectorial
**ChromaDB** persistente en `chroma_db/cs{chunk}_ov{overlap}/` con métrica coseno. Un directorio
por configuración permite comparar iteraciones sin pisar el índice. Metadatos almacenados:
`source`, `page`, `chunk_id` (más el texto del fragmento), usados para mostrar las fuentes.
 
### 6. Recuperación
Búsqueda por **similitud coseno** con `top_k=4` fragmentos. Para preguntas de seguimiento
("¿y para qué sirve?") se **reformula primero la pregunta** como una consulta autónoma usando el
historial, y esa consulta es la que se vectoriza.
 
### 7. Generación (refinamiento del Avance 1)
- **LLM:** Groq `openai/gpt-oss-120b`, `temperature=0.2` (el proveedor es intercambiable con
  `LLM_PROVIDER`; ver `llm.py`).
- **System Prompt:** rol, tono, reglas numeradas, formato de salida (Explicación / Ejemplo / Fuente).
  Novedades respecto al Avance 1: citas obligatorias con archivo y página, frase fija de rechazo,
  instrucción de tratar el contexto como *datos* (defensa ante prompt injection) y manejo de
  preguntas de seguimiento. Existe una variante `estricto` (ver *Iteración de mejora*).
- **Few-Shot:** 2 ejemplos precargados: uno con respuesta en el contexto y otro **sin respuesta**
  (rechazo), para reforzar el control de alucinaciones.
- **Delimitadores:** etiquetas XML `<contexto>`, `<fragmento fuente=".." pagina="..">`, `<pregunta>`.
- **Conversación:** se reenvían los últimos 6 turnos; el contexto recuperado solo se inyecta en el
  turno actual (no se acumula) para ahorrar tokens.
## Evaluación con Ragas
 
### Conjunto y método
`eval_dataset.json`: **20 preguntas** con respuesta de referencia (ground truth) elaborada a partir
de los apuntes del corpus: **16 dentro del corpus** y **4 fuera del corpus** (capital de Francia,
lista enlazada en C++, precio de la API de Gemini, Mundial 2022) para medir el control de
alucinaciones.
 
- **Preguntas en el corpus → 4 métricas de Ragas:** `faithfulness`, `answer_relevancy`,
  `context_precision` y `context_recall`.
- **Preguntas fuera del corpus → tasa de rechazo correcto** (la respuesta es la frase de rechazo
  fija) y, en las del corpus, **tasa de falso rechazo**. No pasan por Ragas porque sus métricas no
  son interpretables para una frase de rechazo: faithfulness, precision y recall dan 0 por
  construcción y hunden los promedios. (En una primera corrida que las incluyó, el promedio global de
  `faithfulness` fue 0,55 frente a 0,68 en el corpus, solo por ese efecto.) La opción
  `--incluir-fuera-corpus` permite evaluarlas también con Ragas.
- **Juez:** el mismo modelo que genera (`openai/gpt-oss-120b` vía Groq, `temperature=0`) con
  embeddings locales para `answer_relevancy`. Limitación: usar el mismo modelo como generador y juez
  puede favorecer las métricas (sobre todo `faithfulness`). El juez debe ser el mismo en las corridas
  que se comparan.
### Resultados
 
Línea base: `chunk_size=500`, `overlap=50`, `top_k=4`, prompt `base`.
Iteración de mejora: mismo pipeline con prompt `estricto`.
 
| Métrica (16 preguntas en corpus) | Baseline | Mejora (prompt estricto) | Variación |
|---|---|---|---|
| faithfulness | **0,682** | _completar_ | |
| answer_relevancy | _completar_ ¹ | _completar_ | |
| context_precision | **0,931** | _completar_ | |
| context_recall | **0,885** | _completar_ | |
| Rechazo correcto (4 preguntas fuera del corpus) | **100 %** (4/4) | _completar_ | |
| Falso rechazo (16 preguntas en corpus) | **0 %** (0/16) | _completar_ | |
 
¹ En la primera corrida, `answer_relevancy` falló en 19 de 20 preguntas: Ragas solicita por defecto
3 generaciones (`n=3`) y la API de Groq solo admite `n=1` (error 400). Se corrigió con
`strictness=1` y se recalcula con `python evaluate_ragas.py --completar baseline --metricas answer_relevancy`.
El valor de esa métrica obtenido en la primera corrida (0,904) provenía de una única pregunta y no
se reporta.
 
Comparación gráfica: `results/comparacion_baseline_vs_mejora.png`.
 
### Análisis de la línea base
 
- **Métrica más baja: `faithfulness` (0,68)**, frente a `context_precision` (0,93) y
  `context_recall` (0,89). Es decir, la recuperación entrega casi siempre la información necesaria,
  pero en la respuesta aparecen afirmaciones que el contexto no respalda.
- **Componente atribuido: generación (prompt).** La evidencia está en preguntas con recuperación
  perfecta y fidelidad baja: *temperatura baja* (faithfulness 0,43 con `context_recall` = 1,0 y
  `context_precision` 0,83) y *pilares del Prompt Engineering* (0,47 con ambas = 1,0). La hipótesis es que el formato
  exigido en el System Prompt (sección "Ejemplo" con una analogía) induce al modelo a añadir contenido
  propio que Ragas cuenta como afirmaciones no sustentadas.
- **Componente secundario: recuperación/chunking.** `context_recall` < 1 en tres preguntas
  (*sesgos* 0,0; *RAG* 0,5; *agentes* 0,67) y `context_precision` < 1 en cuatro (la más baja,
  *bases de datos vectoriales*, 0,5): con fragmentos de 500 caracteres y `top_k=4`, la información de un
  mismo tema puede quedar repartida en varios fragmentos y competir con los de los PDFs.
- **Control de alucinaciones:** el asistente rechazó correctamente las 4 preguntas fuera del corpus
  y no rechazó ninguna de las 16 que sí estaban en el corpus.
### Iteración de mejora
 
**Cambio:** prompt `estricto` (único parámetro modificado; el índice y la recuperación no cambian).
Elimina la sección "Ejemplo", exige que cada afirmación sea verificable en los fragmentos y quita el
ejemplo equivalente del Few-Shot.
 
**Hipótesis:** `faithfulness` sube sin afectar `context_precision` ni `context_recall`, porque la
recuperación es idéntica. Si no sube, la causa no era el formato y habría que revisar el juez o el
chunking.
 
```bash
python evaluate_ragas.py --tag baseline --chunk-size 500 --overlap 50 --top-k 4
python evaluate_ragas.py --tag mejora   --chunk-size 500 --overlap 50 --top-k 4 --prompt estricto
python evaluate_ragas.py --comparar baseline mejora
```
 
**Conclusión de la iteración:** _completar con los resultados de la tabla anterior (¿subió
faithfulness?, ¿cambió el rechazo correcto o el falso rechazo?)._
 
### Notas metodológicas
- El juez consume muchos tokens: una evaluación completa agotó el límite diario del plan gratuito de
  Groq (200.000 tokens/día para `gpt-oss-120b`), lo que dejó dos celdas vacías por error 429.
  `--completar` permite recalcular solo lo que falta; `JUDGE_MODEL` permite repartir el consumo entre
  modelos, siempre que se use el mismo juez en las corridas comparadas.
- Cada corrida registra en `results/ragas_<tag>_resumen.json` los parámetros, el prompt, el juez y
  cuántos valores válidos tuvo cada métrica; el script advierte si alguna quedó incompleta.
## Ejecución local
 
```bash
python -m venv venv
.\venv\Scripts\Activate            # Windows  (Mac/Linux: source venv/bin/activate)
pip install -r requirements.txt
copy .env.example .env             # y edita tu clave (GROQ_API_KEY o GENAI_API_KEY)   (Mac/Linux: cp)
streamlit run app.py
```
 
Para evaluar: `pip install -r requirements-eval.txt` y los comandos de la sección anterior.
 
## Despliegue (Streamlit Community Cloud)
 
1. Sube el proyecto a GitHub (el `.gitignore` ya excluye `.env`, `venv/` y `chroma_db/`).
2. En <https://share.streamlit.io> → **Create app** → selecciona el repo, rama y `app.py`.
3. En **Advanced settings**: Python 3.11 y, en **Secrets**, pega:
```toml
   GROQ_API_KEY = "tu_clave_de_groq_aqui"   # o GENAI_API_KEY si usas Gemini
```
   (Si la evaluación demuestra que el prompt `estricto` es mejor, añade `PROMPT_VARIANTE = "estricto"`.)
4. **Deploy**. En el primer arranque se descarga el modelo de embeddings y se construye el índice
   desde `docs/` (tarda unos minutos); luego queda en caché.
 
## Estructura del repositorio
 
```
.
├── app.py  config.py  ingest.py  llm.py  rag_pipeline.py  evaluate_ragas.py
├── eval_dataset.json
├── docs/                      # corpus (apuntes + PDFs de clase)
├── results/                   # CSV, JSON y gráficos de las evaluaciones
├── avance1/                   # entrega anterior
├── requirements.txt  requirements-eval.txt
├── .env.example  .streamlit/secrets.toml.example  .gitignore
└── README.md
```
 