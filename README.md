# 🏦 Microservicio RAG con ChromaDB (Fintech)

## 📖 Arquitectura Empresarial
Evolución del motor de búsqueda semántica hacia una arquitectura escalable de producción. Se reemplazó el almacenamiento estático por **ChromaDB**, una base de datos vectorial dedicada, y se expuso el motor de Inteligencia Artificial mediante un microservicio REST con **FastAPI**.

## 🛠 Tecnologías Implementadas
* **Base de Datos Vectorial:** ChromaDB (Persistencia en disco para embeddings).
* **Backend REST (MLOps):** FastAPI y Uvicorn.
* **Procesamiento de Lenguaje (NLP):** `paraphrase-multilingual-MiniLM-L12-v2` (Hugging Face).
* **Validación Estricta:** Pydantic.

## 🧠 Flujo del Sistema
1. **Población (ETL Vectorial):** Script encargado de vectorizar normativas financieras y almacenarlas físicamente en la colección de ChromaDB.
2. **Inferencia en Tiempo Real:** El endpoint `/buscar_normativa` recibe una duda del usuario, la vectoriza al instante, calcula la distancia espacial contra la base de datos y retorna el fragmento normativo exacto para ser inyectado a un LLM.

## 🚀 Instrucciones de Ejecución

### Opción A: Instalación por Primera Vez
Si es la primera vez que descargas el proyecto, debes crear la base de datos vectorial.
1. Instalar requerimientos: `pip install chromadb fastapi uvicorn sentence-transformers pydantic`
2. Poblar la base de datos: `py 1_poblar_chromadb.py` (Esto creará la carpeta local `/chroma_datos`).
3. Levantar el microservicio: `py -m uvicorn 2_api_rag:app --reload`

### Opción B: Reinicio y Uso Diario
Como ChromaDB guarda los datos de forma persistente, **no necesitas volver a ejecutar el script de población**. Para continuar trabajando en el día a día, simplemente levanta el servidor:
1. Iniciar la API: `py -m uvicorn 2_api_rag:app --reload`
2. Probar consultas en la interfaz web: `http://127.0.0.1:8000/docs`