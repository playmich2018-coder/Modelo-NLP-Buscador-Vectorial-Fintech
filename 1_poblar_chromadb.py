import chromadb
from sentence_transformers import SentenceTransformer

print("⏳ Inicializando motor vectorial ChromaDB...")
# Crea una base de datos real en la carpeta local './chroma_datos'
cliente_chroma = chromadb.PersistentClient(path="./chroma_datos")

# Crear una "tabla" (colección) para las normativas
coleccion = cliente_chroma.get_or_create_collection(name="politicas_bancarias")

documentos = [
    "Para créditos hipotecarios, el solicitante debe tener una antigüedad laboral mínima de 2 años continuos.",
    "El límite de endeudamiento no debe superar el 40% de los ingresos netos mensuales del cliente.",
    "Las tarjetas de crédito corporativas requieren una revisión anual de los estados financieros de la empresa.",
    "En caso de mora superior a 90 días, la cuenta pasará a departamento de cobro prejurídico.",
    "Se aprobará un incremento de cupo solo si el cliente tiene un puntaje crediticio superior a 750 puntos."
]
ids = [f"doc_{i}" for i in range(len(documentos))]

print("🧠 Vectorizando documentos con modelo multilingüe...")
modelo_nlp = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
vectores = modelo_nlp.encode(documentos).tolist()

print("💾 Inyectando registros en ChromaDB...")
coleccion.add(
    documents=documentos,
    embeddings=vectores,
    ids=ids
)

print("✅ Base de datos poblada exitosamente en la carpeta '/chroma_datos'")