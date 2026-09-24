from fastapi import FastAPI
from pydantic import BaseModel
import chromadb
from sentence_transformers import SentenceTransformer

# 1. Conectar a la base de datos existente
cliente_chroma = chromadb.PersistentClient(path="./chroma_datos")
coleccion = cliente_chroma.get_collection(name="politicas_bancarias")

# 2. Cargar modelo multilingüe para transformar las preguntas en vivo
modelo_nlp = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

app = FastAPI(title="Microservicio RAG Fintech", description="Endpoint de recuperación vectorial")

class ConsultaCliente(BaseModel):
    pregunta: str

@app.post("/buscar_normativa")
def buscar_politica(consulta: ConsultaCliente):
    # Transformar la pregunta a vector
    vector_pregunta = modelo_nlp.encode([consulta.pregunta]).tolist()
    
    # Consultar nativamente en ChromaDB (traemos el Top 1)
    resultados = coleccion.query(
        query_embeddings=vector_pregunta,
        n_results=1
    )
    
    texto_extraido = resultados['documents'][0][0]
    
    return {
        "pregunta_recibida": consulta.pregunta,
        "contexto_recuperado": texto_extraido,
        # ChromaDB devuelve la distancia, a menor distancia, mayor similitud
        "distancia_matematica": round(resultados['distances'][0][0], 4) 
    }