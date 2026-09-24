import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os

if not os.path.exists('vectores_fintech.pkl'):
    print("⚠️ Ejecuta primero: py generar_base_conocimiento.py")
else:
    # ---------------------------------------------------------
    # FASE 1: RETRIEVAL (Recuperación Vectorial)
    # ---------------------------------------------------------
    with open('vectores_fintech.pkl', 'rb') as f:
        datos = pickle.load(f)
        
    df_politicas = datos['textos']
    vectores_documentos = datos['embeddings']
    
    modelo_nlp = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

    pregunta_cliente = "¿Qué sucede si un cliente deja de pagar por varios meses?"
    vector_pregunta = modelo_nlp.encode([pregunta_cliente])
    
    similitudes = cosine_similarity(vector_pregunta, vectores_documentos)[0]
    indice_mejor = similitudes.argmax()
    politica_extraida = df_politicas.iloc[indice_mejor]['Texto']
    
    # ---------------------------------------------------------
    # FASE 2: GENERATION (Inyección de Contexto al LLM)
    # ---------------------------------------------------------
    # Aquí es donde se conecta la API de un LLM en producción
    
    prompt_backend = f"""
    Eres un asistente virtual de un banco. Responde la duda del cliente de forma amable, clara y profesional.
    
    REGLA ESTRICTA: Basa tu respuesta ÚNICAMENTE en el siguiente contexto normativo. Bajo ninguna circunstancia inventes información. Si el contexto no responde la pregunta, indica que debes transferir al cliente con un asesor humano.
    
    CONTEXTO NORMATIVO: {politica_extraida}
    
    PREGUNTA DEL CLIENTE: {pregunta_cliente}
    """

    # Simulación de la llamada a la API del LLM (ej. openai.ChatCompletion.create)
    def llamar_llm_api(prompt):
        return "Hola. Lamento informarle que, según nuestras políticas, si presenta una mora superior a 90 días, su cuenta será transferida automáticamente a nuestro departamento de cobro prejurídico. Le sugerimos regularizar su situación lo antes posible para evitar inconvenientes."

    respuesta_final_ia = llamar_llm_api(prompt_backend)

    # ---------------------------------------------------------
    # RESULTADOS
    # ---------------------------------------------------------
    print("👤 PREGUNTA DEL CLIENTE:")
    print(pregunta_cliente)
    print("\n" + "="*75)
    print("⚙️  LO QUE VE EL SISTEMA POR DETRÁS (Prompt de Backend con Contexto):")
    print("="*75)
    print(prompt_backend)
    print("="*75)
    print("🤖 RESPUESTA FINAL DEL ASISTENTE AL CLIENTE:")
    print("="*75)
    print(respuesta_final_ia)