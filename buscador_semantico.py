import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os

if not os.path.exists('vectores_fintech.pkl'):
    print("⚠️ Faltan los datos vectoriales. Ejecuta primero: py generar_base_conocimiento.py")
else:
    # 1. Cargar la base de datos vectorial
    print("⏳ Iniciando motor de búsqueda semántica...")
    with open('vectores_fintech.pkl', 'rb') as f:
        datos = pickle.load(f)
        
    df_politicas = datos['textos']
    vectores_documentos = datos['embeddings']
    
    modelo_nlp = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

    # 2. Definir una pregunta humana (sin palabras clave exactas)
    pregunta = "¿Qué sucede si un cliente deja de pagar por varios meses?"
    print(f"\n👤 Pregunta ingresada: '{pregunta}'")
    
    # 3. Convertir la pregunta al mismo espacio matemático (Embedding)
    vector_pregunta = modelo_nlp.encode([pregunta])
    
    # 4. Calcular Similitud Coseno (buscar el texto más cercano conceptualmente)
    similitudes = cosine_similarity(vector_pregunta, vectores_documentos)[0]
    
    # 5. Extraer el mejor resultado (Top 1)
    indice_mejor_resultado = similitudes.argmax()
    porcentaje_confianza = similitudes[indice_mejor_resultado] * 100
    politica_encontrada = df_politicas.iloc[indice_mejor_resultado]['Texto']
    
    # 6. Mostrar el resultado de la IA
    print("-" * 75)
    print("🤖 RESULTADO DEL MOTOR RAG (Búsqueda Vectorial)")
    print("-" * 75)
    print(f"📄 Política extraída: {politica_encontrada}")
    print(f"🎯 Nivel de confianza matemática: {porcentaje_confianza:.1f}%")
    print("-" * 75)