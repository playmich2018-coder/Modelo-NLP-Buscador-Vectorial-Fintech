import pandas as pd
from sentence_transformers import SentenceTransformer
import pickle

# 1. Simular la base de conocimientos (Políticas bancarias y compliance)
documentos = [
    "Para créditos hipotecarios, el solicitante debe tener una antigüedad laboral mínima de 2 años continuos.",
    "El límite de endeudamiento no debe superar el 40% de los ingresos netos mensuales del cliente.",
    "Las tarjetas de crédito corporativas requieren una revisión anual de los estados financieros de la empresa.",
    "En caso de mora superior a 90 días, la cuenta pasará a departamento de cobro prejurídico.",
    "Se aprobará un incremento de cupo solo si el cliente tiene un puntaje crediticio superior a 750 puntos."
]

df_politicas = pd.DataFrame({"ID_Politica": range(1, len(documentos) + 1), "Texto": documentos})

# 2. Cargar el modelo de NLP (Transforma texto en vectores)
print("⏳ Cargando modelo de lenguaje de Hugging Face (puede tardar un minuto la primera vez)...")
# Usamos 'all-MiniLM-L6-v2' por ser extremadamente rápido y eficiente para entornos de producción
modelo_nlp = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# 3. Generar Embeddings (Representación matemática del texto)
print("🧠 Convirtiendo políticas financieras a vectores semánticos...")
vectores = modelo_nlp.encode(df_politicas['Texto'].tolist())

# 4. Guardar la "Base de Datos Vectorial" empaquetada
with open('vectores_fintech.pkl', 'wb') as f:
    pickle.dump({'textos': df_politicas, 'embeddings': vectores}, f)

print("✅ Base de conocimiento procesada y empaquetada como 'vectores_fintech.pkl'")