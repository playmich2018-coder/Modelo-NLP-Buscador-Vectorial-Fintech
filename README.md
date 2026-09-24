# 🧠 Motor de Búsqueda Semántica (RAG) para Políticas Fintech

## 📖 ¿Para qué sirve y por qué utilizarlo?
En la banca y el sector Fintech, las normativas y políticas de crédito son extensas y complejas. Los motores de búsqueda tradicionales basados en palabras clave (keyword search) fallan cuando el usuario emplea sinónimos o lenguaje coloquial. 
1. **Búsqueda por Conceptos:** Permite encontrar la normativa correcta evaluando la intención de la pregunta, no la coincidencia exacta de palabras.
2. **Base para GenAI:** Es el motor fundacional (Retrieval) para construir asistentes de Inteligencia Artificial (RAG) que respondan preguntas basándose estrictamente en documentos corporativos, evitando alucinaciones.
3. **Automatización de Compliance:** Reduce el tiempo que los analistas y agentes de servicio al cliente invierten buscando reglas en manuales de cientos de páginas.

## 🎯 Objetivo del Proyecto
Desarrollar un sistema de recuperación de información (Retrieval) utilizando Procesamiento de Lenguaje Natural (NLP) para transformar políticas de crédito en bases de datos vectoriales y procesar consultas humanas mediante similitud matemática.

## 🛠 Metodología y Tecnologías
Se implementó un pipeline de vectorización utilizando transformadores de Hugging Face y cálculo de distancias espaciales.
* **Modelo de Lenguaje:** `paraphrase-multilingual-MiniLM-L12-v2` (Optimizado para entender semántica en español e inglés).
* **Motor de Similitud:** Similitud Coseno (Cosine Similarity) vía Scikit-Learn.
* **Base Vectorial:** Serialización de Embeddings utilizando Pickle para almacenamiento ultraligero y consultas en milisegundos.
* **Librerías Clave:** Sentence-Transformers, Pandas, NumPy.

## 🌍 Casos de Uso y Aplicaciones de Negocio
Este motor vectorial rompe la dependencia de las palabras exactas al entender la "intención" del usuario. Su arquitectura es escalable a múltiples industrias:
* **Atención al Cliente (Chatbots Seguros):** Extrae la política exacta para resolver dudas de clientes alimentando sistemas RAG conversacionales que no inventan datos.
* **Legal Tech y Auditoría:** Búsqueda rápida de cláusulas en contratos o normativas financieras usando conceptos legales, sin depender de la jerga exacta.
* **Sistemas de Tickets (IT / Helpdesk):** Conecta las descripciones coloquiales de fallas hechas por los usuarios con los manuales técnicos de diagnóstico de la empresa.
* **E-commerce y Retail:** Muestra productos relevantes por asociación semántica (ej. buscar "ropa para la nieve" sugiere "chaquetas térmicas"), mejorando la conversión.
* **Recursos Humanos:** Automatiza la consulta de manuales de empleado y agiliza el onboarding corporativo.

## 🧠 Resultados y Aplicación de Negocio
El motor logró mapear con éxito consultas coloquiales (ej. "dejar de pagar") hacia normativas técnicas específicas (ej. "mora superior a 90 días, cobro prejurídico"), alcanzando un nivel de confianza matemática superior al 55% en inferencia directa. Este sistema está listo para integrarse como la capa de recuperación de un modelo generativo mayor (LLM).