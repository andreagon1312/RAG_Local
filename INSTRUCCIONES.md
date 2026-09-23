# Instrucciones para ejecutar el Sistema RAG Local

Sigue estos pasos en orden para poner en marcha tu sistema:

## 1. Instalar dependencias de Python
Abre una terminal en esta carpeta (`C:\Users\impor\.gemini\antigravity\scratch\rag_project`) y ejecuta:
```bash
pip install -r requirements.txt
```

## 2. Iniciar vLLM (Modelo de Embeddings Local)
Necesitas arrancar vLLM con el modelo BAAI. Dado que estás en Windows, si usas WSL (Windows Subsystem for Linux), puedes ejecutar el siguiente comando. (Si tienes Docker, también puedes correr vLLM con Docker).
Abre una terminal nueva y ejecuta:
```bash
python -m vllm.entrypoints.openai.api_server \
  --model BAAI/bge-small-en-v1.5 \
  --task embed \
  --device cpu \
  --dtype float32 \
  --port 8000
```
*Deja esta terminal abierta y corriendo.*

## 3. Iniciar Milvus Local (Base de datos vectorial)
Abre otra terminal y usa Docker para levantar la versión "Standalone" de Milvus usando el script oficial. (Asegúrate de tener Docker Desktop corriendo en Windows):
```bash
curl -sfL https://raw.githubusercontent.com/milvus-io/milvus/master/scripts/standalone_embed.sh -o standalone_embed.sh
bash standalone_embed.sh start
```
Esto levantará los contenedores de Milvus y los expondrá en el puerto `19530`.

## 4. Ingestar tu PDF
Copia o mueve tu archivo PDF a esta misma carpeta. Luego ejecuta el script de ingesta (reemplaza `tu_archivo.pdf` con el nombre real de tu PDF):
```bash
python ingest.py tu_archivo.pdf
```
Verás mensajes en consola confirmando que se conectó a vLLM, generó los vectores y los guardó en Milvus.

## 5. Iniciar el chat (Bucle RAG)
Finalmente, ejecuta el script del chat:
```bash
python rag.py
```
Esto abrirá una interfaz en la consola. Podrás hacerle preguntas sobre tu documento, las cuales usarán los embeddings de vLLM recuperados de Milvus, y generarán la respuesta enviando el contexto a DeepSeek (usando tu API Key). Escribe `EXIT` para salir.
