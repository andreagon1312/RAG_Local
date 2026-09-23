# Sistema RAG Local — Asistente de Documentos con vLLM, Milvus y DeepSeek

> **Construye tu propio sistema RAG local**  
> Implementación de un pipeline de Generación Aumentada por Recuperación (RAG) completamente local para consultar documentos PDF, utilizando embeddings locales, una base de datos vectorial y un LLM remoto.

---

## 📖 Descripción General

Este proyecto implementa un **sistema RAG (Retrieval-Augmented Generation) local** que permite realizar preguntas en lenguaje natural sobre un documento PDF. El sistema combina:

1. **Embeddings locales** generados con `BAAI/bge-small-en-v1.5` (ejecutado vía vLLM o localmente).
2. **Base de datos vectorial Milvus** para almacenar y recuperar los fragmentos (chunks) del documento.
3. **LLM DeepSeek** como modelo generativo para producir respuestas contextualizadas.
4. **LlamaIndex** como framework de orquestación del pipeline RAG.

El flujo es completamente interactivo: el usuario ingesta un PDF, y luego puede chatear con él desde la consola, obteniendo respuestas en streaming basadas únicamente en el contenido del documento.

---

## 🏗️ Arquitectura del Sistema

### Pipeline de Ingesta
 Archivo PDF
│
▼
[ LlamaIndex ] ── Chunking y lectura
│
▼
[ vLLM / HuggingFace ] ── Embeddings (BAAI/bge-small-en-v1.5, CPU)
│
▼
[ Milvus ] ── Almacenamiento de vectores


### Pipeline RAG
👤 Usuario (consola)
│
▼
[ LlamaIndex ] ── Orquestación RAG
│
▼
[ vLLM / HuggingFace ] ── Embedding de la consulta
│
▼
[ Milvus ] ── Recuperación de chunks relevantes (Top-K)
│
▼
[ DeepSeek API ] ── Generación de respuesta con contexto
│
▼
✅ Respuesta en consola

---

## ✨ Características Principales

- 🔒 **Procesamiento local de embeddings:** el modelo `BAAI/bge-small-en-v1.5` corre localmente, sin enviar datos a la nube para la vectorización.
- 🗄️ **Milvus como vector store:** base de datos vectorial de alto rendimiento, levantada con Docker.
- 🤖 **DeepSeek como LLM:** respuestas de alta calidad con soporte para *streaming* en tiempo real.
- 🧩 **LlamaIndex:** abstracción limpia del pipeline de ingesta y consulta.
- 💬 **Chat interactivo en consola:** bucle infinito con detección de `EXIT` y manejo de `Ctrl+C`.
- 📄 **Ingesta de PDF:** lectura y fragmentación automática del documento fuente.

---

## 🛠️ Tecnologías Utilizadas

| Componente | Tecnología |
|---|---|
| Lenguaje | Python 3.10+ |
| Framework RAG | LlamaIndex |
| Modelo de Embeddings | `BAAI/bge-small-en-v1.5` (384 dims) |
| Servidor de Embeddings | vLLM (OpenAI-compatible API) |
| Base de Datos Vectorial | Milvus (Standalone, Docker) |
| LLM Generativo | DeepSeek (`deepseek-chat`) |
| Lectura de PDF | `pypdf` + `llama-index-readers-file` |

---

## 📂 Estructura del Proyecto
rag_project/
├── ingest.py # Script de ingesta: PDF → Milvus
├── rag.py # Bucle interactivo del chat RAG
├── requirements.txt # Dependencias de Python
├── INSTRUCCIONES.md # Guía paso a paso de instalación
├── sample.txt # Documento de ejemplo
└── README.md


---

## 🚀 Instalación y Configuración

### Requisitos Previos
- Python 3.10 o superior
- Docker Desktop (para Milvus)
- WSL2 (si estás en Windows) o Linux/macOS
- API Key de DeepSeek ([obtener aquí](https://platform.deepseek.com/))

### 1. Clonar el repositorio e instalar dependencias

```bash
git clone https://github.com/<tu-usuario>/<tu-repo>.git
cd rag_project
pip install -r requirements.txt

### Configurar tu API Key de DeepSeek
Abre el archivo rag.py y reemplaza el placeholder con tu API Key personal:
DEEPSEEK_API_KEY = "sk-tu-clave-aqui"

### Iniciar vLLM (servidor de embeddings local)
Abre una terminal nueva y deja el servidor corriendo:
python -m vllm.entrypoints.openai.api_server \
  --model BAAI/bge-small-en-v1.5 \
  --task embed \
  --device cpu \
  --dtype float32 \
  --port 8000

### Iniciar Milvus (base de datos vectorial)
En otra terminal, levanta Milvus Standalone usando Docker:
curl -sfL https://raw.githubusercontent.com/milvus-io/milvus/master/scripts/standalone_embed.sh -o standalone_embed.sh
bash standalone_embed.sh start
>>> Esto expondrá Milvus en localhost:19530

### Ingestar tu PDF
Coloca tu archivo PDF en la carpeta del proyecto y ejecuta:
python ingest.py tu_archivo.pdf
>>> Verás mensajes confirmando la conexión a Milvus, la generación de embeddings y la inserción de los vectores.

### Iniciar el chat RAG
python rag.py
>>> Escribe tus preguntas. Para salir, escribe EXIT o presiona Ctrl+C.

### Ejemplo de Uso
$ python rag.py
Configurando componentes del sistema RAG...
Conectando a Milvus (localhost:19530)...
==================================================
¡Sistema RAG Local Listo!
Escribe tu pregunta sobre el PDF.
Escribe 'EXIT' para salir.
==================================================

Usuario: ¿Cuál es la capital de Francia?
DeepSeek: París es la capital de Francia y su ciudad más poblada, 
conocida como la "Ciudad de la Luz"...
--------------------------------------------------

### Decisiones Arquitectónicas
1. vLLM para embeddings: permite servir el modelo BGE como una API compatible con OpenAI, facilitando la integración con LlamaIndex.
2. Milvus Standalone: elegido por su alto rendimiento y facilidad de despliegue con Docker en un solo comando.
3. DeepSeek como LLM: ofrece una API económica y compatible con el cliente de LlamaIndex, ideal para prototipos.
4. Streaming activado: la respuesta se imprime token a token para mejorar la experiencia de usuario.
5. Colección fija rag_collection: simplifica las pruebas al sobreescribir la colección en cada ingesta.
6. Manejo de errores: si Milvus no está disponible, el script rag.py muestra un mensaje claro y termina de forma controlada.

### Aprendizajes Clave
|Construcción de un pipeline RAG completo desde cero.
|Integración de LlamaIndex con un vector store externo (Milvus).
|Uso de modelos de embeddings locales para preservar la privacidad.
|Consumo de APIs de LLM con streaming en Python.
|Orquestación de servicios con Docker (Milvus) y vLLM.
|Manejo de entornos virtuales y dependencias en Python.

###Licencia
Proyecto de carácter académico. Su uso, copia o distribución con fines comerciales no está permitido sin autorización expresa del autor.
⭐ Si te resultó útil este proyecto, no dudes en darle una estrella al repositorio.
