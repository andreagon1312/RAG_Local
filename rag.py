import os
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.deepseek import DeepSeek
from llama_index.core import Settings
import logging
import sys

# Reducir los logs de httpx que pueden ser molestos
logging.getLogger("httpx").setLevel(logging.WARNING)

# API Key de DeepSeek
DEEPSEEK_API_KEY = "sk-tu-clave-aqui"

def main():
    print("Configurando componentes del sistema RAG...")

    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

    # 2. Configurar el LLM (DeepSeek)
    llm = DeepSeek(
        model="deepseek-chat",
        api_key=DEEPSEEK_API_KEY
    )

    # Configurar los settings globales de LlamaIndex
    Settings.embed_model = embed_model
    Settings.llm = llm

    # 3. Conectar a Milvus y cargar el índice
    print("Conectando a Milvus (localhost:19530)...")
    try:
        vector_store = MilvusVectorStore(
            uri="http://localhost:19530",
            collection_name="rag_collection",
            dim=384
        )
        index = VectorStoreIndex.from_vector_store(vector_store)
    except Exception as e:
        print(f"Error conectando a Milvus o cargando el índice: {e}")
        print("¿Asegúrate de haber ejecutado ingest.py primero y de que Milvus esté corriendo?")
        sys.exit(1)

    # 4. Crear el motor de búsqueda (query engine) con streaming activado
    query_engine = index.as_query_engine(streaming=True)

    print("\n" + "="*50)
    print("¡Sistema RAG Local Listo!")
    print("Escribe tu pregunta sobre el PDF.")
    print("Escribe 'EXIT' para salir.")
    print("="*50 + "\n")

    # 5. Bucle interactivo
    while True:
        try:
            query = input("\nUsuario: ")
            
            # Condición de salida
            if query.strip().upper() == "EXIT":
                print("Saliendo del sistema RAG. ¡Hasta luego!")
                break
            
            if not query.strip():
                continue

            print("\nDeepSeek: ", end="", flush=True)
            
            # Ejecutar la consulta RAG
            response = query_engine.query(query)
            
            # Imprimir la respuesta en streaming
            for text in response.response_gen:
                print(text, end="", flush=True)
            print("\n" + "-"*50)

        except KeyboardInterrupt:
            print("\nSaliendo del sistema RAG...")
            break
        except Exception as e:
            print(f"\nError durante la consulta: {e}")
            print("-" * 50)

if __name__ == "__main__":
    main()
