import os
import argparse
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

def main():
    parser = argparse.ArgumentParser(description="Ingesta un PDF a Milvus usando vLLM local.")
    parser.add_argument("pdf_path", type=str, help="Ruta al archivo PDF a ingestar")
    args = parser.parse_args()

    if not os.path.exists(args.pdf_path):
        print(f"Error: El archivo '{args.pdf_path}' no existe.")
        return

    print(f"Cargando el documento PDF: {args.pdf_path}")
    # SimpleDirectoryReader puede tomar un solo archivo si usamos input_files
    documents = SimpleDirectoryReader(input_files=[args.pdf_path]).load_data()
    print(f"Se cargaron {len(documents)} páginas/chunks del documento.")

    print("Configurando el modelo de embedding local (vLLM)...")
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

    print("Conectando a Milvus en localhost:19530...")
    # Configurar la conexión a Milvus
    vector_store = MilvusVectorStore(
        uri="http://localhost:19530", 
        collection_name="rag_collection",
        dim=384, # La dimensión de bge-small-en-v1.5 es 384
        overwrite=True # Sobreescribir si la colección ya existe para pruebas fáciles
    )
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    print("Generando embeddings e insertando en Milvus... (esto puede tomar un momento)")
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        embed_model=embed_model,
        show_progress=True
    )
    
    print("¡Ingesta completada con éxito! Los datos están indexados en Milvus.")

if __name__ == "__main__":
    main()
