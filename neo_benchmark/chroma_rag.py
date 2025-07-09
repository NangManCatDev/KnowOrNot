import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

class ChromaRAG:
    def __init__(self, persist_dir="./chroma_db", embedding_model="jhgan/ko-sroberta-multitask"):
        self.chroma_client = chromadb.Client(Settings(persist_directory=persist_dir))
        self.collection = self.chroma_client.get_or_create_collection("kb_docs")
        self.model = SentenceTransformer(embedding_model)

    def build_kb(self, kb_texts):
        embeddings = self.model.encode(kb_texts, convert_to_numpy=True)
        for i, (text, emb) in enumerate(zip(kb_texts, embeddings)):
            self.collection.add(
                documents=[text],
                embeddings=[emb.tolist()],
                ids=[f"kb_{i}"]
            )

    def query(self, query, top_k=3):
        query_emb = self.model.encode([query], convert_to_numpy=True)[0]
        results = self.collection.query(
            query_embeddings=[query_emb.tolist()],
            n_results=top_k
        )
        return results['documents'][0] if results['documents'] else [] 