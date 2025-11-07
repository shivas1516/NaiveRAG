# Retrieval pipeline to get top 3 matching chunks from the vector store

def retrieve_from_vector_store(vector_store, query: str, top_k: int = 3):

    # semantic similarity search
    results = vector_store.similarity_search(query, k=top_k) # cosine similarity by default

    return results