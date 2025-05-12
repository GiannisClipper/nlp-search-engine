from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

print( 'Load BERT model...' )
model = SentenceTransformer( 'all-MiniLM-L6-v2' )

corpus = [
    "The Eiffel Tower is in Paris.",
    "The Great Wall of China is visible from space.",
    "Python is a popular programming language.",
    "Machine learning enables computers to learn from data."
]

print( 'Encode corpus...' )
corpus_embeddings = model.encode( corpus, convert_to_numpy=True )

print( 'Make FAISS index...' )
embedding_dim = corpus_embeddings.shape[1]
# index = faiss.IndexFlatL2(embedding_dim)  # L2 = Euclidean distance
index = faiss.IndexFlatIP( embedding_dim )  # Inner product for cosine similarity

# faiss.normalize_L2( corpus_embeddings )
index.add( corpus_embeddings ) # type: ignore

print( 'Encode query...' )
query = "Where is the Eiffel Tower located?"
query_embedding = model.encode( query, convert_to_numpy=True )
query_embedding = np.array( [ query_embedding ] )
# faiss.normalize_L2( query_embedding )

print( 'Search query...' )
k = index.ntotal
print( 'query_embedding.shape:', query_embedding.shape )
distances, indices = index.search( query_embedding, k=k ) # type: ignore
print( 'indices:', indices )
print(f"\nQuery: {query}\nTop {k} results:")
for i in range(k):
    print(f"{corpus[indices[0][i]]} (Distance: {distances[0][i]:.4f})")
