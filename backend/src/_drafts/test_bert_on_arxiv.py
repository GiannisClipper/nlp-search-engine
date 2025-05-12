from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

print( 'Load BERT model...' )
model = SentenceTransformer( 'all-MiniLM-L6-v2' )

from ..datasets.arXiv.settings import pickle_paths
from ..helpers.Pickle import PickleLoader
descr = 'sentences-bert'
filename = f"{pickle_paths[ 'corpus_repr' ]}/{descr}.pkl"
corpus_embeddings = PickleLoader( filename ).load()

# print( 'Encode corpus...' )
# corpus_embeddings = model.encode( corpus, convert_to_numpy=True )

print( 'Make FAISS index...' )
embedding_dim = corpus_embeddings.shape[1]
index = faiss.IndexFlatIP( embedding_dim )  # Inner product for cosine similarity
index.add( corpus_embeddings ) # type: ignore

print( 'Encode query...' )
# query = "Are there any available papers about database management systems (both SQL and NoSQL), especially somehow relevant to semantics?"
query = "I am interested on databases (both SQL and NoSQL)..."
query_embedding = model.encode( query, convert_to_numpy=True )
query_embedding = np.array( [ query_embedding ] )

print( 'Search query...' )
k = 100
print( 'query_embedding.shape:', query_embedding.shape )
distances, indices = index.search( query_embedding, k=k ) # type: ignore
print( 'indices:', indices )

from ..datasets.arXiv.Dataset import Dataset
ds = Dataset()
corpus = ds.toList()
sentences, tags = ds.toSentences()

print(f"\nQuery: {query}\nTop {k} results:")
for i in range(k):
    # print(f"{corpus[indices[0][i]]} (Distance: {distances[0][i]:.4f})")
    isent = int( indices[0][i] )
    idoc = int( tags[ isent ].split('.')[0] )
    doc = corpus[ idoc ]
    print( f"{doc['id']} {doc['catg_ids']} {distances[0][i]}" )

# Query: I am interested on databases (both SQL and NoSQL)...
# indices: [[ 9278  7248  9517 10427 12836  9121  7476  8296  7475  9988  9560 10462
#    9119  9123  9055 11053 12307  7421  9120  8380 32349 10908 10239 10724
#   10422 11407  9201 10903 11414 11286  9124 10424 11526 11493 12086  9537
#    7410  9518  9816  8758 10426  9005 10423  9548 12442 12496  9679  7614
#    8226  7813 11415  8655 12087  7594 35940 14325  8003 10532  9844 10856
#    9221  9223  8843 11288  8295  9991  7473  7613 10420  9516 11287 10289
#    8884 10428  7857 10759  8033 10188 39856  7474 10434  7529  9990  8549
#   12834  7808 10375 10760  9818 12309  9519 10076 34220  9222 12315 12476
#    9080  9081 10747  8225]]
