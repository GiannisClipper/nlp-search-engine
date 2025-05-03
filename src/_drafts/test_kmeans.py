# to run: $ python3 -m src.tests.test_kmeans

from ..datasets.arXiv.Dataset import Dataset
from ..datasets.arXiv.settings import pickle_paths
from ..helpers.Pickle import PickleLoader

representations_descr = 'jina-embeddings-sentences'
representations_filename = f"{pickle_paths[ 'corpus_repr' ]}/{representations_descr}.pkl"
embeddings = PickleLoader( representations_filename ).load()

print( embeddings )
print( embeddings.shape )
# print( embeddings[:1000].shape )

from sklearn.cluster import KMeans
import numpy as np

data = embeddings
K = data.shape[ 0 ] // 300
members = np.zeros( K )

kmeans = KMeans( n_clusters=K )
kmeans.fit( data )
print( kmeans.cluster_centers_.shape )

for l in kmeans.labels_:
    members[ l ] += 1

members.sort()
print( members )

print( kmeans.labels_[0:10] )
res = kmeans.predict( data[0:10].reshape( 10, -1 ) )
print( res )