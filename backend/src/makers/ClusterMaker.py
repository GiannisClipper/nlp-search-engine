import sys
from abc import ABC, abstractmethod
from typing import cast
from sklearn.cluster import KMeans, AgglomerativeClustering
import numpy as np
from ..helpers.Pickle import PickleSaver, PickleLoader

# +-----------------------------------------------+
# | Code to carry out represetnations' clustering |
# +-----------------------------------------------+

class AbstractClustersMaker( ABC ):

    def __init__( self, model:KMeans|AgglomerativeClustering, data:list|np.ndarray, filename:str ):
        self._model = model
        self._data = data
        self._filename = filename

    @property
    def model( self ):
        return self._model

    @abstractmethod
    def make( self ):
        pass

    def __str__( self ):
        return self.__class__    

class KMeansClustersMaker( AbstractClustersMaker ):

    def __init__( self, data:np.ndarray, filename:str ):
        K = data.shape[ 0 ] // 200
        model = KMeans( n_clusters=K, random_state=32 )
        super().__init__( model, data, filename )

    def make( self ):
        print( 'Perform clustering...' )
        self._model = cast( KMeans, self._model )
        self._model.fit( self._data )
        print( 'Centroids shape:', self._model.cluster_centers_.shape )
        print( 'Labels shape:', self._model.labels_.shape )

        print( 'Saving model...' )
        PickleSaver( self._filename ).save( self._model )


def clustersMakerFactory( option:str ):

    match option:

        case 'arxiv-sentences-jina-kmeans':
            from ..datasets.arXiv.settings import pickle_paths
            representations_descr = 'sentences-jina'
            representations_filename = f"{pickle_paths[ 'corpus_repr' ]}/{representations_descr}.pkl"
            representations = PickleLoader( representations_filename ).load()
            clustering_descr = 'sentences-jina-kmeans'
            clustering_filename = f"{pickle_paths[ 'clusters' ]}/{clustering_descr}.pkl"
            return KMeansClustersMaker( representations, clustering_filename )

        case 'medical-sentences-jina-kmeans':
            from ..datasets.medical.settings import pickle_paths
            representations_descr = 'sentences-jina'
            representations_filename = f"{pickle_paths[ 'corpus_repr' ]}/{representations_descr}.pkl"
            representations = PickleLoader( representations_filename ).load()
            clustering_descr = 'sentences-jina-kmeans'
            clustering_filename = f"{pickle_paths[ 'clusters' ]}/{clustering_descr}.pkl"
            return KMeansClustersMaker( representations, clustering_filename )

        case 'medical-sentences-bert-kmeans':
            from ..datasets.medical.settings import pickle_paths
            representations_descr = 'sentences-bert'
            representations_filename = f"{pickle_paths[ 'corpus_repr' ]}/{representations_descr}.pkl"
            representations = PickleLoader( representations_filename ).load()
            clustering_descr = 'sentences-bert-kmeans'
            clustering_filename = f"{pickle_paths[ 'clusters' ]}/{clustering_descr}.pkl"
            return KMeansClustersMaker( representations, clustering_filename )

        case _:
            raise Exception( 'clustersMakerFactory(): No valid option.' )


# RUN: python -m src.ClusterMaker [option]
if __name__ == "__main__": 

    option = None
    if len( sys.argv ) >= 2:
        option = sys.argv[ 1 ]

    match option:

        case 'arxiv-sentences-jina-kmeans':
            maker = clustersMakerFactory( option )
            maker.make()

        case 'medical-sentences-jina-kmeans':
            maker = clustersMakerFactory( option )
            maker.make()

        case 'medical-sentences-bert-kmeans':
            maker = clustersMakerFactory( option )
            maker.make()

        case _:
            raise Exception( 'No valid option.' )
