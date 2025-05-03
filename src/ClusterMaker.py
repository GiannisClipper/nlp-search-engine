import sys
from abc import ABC, abstractmethod
from sklearn.cluster import KMeans
import numpy as np
from .helpers.Pickle import PickleSaver, PickleLoader

class AbstractClustersMaker( ABC ):

    def __init__( self, model:KMeans, data:list|np.ndarray, filename:str ):
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
        K = data.shape[ 0 ] // 400
        model = KMeans( n_clusters=K, random_state=32 )
        super().__init__( model, data, filename )

    def make( self ):
        print( 'Perform clustering...' )
        self._model.fit( self._data )
        print( 'Centroids shape:', self._model.cluster_centers_.shape )
        print( 'Labels shape:', self._model.labels_.shape )

        print( 'Saving model...' )
        PickleSaver( self._filename ).save( self._model )


def clustersMakerFactory( option:str ):

    match option:

        case 'arxiv-sentences-jina-kmeans':
            from .arXiv.settings import pickle_paths
            representations_descr = 'jina-embeddings-sentences'
            representations_filename = f"{pickle_paths[ 'corpus_repr' ]}/{representations_descr}.pkl"
            representations = PickleLoader( representations_filename ).load()
            clustering_descr = 'sentences-jina-kmeans'
            clustering_filename = f"{pickle_paths[ 'clusters' ]}/{clustering_descr}.pkl"
            return KMeansClustersMaker( representations, clustering_filename )

        case 'medical-sentences-jina-kmeans':
            from .medical.settings import pickle_paths
            representations_descr = 'jina-embeddings-sentences'
            representations_filename = f"{pickle_paths[ 'corpus_repr' ]}/{representations_descr}.pkl"
            representations = PickleLoader( representations_filename ).load()
            clustering_descr = 'sentences-jina-kmeans'
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

        case _:
            raise Exception( 'No valid option.' )
