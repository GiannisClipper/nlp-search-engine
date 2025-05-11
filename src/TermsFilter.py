from abc import ABC, abstractmethod
import math

from sklearn.cluster import KMeans 
import bm25s
import faiss
from scipy import sparse
from .helpers.typing import QueryAnalyzedType
from .Preprocessor import NaivePreprocessor


class AbstractTermsFilter( ABC ):

    @abstractmethod
    def filter( self, query_analyzed:QueryAnalyzedType ) -> list[str]:
        pass

    def __str__( self ):
        return self.__class__


class AbstractIndexedTermsFilter( AbstractTermsFilter ):

    def __init__( self, index:dict ):
        super().__init__()
        self._index = index


class OccuredTermsFilter( AbstractIndexedTermsFilter ):

    def __init__( self, index:dict, threshold:float=0.5 ):
        super().__init__( index )
        self._threshold = threshold

    def filter( self, query_analyzed:QueryAnalyzedType, threshold:float=0.5 ) -> list[str]:
        
        terms = query_analyzed[ 'terms' ]

        # take 2grams into account
        twograms = []
        for i in range( len( terms ) -1 ):
            twograms.append( terms[ i ] + ' ' + terms[ i + 1 ] )
        terms += twograms

        # keep only terms existing in index
        terms = [ t for t in terms if t in self._index ] 

        doc_stats = {}
        for term in terms:

            # if term is included in index
            # if term in self._index:

                # iterate all docs the term is ocuured within
                for idoc in self._index[ term ].keys():
                    if idoc not in doc_stats:
                        doc_stats[ idoc ] = 0
                    doc_stats[ idoc ] += 1

        # select docs with including the half terms at least
        minTerms = len( terms ) * threshold if threshold > 0.0 else 1
        result = [ ( idoc, stat ) for idoc, stat in doc_stats.items() if stat >= minTerms ]
        result = [ i for i, _ in result ]
        return result


class WeightedTermsFilter( AbstractIndexedTermsFilter ):

    def __init__( self, index:dict, corpus:list[dict], limit:int=200 ):
        super().__init__( index )
        self._corpus = corpus
        self._limit = limit

    def filter( self, query_analyzed:QueryAnalyzedType ) -> list[str]:

        terms = query_analyzed[ 'terms' ]

        # take 2grams into account
        twograms = []
        for i in range( len( terms ) -1 ):
            twograms.append( terms[ i ] + ' ' + terms[ i + 1 ] )
        terms += twograms

        # keep only terms existing in index
        terms = [ t for t in terms if t in self._index ] 

        term_weights = {}
        doc_stats = {}
        for term in terms:

            # if term is included in index
            # if term in self._index:

                # compute a term weight like idf (+1 to avoid zero division)
                term_weights[ term ] = math.log10( len( self._corpus ) / ( 1 + len( self._index[ term ].keys() ) ) )

                # iterate all docs the term is ocurred within
                for idoc in self._index[ term ].keys():
                    if idoc not in doc_stats:
                        doc_stats[ idoc ] = 0
                    doc_stats[ idoc ] += term_weights[ term ]

        # sort the documents' weights and get the top to check similarities
        result = [ ( idoc, weight ) for idoc, weight in doc_stats.items() ]
        result.sort( key=lambda x: x[1], reverse=True )
        result = result[:self._limit] if self._limit > 0 and len( result ) > self._limit else result
        result = [ i for i, _ in result ]
        return result


class ClusteredTermsFilter( AbstractTermsFilter ):

    def __init__( self, model:KMeans ):
        super().__init__()
        self._model = model

    def filter( self, query_analyzed:QueryAnalyzedType ) -> list[str]:
        repr = query_analyzed[ 'repr' ]
        label = self._model.predict( repr )[ 0 ]
        result = [ str(isent) for isent, l in enumerate( self._model.labels_ ) if l == label ]
        return result

class B25TermsFilter( AbstractTermsFilter ):

    def __init__( self, corpus:list[str] ):
        super().__init__()
        self._model = bm25s.BM25() # class BM25: https://github.com/xhluca/bm25s/blob/main/bm25s/__init__.py
        preprocessor = NaivePreprocessor()
        corpus = preprocessor.transform( corpus )
        self._model.index( bm25s.tokenize( corpus ) )

    def filter( self, query_analyzed:QueryAnalyzedType ) -> list[str]:
        # query = ' '.join( query_analyzed[ 'terms' ] )
        # isents, scores = self._model.retrieve( bm25s.tokenize( query ), k=100 )
        isents, scores = self._model.retrieve( [ query_analyzed[ 'terms' ] ], k=100 )
        isents = [ str(isent) for isent in isents[0] ]
        return isents

class FaissTermsFilter( AbstractTermsFilter ):

    def __init__( self, corpus_embeddings ):
        super().__init__()
        embedding_dim = corpus_embeddings.shape[ 1 ]
        self._index = faiss.IndexFlatIP( embedding_dim )  # Inner product for cosine similarity
        # self._index = faiss.IndexFlatL2( embedding_dim )  # L2 = Euclidean distance
        # faiss.normalize_L2( corpus_embeddings )
        self._index.add( corpus_embeddings ) # type: ignore

    def filter( self, query_analyzed:QueryAnalyzedType ) -> list[str]:
        embedding = sparse.lil_matrix( query_analyzed[ 'repr' ] ).toarray()
        # faiss.normalize_L2( embedding )
        print( 'query_embedding.shape:', embedding.shape )
        distances, indices = self._index.search( embedding, k=100 ) # type: ignore
        isents = [ str(isent) for isent in indices[0] ]
        return isents
