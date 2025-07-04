from abc import ABC, abstractmethod
import math

from sklearn.cluster import KMeans
import bm25s
import faiss
from scipy import sparse
from .helpers.typing import QueryAnalyzedType
from .Preprocessor import NaivePreprocessor

# -------------------------------------------------------------------------------- #
# Code to filter docs or sentences regarding the terms or the embedding of a query #
# -------------------------------------------------------------------------------- #

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
        
        tokens = query_analyzed[ 'tokens' ]

        # keep only tokens existing in index
        tokens = [ t for t in tokens if t in self._index ]

        # extract the unique terms
        terms = list( set( tokens ) )

        doc_stats = {}
        for term in terms:

            # iterate all docs the term is ocuured within
            for idoc in self._index[ term ].keys():
                if idoc not in doc_stats:
                    doc_stats[ idoc ] = 0
                doc_stats[ idoc ] += 1

        # select docs with including the half of the single terms at least
        # excluding 2-grams to avoid low rate of filtering due to redundancy of terms  
        single_terms = [ t for t in terms if len( t.split() ) == 1 ]
        minTerms = len( single_terms ) * threshold if threshold > 0.0 else 1
        result = [ ( idoc, stat ) for idoc, stat in doc_stats.items() if stat >= minTerms ]
        result = [ i for i, _ in result ]
        return result


class WeightedTermsFilter( AbstractIndexedTermsFilter ):

    def __init__( self, index:dict, corpus:list[dict], limit:int=200 ):
        super().__init__( index )
        self._corpus = corpus
        self._limit = limit

    def filter( self, query_analyzed:QueryAnalyzedType ) -> list[str]:

        tokens = query_analyzed[ 'tokens' ]

        # keep only tokens existing in index
        tokens = [ t for t in tokens if t in self._index ] 

        # extract the unique terms
        terms = list( set( tokens ) )

        term_weights = {}
        doc_stats = {}
        for term in terms:

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


class BM25TermsFilter( AbstractTermsFilter ):

    def __init__( self, corpus:list[str] ):
        super().__init__()
        self._model = bm25s.BM25() # class BM25: https://github.com/xhluca/bm25s/blob/main/bm25s/__init__.py
        preprocessor = NaivePreprocessor()
        corpus = preprocessor.transform( corpus )
        self._model.index( bm25s.tokenize( corpus ) )

    def filter( self, query_analyzed:QueryAnalyzedType ) -> list[str]:
        # query = ' '.join( query_analyzed[ 'tokens' ] )
        # isents, scores = self._model.retrieve( bm25s.tokenize( query ), k=100 )
        isents, scores = self._model.retrieve( [ query_analyzed[ 'tokens' ] ], k=200 )
        isents = [ str(isent) for isent in isents[0] ]
        return isents


class FaissTermsFilter( AbstractTermsFilter ):

    def __init__( self, sentences_embeddings ):
        super().__init__()
        embedding_dim = sentences_embeddings.shape[ 1 ]
        faiss.normalize_L2( sentences_embeddings ) # normalize in place
        self._index = faiss.IndexFlatIP( embedding_dim )  # Inner product for cosine similarity
        # self._index = faiss.IndexFlatL2( embedding_dim )  # L2 = Euclidean distance
        self._index.add( sentences_embeddings ) # type: ignore
        # print( 'DEBUG-FaissTermsFilter-sentences.shape', corpus_embeddings.shape )

    def filter( self, query_analyzed:QueryAnalyzedType ) -> list[str]:
        embedding = sparse.lil_matrix( query_analyzed[ 'repr' ] ).toarray()
        faiss.normalize_L2( embedding ) # normalize in place
        # print( 'DEBUG-FaissTermsFilter-query.shape:', embedding.shape )
        distances, indices = self._index.search( embedding, k=200 ) # type: ignore
        isents = [ str(isent) for isent in indices[0] ]
        # print( 'DEBUG-FaissTermsFilter-isents:', isents )
        return isents
