import sys
from abc import ABC, abstractmethod

import nltk
nltk.download( 'punkt_tab' ) # required by word_tokenize()
from nltk.tokenize import word_tokenize

import numpy as np
import math

from .Preprocessor import Preprocessor, LemmPreprocessor, StemmPreprocessor

from .helpers.decorators import with_time_counter
from .helpers.computators import compute_similarities0, compute_similarities1
from .helpers.Pickle import PickleLoader




class AbstractDocumentFinder( ABC ):

    def __init__( 
        self, 
        preprocessor:Preprocessor, 
        vectorizerLoader:PickleLoader,
        corpusReprLoader:PickleLoader,
        indexLoader:PickleLoader,
        dataset:object
    ):
        self._preprocessor = preprocessor
        self._vectorizer = vectorizerLoader.load()
        self._corpus_repr = corpusReprLoader.load()
        self._index = indexLoader.load()
        self._corpus = dataset.toList()

    @abstractmethod
    def find( self, query:str ):
        pass

    def __str__( self ):
        return self.__class__


class DocumentFinder( AbstractDocumentFinder ):

    def find( self, query:str ):

        @with_time_counter
        def prepare_query( message=None, *args, **kwargs ):
            query_preprocessed = self._preprocessor.transform( [ query ] )
            query_terms = tuple( word_tokenize( query_preprocessed[ 0 ] ) )
            query_repr = self._vectorizer.transform( query_preprocessed )
            return query_terms, query_repr

        query_terms, query_repr = prepare_query( 'Preparing query...' )

        @with_time_counter
        def compute_similarities( message=None, *args, **kwargs ):

            term_weights = {}
            doc_count = {}

            # iterate query terms
            for term in query_terms:

                # if term is included in index
                if term in self._index:

                    # compute a term weight like idf (+1 to avoid zero division)
                    term_weights[ term ] = math.log( len( self._corpus ) / ( 1 + len( self._index[ term ].keys() ) ) )

                    # iterate all docs the term is ocuured within
                    for key in self._index[ term ].keys():
                        # doc_count[ key ] = doc_count.get( key, 0 ) + 1
                        doc_count[ key ] = doc_count.get( key, 0 ) + term_weights[ term ]

            # required_count = len( query_terms ) // 2
            # filtered_keys = [ key for key, count in doc_count.items() if count >= required_count ]
            # filtered_corpus_repr = np.array( [ self._corpus_repr[ key ] for key in filtered_keys ] )
            # filtered_corpus_repr.reshape( len( filtered_keys ), -1 )

            # in case of no results
            if len( doc_count.items() ) == 0:
                return None

            # sort the documents' weights and get the top 100 to check similarities
            doc_weights = [ ( key, weight ) for key, weight in doc_count.items() ]
            doc_weights.sort( key=lambda x: x[1], reverse=True )
            doc_weights = doc_weights[:100] if len( doc_weights ) > 100 else doc_weights
            filtered_corpus_repr = np.array( [ self._corpus_repr[ key ] for key, weight in doc_weights ] )
            filtered_corpus_repr.reshape( len( doc_weights ), -1 )

            similarities = compute_similarities0( query_repr, filtered_corpus_repr )

            results = []
            # for key, similarity in zip( filtered_keys, similarities ):
            for ( key, weight ), similarity in zip( doc_weights, similarities ):
                results.append( (
                    self._corpus[ key ][ 'id' ],
                    self._corpus[ key ][ 'catg_ids' ],
                    similarity
                ) )
            
            results.sort( key=lambda x: x[2], reverse=True )
            return results

        return compute_similarities( 'Computing similarities...' )

# RUN: python -m src.DocumentFinder [parameters] [query]
if __name__ == "__main__": 

    from .arXiv.Dataset import Dataset
    from .arXiv.settings import pickle_paths

    option = None
    if len( sys.argv ) >= 2:
        option = sys.argv[ 1 ]

    query = "Available literature about databases (both SQL and NoSQL), especially somehow relevant to semantics?"
    if len( sys.argv ) >= 3:
        query = sys.argv[ 2 ]

    results = None

    match option:

        case 'stemm-single-count':

            vectorizer_descr = 'title-summary_lower-punct-specials-stops-stemm_single_count'
            vectorizer_filename = f"{pickle_paths[ 'vectorizers' ]}/{vectorizer_descr}.pkl"
            corpus_repr_filename = f"{pickle_paths[ 'corpus_repr' ]}/{vectorizer_descr}.pkl"

            index_descr = 'title-summary_lower-punct-specials-stops-stemm_single'
            index_filename = f"{pickle_paths[ 'indexes' ]}/{index_descr}.pkl"

            documentFinder = DocumentFinder(
                StemmPreprocessor(), 
                PickleLoader( vectorizer_filename ),
                PickleLoader( corpus_repr_filename ),
                PickleLoader( index_filename ),
                Dataset()
            )

            results = documentFinder.find( query )

        case 'lemm-single-tfidf':

            vectorizer_descr = 'title-summary_lower-punct-specials-stops-lemm_single_tfidf'
            vectorizer_filename = f"{pickle_paths[ 'vectorizers' ]}/{vectorizer_descr}.pkl"
            corpus_repr_filename = f"{pickle_paths[ 'corpus_repr' ]}/{vectorizer_descr}.pkl"

            index_descr = 'title-summary_lower-punct-specials-stops-lemm_single'
            index_filename = f"{pickle_paths[ 'indexes' ]}/{index_descr}.pkl"

            documentFinder = DocumentFinder(
                LemmPreprocessor(), 
                PickleLoader( vectorizer_filename ),
                PickleLoader( corpus_repr_filename ),
                PickleLoader( index_filename ),
                Dataset()
            )

            results = documentFinder.find( query )

        case _:
            raise Exception( 'No valid parameters passed.' )


    if not results:
        print( 'No results found' )

    else:
        limit = 20 if len( results ) > 20 else len( results )
        print()
        for i in range( limit ):
            print( results[ i ] )
