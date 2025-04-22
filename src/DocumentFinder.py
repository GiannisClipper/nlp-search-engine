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
        corpus:list[dict],
    ):
        self._preprocessor = preprocessor
        self._vectorizer = vectorizerLoader.load()
        self._corpus_repr = corpusReprLoader.load()
        self._index = indexLoader.load()
        self._corpus = corpus

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
                results.append( ( self._corpus[ key ], round( float( similarity ), 4 ) ) )
                    
            results.sort( key=lambda x: x[1], reverse=True )
            return results

        return compute_similarities( 'Computing similarities...' )


def find_and_show( 
    pickle_paths:dict,
    vectorizer_descr:str,
    index_descr:str,
    PreprocessorClass, 
    corpus:list[dict], 
):
    vectorizer_filename = f"{pickle_paths[ 'vectorizers' ]}/{vectorizer_descr}.pkl"
    corpus_repr_filename = f"{pickle_paths[ 'corpus_repr' ]}/{vectorizer_descr}.pkl"
    index_filename = f"{pickle_paths[ 'indexes' ]}/{index_descr}.pkl"

    documentFinder = DocumentFinder(
        preprocessor=StemmPreprocessor(), 
        vectorizerLoader=PickleLoader( vectorizer_filename ),
        corpusReprLoader=PickleLoader( corpus_repr_filename ),
        indexLoader=PickleLoader( index_filename ),
        corpus=corpus
    )

    results = documentFinder.find( query )

    if not results:
        print( 'No results found' )

    else:
        cleaned_results = []
        for result in results:
            if 'catg_ids' in result[ 0 ]:
                cleaned_results.append( ( result[0][ 'id' ], result[0][ 'catg_ids' ], result[1] ) )
            else:
                cleaned_results.append( ( result[0][ 'id' ], result[0][ 'url' ], result[1] ) )

        limit = 20 if len( cleaned_results ) > 20 else len( cleaned_results )
        print()
        for i in range( limit ):
            print( cleaned_results[ i ] )


# RUN: python -m src.DocumentFinder [option]
if __name__ == "__main__": 

    from .arXiv.Dataset import Dataset
    from .arXiv.settings import pickle_paths

    option = None
    if len( sys.argv ) >= 2:
        option = sys.argv[ 1 ]

    query = "Is there any available literature about databases (both SQL and NoSQL), especially somehow relevant to semantics?"
    if len( sys.argv ) >= 3:
        query = sys.argv[ 2 ]

    results = None

    match option:

        case 'arxiv-stemm-single-count':
            from .arXiv.Dataset import Dataset
            from .arXiv.settings import pickle_paths
            find_and_show(
                pickle_paths,
                vectorizer_descr = 'title-summary_lower-punct-specials-stops-stemm_single_count',
                index_descr = 'title-summary_lower-punct-specials-stops-stemm_single',
                PreprocessorClass=StemmPreprocessor,
                corpus=Dataset().toList()
            )

        case 'arxiv-lemm-single-tfidf':
            from .arXiv.Dataset import Dataset
            from .arXiv.settings import pickle_paths
            find_and_show(
                pickle_paths,
                vectorizer_descr = 'title-summary_lower-punct-specials-stops-lemm_single_tfidf',
                index_descr = 'title-summary_lower-punct-specials-stops-lemm_single',
                PreprocessorClass=LemmPreprocessor,
                corpus=Dataset().toList()
            )

        case 'medical-lemm-single-tfidf':
            from .medical.Dataset import Dataset
            from .medical.settings import pickle_paths
            find_and_show(
                pickle_paths,
                vectorizer_descr = 'title-summary_lower-punct-specials-stops-lemm_single_tfidf',
                index_descr = 'title-summary_lower-punct-specials-stops-lemm_single',
                PreprocessorClass=LemmPreprocessor,
                corpus=Dataset().toList()
            )

        case _:
            raise Exception( 'No valid option.' )

