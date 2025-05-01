##################################################################################
# DEPRICATED: it's functionality is splitted into QueryAnalyzer and DocEstimator #
##################################################################################

import sys
from abc import ABC, abstractmethod

import nltk
nltk.download( 'punkt_tab' ) # required by word_tokenize()
from nltk.tokenize import word_tokenize

import numpy as np

from .Preprocessor import Preprocessor, LemmPreprocessor, StemmPreprocessor
from .TermsFilter import AbstractTermsFilter, TermsFilter
from .helpers.decorators import with_time_counter
from .helpers.computators import compute_similarities0, compute_similarities1
from .helpers.Pickle import PickleLoader


class AbstractDocFinder( ABC ):

    def __init__( 
        self,
        preprocessor:Preprocessor, 
        vectorizerLoader:PickleLoader,
        corpusReprLoader:PickleLoader,
        corpus:list[dict],
        termsFilter:AbstractTermsFilter
    ):
        self._preprocessor = preprocessor
        self._vectorizer = vectorizerLoader.load()
        self._corpus_repr = corpusReprLoader.load()
        self._corpus = corpus
        self._termsFilter = termsFilter

    @abstractmethod
    def find( self, query:str ):
        pass

    def __str__( self ):
        return self.__class__


class DocFinder( AbstractDocFinder ):

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

            # select through index some documents to compare
            doc_selection = self._termsFilter.filter( query_terms )
            if len( doc_selection ) == 0:
                return []

            # get the corresponding corpus representations
            filtered_corpus_repr = np.array( [ self._corpus_repr[ key ] for key in doc_selection ] )
            filtered_corpus_repr.reshape( len( doc_selection ), -1 )

            # compute the similarities
            similarities = compute_similarities0( query_repr, filtered_corpus_repr )

            # put together document ids and similarities
            results = []
            for key, similarity in zip( doc_selection, similarities ):
                results.append( ( self._corpus[ key ], round( float( similarity ), 4 ) ) ) # type: ignore

            # greater similarities at the top
            results.sort( key=lambda x: x[1], reverse=True )

            return results

        return compute_similarities( 'Computing similarities...' )


def find_and_show( 
    pickle_paths:dict,
    vectorizer_descr:str,
    index_descr:str,
    PreprocessorClass, 
    corpus:list[dict], 
    TermsFilterClass,
):
    vectorizer_filename = f"{pickle_paths[ 'vectorizers' ]}/{vectorizer_descr}.pkl"
    corpus_repr_filename = f"{pickle_paths[ 'corpus_repr' ]}/{vectorizer_descr}.pkl"
    index_filename = f"{pickle_paths[ 'indexes' ]}/{index_descr}.pkl"
    index = PickleLoader( index_filename ).load()

    docFinder = DocFinder(
        preprocessor=PreprocessorClass(), 
        vectorizerLoader=PickleLoader( vectorizer_filename ),
        corpusReprLoader=PickleLoader( corpus_repr_filename ),
        corpus=corpus,
        termsFilter=TermsFilterClass( corpus, index )
    )

    results = docFinder.find( query )

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


# RUN: python -m src.DocFinder [option]
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
                corpus=Dataset().toList(),
                TermsFilterClass=TermsFilter
            )

        case 'arxiv-lemm-single-tfidf':
            from .arXiv.Dataset import Dataset
            from .arXiv.settings import pickle_paths
            find_and_show(
                pickle_paths,
                vectorizer_descr = 'title-summary_lower-punct-specials-stops-lemm_single_tfidf',
                index_descr = 'title-summary_lower-punct-specials-stops-lemm_single',
                PreprocessorClass=LemmPreprocessor,
                corpus=Dataset().toList(),
                TermsFilterClass=TermsFilter
            )

        case 'medical-lemm-single-tfidf':
            from .medical.Dataset import Dataset
            from .medical.settings import pickle_paths
            find_and_show(
                pickle_paths,
                vectorizer_descr = 'title-summary_lower-punct-specials-stops-lemm_single_tfidf',
                index_descr = 'title-summary_lower-punct-specials-stops-lemm_single',
                PreprocessorClass=LemmPreprocessor,
                corpus=Dataset().toList(),
                TermsFilterClass=TermsFilter
            )

        case _:
            raise Exception( 'No valid option.' )

