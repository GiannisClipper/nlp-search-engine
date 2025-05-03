import sys
from abc import ABC, abstractmethod

import nltk
nltk.download( 'punkt_tab' ) # required by word_tokenize()
from nltk.tokenize import word_tokenize

import pytrie

from ..Preprocessor import Preprocessor, LemmPreprocessor, StemmPreprocessor

from ..helpers.decorators import with_time_counter
from ..helpers.Pickle import PickleLoader, PickleSaver


class AbstractIndexMaker( ABC ):

    def __init__( 
        self, 
        corpus:list[str], 
        preprocessor:Preprocessor, 
        vocabularyLoader:PickleLoader
    ):
        self._corpus = corpus
        self._preprocessor = preprocessor
        self._vocabularyLoader = vocabularyLoader

    @abstractmethod
    def make( self ):
        pass

    def __str__( self ):
        return self.__class__


class IndexMaker( AbstractIndexMaker ):

    def make( self ):
        print( f'\nPreprocessing...' )
        corpus = self._preprocessor.transform( self._corpus )
        vocabulary = self._vocabularyLoader.load()

        @with_time_counter
        def create_index( message=None, *args, **kwargs ):

            corpus = kwargs[ 'corpus' ]
            vocabulary = kwargs[ 'vocabulary' ]

            # split documents in tokens
            for i in range( len( corpus ) ):
                corpus[ i ] = tuple( word_tokenize( corpus[ i ] ) )

            # initialize index with all terms  
            index = pytrie.StringTrie()
            for term in vocabulary:
                index[ term ] = {}

            # iterate the documents of the corpus
            for i in range( len( corpus ) ):

                # iterate the tokens of a document
                for j in range( len( corpus[ i ] ) ):

                    # if the term exists in index/vocabulary
                    if corpus[ i ][ j ] in index:

                        # if term not already occurred in document
                        if i not in index[ corpus[ i ][ j ] ]: # type: ignore
                            index[ corpus[ i ][ j ] ][ i ] = [] # type: ignore

                        # add the term occurrence
                        index[ corpus[ i ][ j ] ][ i ].append( j ) # type: ignore

            return index
        
        return create_index( '\nCreating index...', corpus=corpus, vocabulary=vocabulary )


def make_and_save( 
    pickle_paths:dict,
    vocabulary_descr:str, 
    corpus:list[str], 
    PreprocessorClass, 
):
    vocabulary_filename = f"{pickle_paths[ 'vocabularies' ]}/{vocabulary_descr}.pkl"
    indexMaker = IndexMaker(
        corpus,
        PreprocessorClass(),
        PickleLoader( vocabulary_filename )
    )
    index = indexMaker.make()

    index_filename = f"{pickle_paths[ 'indexes' ]}/{vocabulary_descr}.pkl"
    PickleSaver( index_filename ).save( index )


# RUN: python -m arXiv.IndexMaker [option]
if __name__ == "__main__": 

    option = None
    if len( sys.argv ) > 1:
        option = sys.argv[ 1 ]

    match option:

        case 'arxiv-stemm-single':
            from ..datasets.arXiv.Dataset import Dataset
            from ..datasets.arXiv.settings import pickle_paths
            make_and_save(
                pickle_paths,
                vocabulary_descr='title-summary_lower-punct-specials-stops-stemm_single',
                corpus=Dataset().toListTitlesSummaries(),
                PreprocessorClass=StemmPreprocessor
            )

        case 'arxiv-lemm-single':
            from ..datasets.arXiv.Dataset import Dataset
            from ..datasets.arXiv.settings import pickle_paths
            make_and_save(
                pickle_paths,
                vocabulary_descr='title-summary_lower-punct-specials-stops-lemm_single',
                corpus=Dataset().toListTitlesSummaries(),
                PreprocessorClass=LemmPreprocessor
            )

        case 'medical-lemm-single':
            from ..datasets.medical.Dataset import Dataset
            from ..datasets.medical.settings import pickle_paths
            make_and_save(
                pickle_paths,
                vocabulary_descr='title-summary_lower-punct-specials-stops-lemm_single',
                corpus=Dataset().toListTitlesAbstracts(),
                PreprocessorClass=LemmPreprocessor
            )

        case _:
            raise Exception( 'No valid option.' )
