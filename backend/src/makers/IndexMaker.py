import sys
from abc import ABC, abstractmethod

import nltk
nltk.download( 'punkt_tab' ) # required by word_tokenize()
from nltk.tokenize import word_tokenize

import pytrie

from ..Preprocessor import Preprocessor, LemmPreprocessor, StemmPreprocessor
from ..makers.Tokenizer import AbstractTokenizer, SingleTokenizer, SingleAndTwogramTokenizer
from ..helpers.decorators import with_time_counter
from ..helpers.Pickle import PickleLoader, PickleSaver
from ..helpers.Timer import Timer

# ------------------------------------------------- #
# Class to create reverse indexes from vocabularies #
# ------------------------------------------------- #

class AbstractIndexMaker( ABC ):

    def __init__( 
        self, 
        corpus:list[str], 
        preprocessor:Preprocessor,
        tokenizer:AbstractTokenizer,
        vocabulary_filename:str
    ):
        self._corpus = corpus
        self._preprocessor = preprocessor
        self._tokenizer = tokenizer
        self._vocabulary_filename = vocabulary_filename

    @abstractmethod
    def make( self ):
        pass

    def __str__( self ):
        return self.__class__


class TrieIndexMaker( AbstractIndexMaker ):

    def make( self ):

        print( f'\nPreprocessing...' )
        timer = Timer( start=True )        
        vocabulary = PickleLoader( self._vocabulary_filename ).load()
        corpus = self._preprocessor.transform( self._corpus )
        tokenized_corpus:list[list[str]] = [ self._tokenizer.tokenize( doc ) for doc in corpus ]
        print( f'(passed {timer.stop()} secs)' )

        print( f'\nCreating index...' )
        timer = Timer( start=True )        

        # initialize index with all terms  
        index = pytrie.StringTrie()
        for term in vocabulary:
            index[ term ] = {}

        # iterate the documents of the corpus
        for i in range( len( tokenized_corpus ) ):

            # iterate the tokens of a document
            for j in range( len( tokenized_corpus[ i ] ) ):

                # if the term exists in index/vocabulary
                if tokenized_corpus[ i ][ j ] in index:

                    # if term not already occurred in document
                    if i not in index[ tokenized_corpus[ i ][ j ] ]: # type: ignore
                        index[ tokenized_corpus[ i ][ j ] ][ i ] = [] # type: ignore

                    # add the term occurrence
                    index[ tokenized_corpus[ i ][ j ] ][ i ].append( j ) # type: ignore

        print( f'(passed {timer.stop()} secs)' )
        return index


def make_and_save( 
    pickle_paths:dict,
    vocabulary_descr:str, 
    corpus:list[str],
    PreprocessorClass,
    TokenizerClass, 
):
    
    # make index
    vocabulary_filename = f"{pickle_paths[ 'vocabularies' ]}/{vocabulary_descr}.pkl"
    indexMaker = TrieIndexMaker(
        corpus,
        PreprocessorClass(),
        TokenizerClass(),
        vocabulary_filename
    )
    index = indexMaker.make()

    # save index   
    index_filename = f"{pickle_paths[ 'indexes' ]}/{vocabulary_descr}.pkl"
    PickleSaver( index_filename ).save( index )


# RUN: python -m src.makers.IndexMaker [option]
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
                PreprocessorClass=StemmPreprocessor,
                TokenizerClass=SingleTokenizer
            )

        case 'arxiv-lemm-single':
            from ..datasets.arXiv.Dataset import Dataset
            from ..datasets.arXiv.settings import pickle_paths
            make_and_save(
                pickle_paths,
                vocabulary_descr='title-summary_lower-punct-specials-stops-lemm_single',
                corpus=Dataset().toListTitlesSummaries(),
                PreprocessorClass=LemmPreprocessor,
                TokenizerClass=SingleTokenizer
            )

        case 'arxiv-lemm-2gram':
            from ..datasets.arXiv.Dataset import Dataset
            from ..datasets.arXiv.settings import pickle_paths
            make_and_save(
                pickle_paths,
                vocabulary_descr='title-summary_lower-punct-specials-stops-lemm_2gram',
                corpus=Dataset().toListTitlesSummaries(),
                PreprocessorClass=LemmPreprocessor,
                TokenizerClass=SingleAndTwogramTokenizer
            )

        case 'medical-lemm-single':
            from ..datasets.medical.Dataset import Dataset
            from ..datasets.medical.settings import pickle_paths
            make_and_save(
                pickle_paths,
                vocabulary_descr='title-summary_lower-punct-specials-stops-lemm_single',
                corpus=Dataset().toListTitlesAbstracts(),
                PreprocessorClass=LemmPreprocessor,
                TokenizerClass=SingleTokenizer
            )

        case 'medical-lemm-2gram':
            from ..datasets.medical.Dataset import Dataset
            from ..datasets.medical.settings import pickle_paths
            make_and_save(
                pickle_paths,
                vocabulary_descr='title-summary_lower-punct-specials-stops-lemm_2gram',
                corpus=Dataset().toListTitlesAbstracts(),
                PreprocessorClass=LemmPreprocessor,
                TokenizerClass=SingleAndTwogramTokenizer
            )

        case _:
            raise Exception( 'No valid option.' )
