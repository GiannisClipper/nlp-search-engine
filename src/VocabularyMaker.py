import sys
from abc import ABC, abstractmethod

from .CorpusLoader import CorpusLoader, TitleSummaryCorpusLoader
from .Preprocessor import Preprocessor, LemmPreprocessor, StemmPreprocessor

from .TokenMaker import TokenMaker, SingleTokenMaker

from .helpers.decorators import with_time_counter
from .helpers.Pickle import PickleSaver

class VocabularyMaker:

    def __init__( self, corpusLoader:CorpusLoader, preprocessor:Preprocessor, tokenMaker:TokenMaker ):
        self._corpusLoader = corpusLoader
        self._preprocessor = preprocessor
        self._tokenMaker = tokenMaker

    def make( self ) -> tuple[str,...]:
        print( f'\nPreprocessing...' )
        corpus = self._corpusLoader.load()
        corpus = self._preprocessor.transform( corpus )
        step = len( corpus ) // 5 if len( corpus ) > 5 else 1 
        print( f'corpus[::{step}]:', corpus[::step] )

        @with_time_counter
        def create_vocabulary( message=None, *args, **kwargs ):
            return self._tokenMaker.make( ' '.join( corpus ) )

        tokens = create_vocabulary( '\nCreating vocabulary...' )
        step = len( tokens ) // 50 if len( tokens ) > 50 else 1 
        print( f'\nVocabulary[::{step}]:', len( tokens ), tokens[::step] )
        return tokens

    def vocabulary( self ):
        return self._tokenMaker.tokens

    def __str__( self ):
        return self.__class__


# RUN: python -m src.VocabularyMaker [parameter]
if __name__ == "__main__": 

    from .arXiv.settings import pickle_paths

    option = None
    if len( sys.argv ) > 1:
        option = sys.argv[ 1 ]

    match option:

        case 'stemm-single':

            vocabularyMaker = VocabularyMaker(
                TitleSummaryCorpusLoader(),
                StemmPreprocessor(),
                SingleTokenMaker()
            )
            vocabulary = vocabularyMaker.make()

            vocabulary_decsr = 'title-summary_lower-punct-specials-stops-stemm_single'
            vocabulary_filename = f"{pickle_paths[ 'vocabularies' ]}/{vocabulary_decsr}.pkl"
            PickleSaver( vocabulary_filename ).save( vocabulary )

        case 'lemm-single':

            vocabularyMaker = VocabularyMaker(
                TitleSummaryCorpusLoader(),
                LemmPreprocessor(),
                SingleTokenMaker()
            )
            vocabulary = vocabularyMaker.make()

            vocabulary_decsr = 'title-summary_lower-punct-specials-stops-lemm_single'
            vocabulary_filename = f"{pickle_paths[ 'vocabularies' ]}/{vocabulary_decsr}.pkl"
            PickleSaver( vocabulary_filename ).save( vocabulary )

        case _:
            raise Exception( 'No valid parameters passed.' )
