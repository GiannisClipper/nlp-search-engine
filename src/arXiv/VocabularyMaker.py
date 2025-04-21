from abc import ABC, abstractmethod

import pickle

from .CorpusLoader import CorpusLoader, TitleSummaryCorpusLoader
from .Preprocessor import Preprocessor, LemmPreprocessor, StemmPreprocessor

from .TokenMaker import TokenMaker, SingleTokenMaker
from .settings import pickle_paths

from .helpers.decorators import with_time_counter

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

    def save( self, descr ):

        print( f'Saving vocabulary {descr}.pkl in disk...' )
        with open( f"{pickle_paths[ 'vocabularies' ]}/{descr}.pkl", 'wb' ) as f:
            pickle.dump( self._tokenMaker.tokens, f )

    def vocabulary( self ):
        return self._tokenMaker.tokens

    def __str__( self ):
        return self.__class__


# RUN: python -m arXiv.VocabularyMaker
if __name__ == "__main__": 

    voc_decsr = 'title-summary_lower-punct-specials-stops-stemm_single'
    PreprocessorClass = StemmPreprocessor

    # voc_decsr = 'title-summary_lower-punct-specials-stops-lemm_single'
    # PreprocessorClass = LemmPreprocessor

    vocabularyMaker = VocabularyMaker(
        TitleSummaryCorpusLoader(),
        PreprocessorClass(),
        SingleTokenMaker()
    )
    vocabularyMaker.make()
    vocabularyMaker.save( voc_decsr )