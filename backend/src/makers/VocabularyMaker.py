import sys

from ..Preprocessor import Preprocessor, LemmPreprocessor, StemmPreprocessor
from .TermsMaker import AbstractTermsMaker, SingleTermsMaker, TwogramTermsMaker
from ..helpers.Pickle import PickleSaver
from ..helpers.Timer import Timer

# ----------------------------------------- #
# Class to create vocabularies from corpora #
# ----------------------------------------- #

class VocabularyMaker:

    def __init__( 
            self, 
            corpus:list[str], 
            preprocessor:Preprocessor, 
            termsMakers:list[AbstractTermsMaker] 
        ):
        self._corpus = corpus
        self._preprocessor = preprocessor
        self._termsMakers = termsMakers
        self._terms:list[str]

    def make( self ) -> list[str]:
        print( f'\nPreprocessing...' )
        corpus = self._preprocessor.transform( self._corpus )
 
        # print some indicative output
        # step = len( corpus ) // 5 if len( corpus ) > 5 else 1 
        # print( f'corpus[::{step}]:', corpus[::step] )

        print( 'Creating vocabulary...' )
        timer = Timer( start=True )        
        result = []
        for termsMaker in self._termsMakers:
            result += termsMaker.make( ' '.join( corpus ) )
        self._terms = result
        print( f'(passed {timer.stop()} secs)' )
        print( f'Number of terms:{len(self._terms)}' )

        # print some indicative output
        # step = len( self._terms ) // 50 if len( self._terms ) > 50 else 1 
        # print( f'\nVocabulary[::{step}]:', len( self._terms ), self._terms[::step] )
 
        return self._terms

    def vocabulary( self ) -> list[str]:
        return self._terms

    def __str__( self ):
        return self.__class__


def make_and_save( 
    pickle_paths:dict,
    vocabulary_decsr:str,
    corpus:list[str], 
    preprocessor:Preprocessor,
    termsMakers:list[AbstractTermsMaker]
):
    vocabularyMaker = VocabularyMaker( corpus, preprocessor, termsMakers )
    vocabulary = vocabularyMaker.make()

    vocabulary_filename = f"{pickle_paths[ 'vocabularies' ]}/{vocabulary_decsr}.pkl"
    PickleSaver( vocabulary_filename ).save( vocabulary )


# RUN: python -m src.VocabularyMaker [option]
if __name__ == "__main__":

    option = None
    if len( sys.argv ) >= 2:
        option = sys.argv[ 1 ]

    match option:

        case 'arxiv-stemm-single':
            from ..datasets.arXiv.Dataset import Dataset
            from ..datasets.arXiv.settings import pickle_paths
            make_and_save( 
                pickle_paths,
                vocabulary_decsr='title-summary_lower-punct-specials-stops-stemm_single',
                corpus=Dataset().toListTitlesSummaries(),
                preprocessor=StemmPreprocessor(),
                termsMakers=[ SingleTermsMaker() ]
            )

        case 'arxiv-lemm-single':
            from ..datasets.arXiv.Dataset import Dataset
            from ..datasets.arXiv.settings import pickle_paths
            make_and_save( 
                pickle_paths,
                vocabulary_decsr='title-summary_lower-punct-specials-stops-lemm_single',
                corpus=Dataset().toListTitlesSummaries(),
                preprocessor=LemmPreprocessor(),
                termsMakers=[ SingleTermsMaker() ]
            )

        case 'arxiv-lemm-2gram':
            from ..datasets.arXiv.Dataset import Dataset
            from ..datasets.arXiv.settings import pickle_paths
            make_and_save( 
                pickle_paths,
                vocabulary_decsr='title-summary_lower-punct-specials-stops-lemm_2gram',
                corpus=Dataset().toListTitlesSummaries(),
                preprocessor=LemmPreprocessor(),
                termsMakers=[ SingleTermsMaker(), TwogramTermsMaker( limit=1000 ) ]
            )

        ###########
        # medical #
        ###########

        case 'medical-stemm-single':
            from ..datasets.medical.Dataset import Dataset
            from ..datasets.medical.settings import pickle_paths
            make_and_save( 
                pickle_paths,
                vocabulary_decsr='title-summary_lower-punct-specials-stops-stemm_single',
                corpus=Dataset().toListTitlesAbstracts(),
                preprocessor=StemmPreprocessor(),
                termsMakers=[ SingleTermsMaker() ]
            )

        case 'medical-lemm-single':
            from ..datasets.medical.Dataset import Dataset
            from ..datasets.medical.settings import pickle_paths
            make_and_save( 
                pickle_paths,
                vocabulary_decsr='title-summary_lower-punct-specials-stops-lemm_single',
                corpus=Dataset().toListTitlesAbstracts(),
                preprocessor=LemmPreprocessor(),
                termsMakers=[ SingleTermsMaker() ]
            )

        case 'medical-lemm-2gram':
            from ..datasets.medical.Dataset import Dataset
            from ..datasets.medical.settings import pickle_paths
            make_and_save( 
                pickle_paths,
                vocabulary_decsr='title-summary_lower-punct-specials-stops-lemm_2gram',
                corpus=Dataset().toListTitlesAbstracts(),
                preprocessor=LemmPreprocessor(),
                termsMakers=[ SingleTermsMaker(), TwogramTermsMaker( limit=1000 ) ]
            )

        case _:
            raise Exception( 'No valid option.' )
