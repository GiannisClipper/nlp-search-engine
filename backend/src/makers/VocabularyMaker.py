import sys

from ..Preprocessor import Preprocessor, LemmPreprocessor, StemmPreprocessor
from .TokenMaker import TokenMaker, SingleTokenMaker, TwogramTokenMaker

from ..helpers.decorators import with_time_counter
from ..helpers.Pickle import PickleSaver

class VocabularyMaker:

    def __init__( 
            self, 
            corpus:list[str], 
            preprocessor:Preprocessor, 
            tokenMakers:list[TokenMaker] 
        ):
        self._corpus = corpus
        self._preprocessor = preprocessor
        self._tokenMakers = tokenMakers
        self._tokens:list[str]

    def make( self ) -> list[str]:
        print( f'\nPreprocessing...' )
        corpus = self._preprocessor.transform( self._corpus )
        step = len( corpus ) // 5 if len( corpus ) > 5 else 1 
        print( f'corpus[::{step}]:', corpus[::step] )

        @with_time_counter
        def create_vocabulary( message=None, *args, **kwargs ):
            result = []
            for tokenMaker in self._tokenMakers:
                result += tokenMaker.make( ' '.join( corpus ) )
            return result

        self._tokens = create_vocabulary( '\nCreating vocabulary...' )
        step = len( self._tokens ) // 50 if len( self._tokens ) > 50 else 1 
        print( f'\nVocabulary[::{step}]:', len( self._tokens ), self._tokens[::step] )
        return self._tokens

    def vocabulary( self ) -> list[str]:
        return self._tokens

    def __str__( self ):
        return self.__class__


def make_and_save( 
    pickle_paths:dict,
    vocabulary_decsr:str,
    corpus:list[str], 
    preprocessor:Preprocessor,
    tokenMakers:list[TokenMaker]
):
    vocabularyMaker = VocabularyMaker( corpus, preprocessor, tokenMakers )
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
                tokenMakers=[ SingleTokenMaker() ]
            )

        case 'arxiv-lemm-single':
            from ..datasets.arXiv.Dataset import Dataset
            from ..datasets.arXiv.settings import pickle_paths
            make_and_save( 
                pickle_paths,
                vocabulary_decsr='title-summary_lower-punct-specials-stops-lemm_single',
                corpus=Dataset().toListTitlesSummaries(),
                preprocessor=LemmPreprocessor(),
                tokenMakers=[ SingleTokenMaker() ]
            )

        case 'arxiv-lemm-2gram':
            from ..datasets.arXiv.Dataset import Dataset
            from ..datasets.arXiv.settings import pickle_paths
            make_and_save( 
                pickle_paths,
                vocabulary_decsr='title-summary_lower-punct-specials-stops-lemm_2gram',
                corpus=Dataset().toListTitlesSummaries(),
                preprocessor=LemmPreprocessor(),
                tokenMakers=[ SingleTokenMaker(), TwogramTokenMaker( limit=1000 ) ]
            )

        case 'medical-lemm-single':
            from ..datasets.medical.Dataset import Dataset
            from ..datasets.medical.settings import pickle_paths
            make_and_save( 
                pickle_paths,
                vocabulary_decsr='title-summary_lower-punct-specials-stops-lemm_single',
                corpus=Dataset().toListTitlesAbstracts(),
                preprocessor=LemmPreprocessor(),
                tokenMakers=[ SingleTokenMaker() ]
            )

        case 'medical-lemm-2gram':
            from ..datasets.medical.Dataset import Dataset
            from ..datasets.medical.settings import pickle_paths
            make_and_save( 
                pickle_paths,
                vocabulary_decsr='title-summary_lower-punct-specials-stops-lemm_2gram',
                corpus=Dataset().toListTitlesAbstracts(),
                preprocessor=LemmPreprocessor(),
                tokenMakers=[ SingleTokenMaker(), TwogramTokenMaker( limit=1000 ) ]
            )

        case _:
            raise Exception( 'No valid option.' )
