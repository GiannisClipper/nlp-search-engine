import sys

from .Preprocessor import Preprocessor, LemmPreprocessor, StemmPreprocessor
from .TokenMaker import TokenMaker, SingleTokenMaker

from .helpers.decorators import with_time_counter
from .helpers.Pickle import PickleSaver

class VocabularyMaker:

    def __init__( 
            self, 
            corpus:list[str], 
            preprocessor:Preprocessor, 
            tokenMaker:TokenMaker 
        ):
        self._corpus = corpus
        self._preprocessor = preprocessor
        self._tokenMaker = tokenMaker

    def make( self ) -> tuple[str,...]:
        print( f'\nPreprocessing...' )
        corpus = self._preprocessor.transform( self._corpus )
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


def make_and_save( 
    pickle_paths:dict,
    vocabulary_decsr:str,
    corpus:list[str], 
    preprocessor:Preprocessor,
    tokenMaker:TokenMaker
):
    vocabularyMaker = VocabularyMaker( corpus, preprocessor, tokenMaker )
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
            from .arXiv.Dataset import Dataset
            from .arXiv.settings import pickle_paths
            make_and_save( 
                pickle_paths,
                vocabulary_decsr='title-summary_lower-punct-specials-stops-stemm_single',
                corpus=Dataset().toListTitlesSummaries(),
                preprocessor=StemmPreprocessor(),
                tokenMaker=SingleTokenMaker()
            )

        case 'arxiv-lemm-single':
            from .arXiv.Dataset import Dataset
            from .arXiv.settings import pickle_paths
            make_and_save( 
                pickle_paths,
                vocabulary_decsr='title-summary_lower-punct-specials-stops-lemm_single',
                corpus=Dataset().toListTitlesSummaries(),
                preprocessor=LemmPreprocessor(),
                tokenMaker=SingleTokenMaker()
            )
                
        case 'medical-lemm-single':
            from .medical.Dataset import Dataset
            from .medical.settings import pickle_paths
            make_and_save( 
                pickle_paths,
                vocabulary_decsr='title-summary_lower-punct-specials-stops-lemm_single',
                corpus=Dataset().toListTitlesAbstracts(),
                preprocessor=LemmPreprocessor(),
                tokenMaker=SingleTokenMaker()
            )

        case _:
            raise Exception( 'No valid option.' )
