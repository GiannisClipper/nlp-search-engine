import sys
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from .Preprocessor import Preprocessor, LemmPreprocessor, StemmPreprocessor

from .helpers.decorators import with_time_counter
from .helpers.Pickle import PickleLoader, PickleSaver

class VectorizerMaker:

    def __init__( 
            self, 
            corpus:list[str], 
            preprocessor:Preprocessor, 
            vocabularyLoader:PickleLoader, 
            VectorizerClass:type[CountVectorizer|TfidfVectorizer]
        ):
        self._corpus = corpus
        self._preprocessor = preprocessor
        self._vocabularyLoader = vocabularyLoader
        self._VectorizerClass = VectorizerClass

    def make( self ):
        print( f'\nPreprocessing...' )
        corpus = self._preprocessor.transform( self._corpus )
        vocabulary = self._vocabularyLoader.load()

        @with_time_counter
        def create_vectorizer( message=None, *args, **kwargs ):
            vectorizer = self._VectorizerClass( vocabulary=vocabulary )
            corpus_repr = vectorizer.fit_transform( corpus )
            return vectorizer, corpus_repr

        return create_vectorizer( '\nCreating vectorizer...' )

    def __str__( self ):
        return self.__class__    


def make_and_save( 
    pickle_paths:dict,
    vocabulary_descr:str, 
    vectorizer_descr:str, 
    corpus:list[str], 
    PreprocessorClass, 
    VectorizerClass
):

    vocabulary_filename = f"{pickle_paths[ 'vocabularies' ]}/{vocabulary_descr}.pkl"

    vectorizerMaker = VectorizerMaker(
        corpus,
        PreprocessorClass(),
        PickleLoader( vocabulary_filename ),
        VectorizerClass
    )
    vectorizer, corpus_repr = vectorizerMaker.make()

    vectorizer_filename = f"{pickle_paths[ 'vectorizers' ]}/{vectorizer_descr}.pkl"
    PickleSaver( vectorizer_filename ).save( vectorizer )

    # common description with vectorizer
    corpus_repr_filename = f"{pickle_paths[ 'corpus_repr' ]}/{vectorizer_descr}.pkl"
    PickleSaver( corpus_repr_filename ).save( corpus_repr )


# RUN: python -m src.VectorizerMaker [option]
if __name__ == "__main__": 

    option = None
    if len( sys.argv ) >= 2:
        option = sys.argv[ 1 ]

    match option:

        case 'arxiv-stemm-single-count':
            from .arXiv.Dataset import Dataset
            from .arXiv.settings import pickle_paths
            make_and_save( 
                pickle_paths, 
                vocabulary_descr='title-summary_lower-punct-specials-stops-stemm_single', 
                vectorizer_descr='title-summary_lower-punct-specials-stops-stemm_single_count', 
                corpus=Dataset().toListTitlesSummaries(), 
                PreprocessorClass=StemmPreprocessor,
                VectorizerClass=CountVectorizer
            )

        case 'arxiv-stemm-single-tfidf':
            from .arXiv.Dataset import Dataset
            from .arXiv.settings import pickle_paths
            make_and_save( 
                pickle_paths, 
                vocabulary_descr='title-summary_lower-punct-specials-stops-stemm_single', 
                vectorizer_descr='title-summary_lower-punct-specials-stops-stemm_single_tfidf', 
                corpus=Dataset().toListTitlesSummaries(), 
                PreprocessorClass=StemmPreprocessor,
                VectorizerClass=TfidfVectorizer
            )

        case 'arxiv-lemm-single-count':
            from .arXiv.Dataset import Dataset
            from .arXiv.settings import pickle_paths
            make_and_save( 
                pickle_paths, 
                vocabulary_descr='title-summary_lower-punct-specials-stops-lemm_single', 
                vectorizer_descr='title-summary_lower-punct-specials-stops-lemm_single_count', 
                corpus=Dataset().toListTitlesSummaries(), 
                PreprocessorClass=LemmPreprocessor,
                VectorizerClass=CountVectorizer
            )

        case 'arxiv-lemm-single-tfidf':
            from .arXiv.Dataset import Dataset
            from .arXiv.settings import pickle_paths
            make_and_save( 
                pickle_paths, 
                vocabulary_descr='title-summary_lower-punct-specials-stops-lemm_single', 
                vectorizer_descr='title-summary_lower-punct-specials-stops-lemm_single_tfidf', 
                corpus=Dataset().toListTitlesSummaries(), 
                PreprocessorClass=LemmPreprocessor,
                VectorizerClass=TfidfVectorizer
            )

        case 'medical-lemm-single-tfidf':
            from .medical.Dataset import Dataset
            from .medical.settings import pickle_paths
            make_and_save( 
                pickle_paths,
                vocabulary_descr='title-summary_lower-punct-specials-stops-lemm_single', 
                vectorizer_descr='title-summary_lower-punct-specials-stops-lemm_single_tfidf', 
                corpus=Dataset().toListTitlesAbstracts(), 
                PreprocessorClass=LemmPreprocessor,
                VectorizerClass=TfidfVectorizer
            )

        case _:
            raise Exception( 'No valid option.' )
