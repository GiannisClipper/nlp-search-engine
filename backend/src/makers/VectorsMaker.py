import sys
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from ..Preprocessor import Preprocessor, LemmPreprocessor, StemmPreprocessor
from ..helpers.Pickle import PickleLoader, PickleSaver
from ..helpers.Timer import Timer

# -------------------------------------------------- #
# Class to create vectors (tf or tfidf) from corpora #
# -------------------------------------------------- #

class VectorsMaker:

    def __init__( 
            self, 
            VectorizerClass:type[CountVectorizer|TfidfVectorizer],
            vocabulary:list[str], 
            corpus:list[str], 
            preprocessor:Preprocessor, 
        ):
        self._vocabulary = vocabulary
        self._VectorizerClass = VectorizerClass
        self._corpus = corpus
        self._preprocessor = preprocessor

    def make( self ):
        print( f'Preprocessing...' )
        timer = Timer( start=True )        
        corpus = self._preprocessor.transform( self._corpus )
        print( f'(passed {timer.stop()} secs)' )

        print( f'Creating vector representations...' )
        timer = Timer( start=True )        
        vectorizer = self._VectorizerClass( vocabulary=self._vocabulary )
        corpus_repr = vectorizer.fit_transform( corpus )
        print( f'(passed {timer.stop()} secs)' )

        return vectorizer, corpus_repr

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

    vocabulary = PickleLoader( f"{pickle_paths[ 'vocabularies' ]}/{vocabulary_descr}.pkl" ).load()

    vectorizerMaker = VectorsMaker(
        VectorizerClass,
        vocabulary,
        corpus,
        PreprocessorClass(),
    )
    vectorizer, corpus_repr = vectorizerMaker.make()

    vectorizer_filename = f"{pickle_paths[ 'vectorizers' ]}/{vectorizer_descr}.pkl"
    PickleSaver( vectorizer_filename ).save( vectorizer )

    # corpus representation has common description with vectorizer
    corpus_repr_filename = f"{pickle_paths[ 'corpus_repr' ]}/{vectorizer_descr}.pkl"
    PickleSaver( corpus_repr_filename ).save( corpus_repr )


# RUN: python -m src.makers.VectorsMaker [option]
if __name__ == "__main__": 

    option = None
    if len( sys.argv ) >= 2:
        option = sys.argv[ 1 ]

    match option:

        case 'arxiv-stemm-single-count':
            from ..datasets.arXiv.Dataset import Dataset
            from ..datasets.arXiv.settings import pickle_paths
            make_and_save( 
                pickle_paths, 
                vocabulary_descr='title-summary_lower-punct-specials-stops-stemm_single', 
                vectorizer_descr='title-summary_lower-punct-specials-stops-stemm_single_count', 
                corpus=Dataset().toListTitlesSummaries(), 
                PreprocessorClass=StemmPreprocessor,
                VectorizerClass=CountVectorizer
            )

        case 'arxiv-stemm-single-tfidf':
            from ..datasets.arXiv.Dataset import Dataset
            from ..datasets.arXiv.settings import pickle_paths
            make_and_save( 
                pickle_paths, 
                vocabulary_descr='title-summary_lower-punct-specials-stops-stemm_single', 
                vectorizer_descr='title-summary_lower-punct-specials-stops-stemm_single_tfidf', 
                corpus=Dataset().toListTitlesSummaries(), 
                PreprocessorClass=StemmPreprocessor,
                VectorizerClass=TfidfVectorizer
            )

        case 'arxiv-lemm-single-count':
            from ..datasets.arXiv.Dataset import Dataset
            from ..datasets.arXiv.settings import pickle_paths
            make_and_save( 
                pickle_paths, 
                vocabulary_descr='title-summary_lower-punct-specials-stops-lemm_single', 
                vectorizer_descr='title-summary_lower-punct-specials-stops-lemm_single_count', 
                corpus=Dataset().toListTitlesSummaries(), 
                PreprocessorClass=LemmPreprocessor,
                VectorizerClass=CountVectorizer
            )

        case 'arxiv-lemm-single-tfidf':
            from ..datasets.arXiv.Dataset import Dataset
            from ..datasets.arXiv.settings import pickle_paths
            make_and_save( 
                pickle_paths, 
                vocabulary_descr='title-summary_lower-punct-specials-stops-lemm_single', 
                vectorizer_descr='title-summary_lower-punct-specials-stops-lemm_single_tfidf', 
                corpus=Dataset().toListTitlesSummaries(), 
                PreprocessorClass=LemmPreprocessor,
                VectorizerClass=TfidfVectorizer
            )

        case 'arxiv-lemm-2gram-tfidf':
            from ..datasets.arXiv.Dataset import Dataset
            from ..datasets.arXiv.settings import pickle_paths
            make_and_save( 
                pickle_paths, 
                vocabulary_descr='title-summary_lower-punct-specials-stops-lemm_2gram', 
                vectorizer_descr='title-summary_lower-punct-specials-stops-lemm_2gram_tfidf', 
                corpus=Dataset().toListTitlesSummaries(), 
                PreprocessorClass=LemmPreprocessor,
                VectorizerClass=TfidfVectorizer
            )

        case 'medical-lemm-single-tfidf':
            from ..datasets.medical.Dataset import Dataset
            from ..datasets.medical.settings import pickle_paths
            make_and_save( 
                pickle_paths,
                vocabulary_descr='title-summary_lower-punct-specials-stops-lemm_single', 
                vectorizer_descr='title-summary_lower-punct-specials-stops-lemm_single_tfidf', 
                corpus=Dataset().toListTitlesAbstracts(), 
                PreprocessorClass=LemmPreprocessor,
                VectorizerClass=TfidfVectorizer
            )

        case 'medical-lemm-2gram-tfidf':
            from ..datasets.medical.Dataset import Dataset
            from ..datasets.medical.settings import pickle_paths
            make_and_save( 
                pickle_paths,
                vocabulary_descr='title-summary_lower-punct-specials-stops-lemm_2gram', 
                vectorizer_descr='title-summary_lower-punct-specials-stops-lemm_2gram_tfidf', 
                corpus=Dataset().toListTitlesAbstracts(), 
                PreprocessorClass=LemmPreprocessor,
                VectorizerClass=TfidfVectorizer
            )

        case _:
            raise Exception( 'No valid option.' )
