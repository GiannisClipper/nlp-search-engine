import sys
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from .CorpusLoader import CorpusLoader, TitleSummaryCorpusLoader
from .Preprocessor import Preprocessor, LemmPreprocessor, StemmPreprocessor

from .helpers.decorators import with_time_counter
from .helpers.Pickle import PickleLoader, PickleSaver

class VectorizerMaker:

    def __init__( 
            self, 
            corpusLoader:CorpusLoader, 
            preprocessor:Preprocessor, 
            vocabularyLoader:PickleLoader, 
            VectorizerClass:type[CountVectorizer|TfidfVectorizer]
        ):
        self._corpusLoader = corpusLoader
        self._preprocessor = preprocessor
        self._vocabularyLoader = vocabularyLoader
        self._VectorizerClass = VectorizerClass

    def make( self ):
        print( f'\nPreprocessing...' )
        corpus = self._corpusLoader.load()
        corpus = self._preprocessor.transform( corpus )
        vocabulary = self._vocabularyLoader.load()

        @with_time_counter
        def create_vectorizer( message=None, *args, **kwargs ):
            vectorizer = self._VectorizerClass( vocabulary=vocabulary )
            corpus_repr = vectorizer.fit_transform( corpus )
            return vectorizer, corpus_repr

        return create_vectorizer( '\nCreating vectorizer...' )

    def __str__( self ):
        return self.__class__


# RUN: python -m src.VectorizerMaker [parameter]
if __name__ == "__main__": 

    from .arXiv.settings import pickle_paths

    option = None
    if len( sys.argv ) > 1:
        option = sys.argv[ 1 ]

    match option:

        case 'stemm-count':

            vocabulary_descr = 'title-summary_lower-punct-specials-stops-stemm_single'
            vocabulary_filename = f"{pickle_paths[ 'vocabularies' ]}/{vocabulary_descr}.pkl"

            vectorizer_descr = 'title-summary_lower-punct-specials-stops-stemm_single_count'
            vectorizerMaker = VectorizerMaker(
                TitleSummaryCorpusLoader(),
                StemmPreprocessor(),
                PickleLoader( vocabulary_filename ),
                CountVectorizer
            )
            vectorizer, corpus_repr = vectorizerMaker.make()

            vectorizer_filename = f"{pickle_paths[ 'vectorizers' ]}/{vectorizer_descr}.pkl"
            PickleSaver( vectorizer_filename ).save( vectorizer )

            # common description with vectorizer
            corpus_repr_filename = f"{pickle_paths[ 'corpus_repr' ]}/{vectorizer_descr}.pkl"
            PickleSaver( corpus_repr_filename ).save( corpus_repr )

        case 'stemm-tfidf':

            vocabulary_descr = 'title-summary_lower-punct-specials-stops-stemm_single'
            vocabulary_filename = f"{pickle_paths[ 'vocabularies' ]}/{vocabulary_descr}.pkl"

            vectorizer_descr = 'title-summary_lower-punct-specials-stops-stemm_single_tfidf'
            vectorizerMaker = VectorizerMaker(
                TitleSummaryCorpusLoader(),
                StemmPreprocessor(),
                PickleLoader( vocabulary_filename ),
                TfidfVectorizer
            )
            vectorizer, corpus_repr = vectorizerMaker.make()

            vectorizer_filename = f"{pickle_paths[ 'vectorizers' ]}/{vectorizer_descr}.pkl"
            PickleSaver( vectorizer_filename ).save( vectorizer )

            # common description with vectorizer
            corpus_repr_filename = f"{pickle_paths[ 'corpus_repr' ]}/{vectorizer_descr}.pkl"
            PickleSaver( corpus_repr_filename ).save( corpus_repr )

        case 'lemm-count':

            vocabulary_descr = 'title-summary_lower-punct-specials-stops-lemm_single'
            vocabulary_filename = f"{pickle_paths[ 'vocabularies' ]}/{vocabulary_descr}.pkl"

            vectorizer_descr = 'title-summary_lower-punct-specials-stops-lemm_single_count'
            vectorizerMaker = VectorizerMaker(
                TitleSummaryCorpusLoader(),
                LemmPreprocessor(),
                PickleLoader( vocabulary_filename ),
                CountVectorizer
            )
            vectorizer, corpus_repr = vectorizerMaker.make()

            vectorizer_filename = f"{pickle_paths[ 'vectorizers' ]}/{vectorizer_descr}.pkl"
            PickleSaver( vectorizer_filename ).save( vectorizer )

            # common description with vectorizer
            corpus_repr_filename = f"{pickle_paths[ 'corpus_repr' ]}/{vectorizer_descr}.pkl"
            PickleSaver( corpus_repr_filename ).save( corpus_repr )

        case 'lemm-tfidf':

            vocabulary_descr = 'title-summary_lower-punct-specials-stops-lemm_single'
            vocabulary_filename = f"{pickle_paths[ 'vocabularies' ]}/{vocabulary_descr}.pkl"

            vectorizer_descr = 'title-summary_lower-punct-specials-stops-lemm_single_tfidf'
            vectorizerMaker = VectorizerMaker(
                TitleSummaryCorpusLoader(),
                LemmPreprocessor(),
                PickleLoader( vocabulary_filename ),
                TfidfVectorizer
            )
            vectorizer, corpus_repr = vectorizerMaker.make()

            vectorizer_filename = f"{pickle_paths[ 'vectorizers' ]}/{vectorizer_descr}.pkl"
            PickleSaver( vectorizer_filename ).save( vectorizer )

            # common description with vectorizer
            corpus_repr_filename = f"{pickle_paths[ 'corpus_repr' ]}/{vectorizer_descr}.pkl"
            PickleSaver( corpus_repr_filename ).save( corpus_repr )

        case _:
            raise Exception( 'No valid parameters passed.' )
