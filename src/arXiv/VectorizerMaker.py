import pickle
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from .CorpusLoader import CorpusLoader, TitleSummaryCorpusLoader
from .Preprocessor import Preprocessor, LemmPreprocessor
from .VocabularyLoader import VocabularyLoader

from .settings import pickle_paths

from .helpers.decorators import with_time_counter

class VectorizerMaker:

    def __init__( 
            self, 
            corpusLoader:CorpusLoader, 
            preprocessor:Preprocessor, 
            vocabularyLoader:VocabularyLoader, 
            VectorizerClass:type[CountVectorizer|TfidfVectorizer]
        ):
        self._corpusLoader = corpusLoader
        self._preprocessor = preprocessor
        self._vocabularyLoader = vocabularyLoader
        self._VectorizerClass = Vectorizer

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

        self._vectorizer, self._corpus_repr = create_vectorizer( '\nCreating vectorizer...' )
        return self._vectorizer, self._corpus_repr

    def save( self, descr ):

        print( f'Saving vectorizer {descr}.pkl in disk...' )
        with open( f"{pickle_paths[ 'vectorizers' ]}/{descr}.pkl", 'wb' ) as f:
            pickle.dump( self._vectorizer, f )

        print( f'Saving corpus repr {descr}.pkl in disk...' )
        with open( f"{pickle_paths[ 'corpus_repr' ]}/{descr}.pkl", 'wb' ) as f:
            pickle.dump( self._corpus_repr, f )

    def __str__( self ):
        return self.__class__


# RUN: python -m arXiv.VectorizerMaker
if __name__ == "__main__": 

    voc_descr = 'title-summary_lower-punct-specials-stops-lemm_single'

    # vec_descr = 'title-summary_lower-punct-specials-stops-lemm_single_count'
    # Vectorizer = CountVectorizer

    vec_descr = 'title-summary_lower-punct-specials-stops-lemm_single_tfidf'
    Vectorizer = TfidfVectorizer

    vectorizerMaker = VectorizerMaker(
        TitleSummaryCorpusLoader(),
        LemmPreprocessor(),
        VocabularyLoader( voc_descr ),
        CountVectorizer
    )
    vectorizerMaker.make()
    vectorizerMaker.save( vec_descr )