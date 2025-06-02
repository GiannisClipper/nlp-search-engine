import sys
from abc import ABC, abstractmethod
from scipy.sparse import spmatrix, csr_matrix
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sentence_transformers import SentenceTransformer

from .Preprocessor import Preprocessor, DummyPreprocessor, LowerWordsPreprocessor, NaivePreprocessor, LemmPreprocessor, StemmPreprocessor
from .makers.Tokenizer import AbstractTokenizer, SingleTokenizer, SingleAndTwogramTokenizer
from .helpers.Pickle import PickleLoader, CachedPickleLoader
from .helpers.typing import QueryAnalyzedType
from .models.GloveModel import GloveModel, gloveModelFactory

# ----------------------------------------------------------------------------- #
# Code to analyze queries into tokens and representation (vector or embeddings) #
# ----------------------------------------------------------------------------- #

class AbstractQueryAnalyzer( ABC ):

    def __init__( self, preprocessor:Preprocessor, tokenizer:AbstractTokenizer ):
        self._preprocessor = preprocessor
        self._tokenizer = tokenizer

    @abstractmethod
    def analyze( self, query:str ) -> QueryAnalyzedType:
        pass


class QueryAnalyzerWithVectorizer( AbstractQueryAnalyzer):

    def __init__( self, preprocessor:Preprocessor, tokenizer:AbstractTokenizer, vectorizer:CountVectorizer|TfidfVectorizer ):
        super().__init__( preprocessor, tokenizer )
        self._vectorizer = vectorizer

    def analyze( self, query:str ) -> QueryAnalyzedType:
        query_preprocessed = self._preprocessor.transform( [ query ] )
        tokens = self._tokenizer.tokenize( query_preprocessed[ 0 ] )
        vectors = self._vectorizer.transform( query_preprocessed )
        return { 'query': query, 'tokens': tokens, 'repr': vectors }


class QueryAnalyzerWithPretrained( AbstractQueryAnalyzer):

    def __init__( self, preprocessor:Preprocessor, tokenizer:AbstractTokenizer, model:SentenceTransformer|GloveModel ):
        super().__init__( preprocessor, tokenizer )
        self._model = model

    def analyze( self, query:str ) -> QueryAnalyzedType:
        query_preprocessed = self._preprocessor.transform( [ query ] )
        tokens = self._tokenizer.tokenize( query_preprocessed[ 0 ] )
        embeddings = self._model.encode( query )
        return { 'query': query, 'tokens': tokens, 'repr': csr_matrix( embeddings ) } # convert to spmatrix


def queryAnalyzerFactory( option:str ) -> AbstractQueryAnalyzer:

    match option:

        case 'arxiv-stemm-single-count':
            from .datasets.arXiv.settings import pickle_paths
            preprocessor = StemmPreprocessor()
            tokenizer = SingleTokenizer()
            vectorizer_descr = 'title-summary_lower-punct-specials-stops-stemm_single_count'
            vectorizer_filename = f"{pickle_paths[ 'vectorizers' ]}/{vectorizer_descr}.pkl"
            vectorizer = CachedPickleLoader( vectorizer_filename ).load()
            return QueryAnalyzerWithVectorizer( preprocessor, tokenizer, vectorizer )

        case 'arxiv-lemm-single-tfidf':
            from .datasets.arXiv.settings import pickle_paths
            preprocessor = LemmPreprocessor()
            tokenizer = SingleTokenizer()
            vectorizer_descr = 'title-summary_lower-punct-specials-stops-lemm_single_tfidf'
            vectorizer_filename = f"{pickle_paths[ 'vectorizers' ]}/{vectorizer_descr}.pkl"
            vectorizer = CachedPickleLoader( vectorizer_filename ).load()
            return QueryAnalyzerWithVectorizer( preprocessor, tokenizer, vectorizer )

        case 'arxiv-lemm-2gram-tfidf':
            from .datasets.arXiv.settings import pickle_paths
            preprocessor = LemmPreprocessor()
            tokenizer = SingleAndTwogramTokenizer()
            vectorizer_descr = 'title-summary_lower-punct-specials-stops-lemm_2gram_tfidf'
            vectorizer_filename = f"{pickle_paths[ 'vectorizers' ]}/{vectorizer_descr}.pkl"
            vectorizer = CachedPickleLoader( vectorizer_filename ).load()
            return QueryAnalyzerWithVectorizer( preprocessor, tokenizer, vectorizer )

        case 'arxiv-naive-glove':
            preprocessor = NaivePreprocessor()
            tokenizer = SingleTokenizer()
            model = gloveModelFactory( 'arxiv' )
            return QueryAnalyzerWithPretrained( preprocessor, tokenizer, model )

        case 'arxiv-naive-glove-retrained':
            preprocessor = NaivePreprocessor()
            tokenizer = SingleTokenizer()
            model = gloveModelFactory( 'arxiv-retrained' )
            return QueryAnalyzerWithPretrained( preprocessor, tokenizer, model )

        ###########
        # medical #
        ###########

        case 'medical-stemm-single-count':
            from .datasets.medical.settings import pickle_paths
            preprocessor = StemmPreprocessor()
            tokenizer = SingleTokenizer()
            vectorizer_descr = 'title-summary_lower-punct-specials-stops-stemm_single_count'
            vectorizer_filename = f"{pickle_paths[ 'vectorizers' ]}/{vectorizer_descr}.pkl"
            vectorizer = CachedPickleLoader( vectorizer_filename ).load()
            return QueryAnalyzerWithVectorizer( preprocessor, tokenizer, vectorizer )

        case 'medical-lemm-single-tfidf':
            from .datasets.medical.settings import pickle_paths
            preprocessor = LemmPreprocessor()
            tokenizer = SingleTokenizer()
            vectorizer_descr = 'title-summary_lower-punct-specials-stops-lemm_single_tfidf'
            vectorizer_filename = f"{pickle_paths[ 'vectorizers' ]}/{vectorizer_descr}.pkl"
            vectorizer = CachedPickleLoader( vectorizer_filename ).load()
            return QueryAnalyzerWithVectorizer( preprocessor, tokenizer, vectorizer )

        case 'medical-lemm-2gram-tfidf':
            from .datasets.medical.settings import pickle_paths
            preprocessor = LemmPreprocessor()
            tokenizer = SingleAndTwogramTokenizer()
            vectorizer_descr = 'title-summary_lower-punct-specials-stops-lemm_2gram_tfidf'
            vectorizer_filename = f"{pickle_paths[ 'vectorizers' ]}/{vectorizer_descr}.pkl"
            vectorizer = CachedPickleLoader( vectorizer_filename ).load()
            return QueryAnalyzerWithVectorizer( preprocessor, tokenizer, vectorizer )

        case 'medical-lemm-single-jina':
            from .datasets.medical.settings import pickle_paths
            preprocessor = LemmPreprocessor()
            tokenizer = SingleTokenizer()
            model = SentenceTransformer( "jinaai/jina-embeddings-v2-base-en", trust_remote_code=True ) #, local_files_only=True
            model.max_seq_length = 1024 # control your input sequence length up to 8192
            return QueryAnalyzerWithPretrained( preprocessor, tokenizer, model )

        case 'medical-naive-glove':
            preprocessor = LowerWordsPreprocessor()
            tokenizer = SingleTokenizer()
            model = gloveModelFactory( 'medical' )
            return QueryAnalyzerWithPretrained( preprocessor, tokenizer, model )

        case 'medical-naive-glove-retrained':
            preprocessor = NaivePreprocessor()
            tokenizer = SingleTokenizer()
            model = gloveModelFactory( 'medical-retrained' )
            return QueryAnalyzerWithPretrained( preprocessor, tokenizer, model )

        case 'naive-jina':
            preprocessor = NaivePreprocessor()
            tokenizer = SingleTokenizer()
            model = SentenceTransformer( "jinaai/jina-embeddings-v2-base-en", trust_remote_code=True ) #, local_files_only=True
            model.max_seq_length = 1024 # control your input sequence length up to 8192
            return QueryAnalyzerWithPretrained( preprocessor, tokenizer, model )

        case 'dummy-jina':
            preprocessor = DummyPreprocessor()
            tokenizer = SingleTokenizer()
            model = SentenceTransformer( "jinaai/jina-embeddings-v2-base-en", trust_remote_code=True ) #, local_files_only=True
            model.max_seq_length = 1024 # control your input sequence length up to 8192
            return QueryAnalyzerWithPretrained( preprocessor, tokenizer, model )

        case 'naive-bert':
            preprocessor = NaivePreprocessor()
            tokenizer = SingleTokenizer()
            model = SentenceTransformer( "all-MiniLM-L6-v2", trust_remote_code=True ) #, local_files_only=True
            return QueryAnalyzerWithPretrained( preprocessor, tokenizer, model )

        case 'dummy-bert':
            preprocessor = DummyPreprocessor()
            tokenizer = SingleTokenizer()
            model = SentenceTransformer( 'all-MiniLM-L6-v2', trust_remote_code=True ) #, local_files_only=True
            return QueryAnalyzerWithPretrained( preprocessor, tokenizer, model )

        case 'dummy-bert-retrained':
            preprocessor = DummyPreprocessor()
            tokenizer = SingleTokenizer()
            from .datasets.medical.settings import pickle_paths
            model_folder = f"{pickle_paths[ 'corpus_repr' ]}/bert-retrained"
            model = SentenceTransformer( model_folder, trust_remote_code=True ) #, local_files_only=True
            return QueryAnalyzerWithPretrained( preprocessor, tokenizer, model )

        case _:
            raise Exception( 'queryAnalyzerFactory(): No valid option.' )


# +----------------------------------------+
# | For development and debugging purposes |
# +----------------------------------------+

# RUN: python -m src.QueryAnalyzer [option]
if __name__ == "__main__": 

    option = None
    if len( sys.argv ) >= 2:
        option = sys.argv[ 1 ]

    query = "Is there any available literature about databases (both SQL and NoSQL), especially somehow relevant to semantics?"

    match option:

        case 'arxiv-stemm-single-count':
            analyzer = queryAnalyzerFactory( option )
            print( analyzer.analyze( query ) )

        case 'arxiv-lemm-single-tfidf':
            analyzer = queryAnalyzerFactory( option )
            print( analyzer.analyze( query ) )

        case 'medical-lemm-single-tfidf':
            analyzer = queryAnalyzerFactory( option )
            print( analyzer.analyze( query ) )

        case 'lemm-single-jina':
            analyzer = queryAnalyzerFactory( option )
            print( analyzer.analyze( query ) )

        case _:
            raise Exception( 'No valid option.' )

