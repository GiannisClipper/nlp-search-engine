from abc import ABC, abstractmethod

import sys

import nltk

nltk.download( 'punkt_tab' ) # required by word_tokenize()
from nltk.tokenize import word_tokenize

import numpy as np
from scipy.sparse import spmatrix, csr_matrix
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sentence_transformers import SentenceTransformer
from .Preprocessor import Preprocessor, NaivePreprocessor, LemmPreprocessor, StemmPreprocessor
from .helpers.Pickle import PickleLoader
from .helpers.typing import QueryAnalyzedType
from .models.GloveModel import GloveModel, gloveModelFactory

class AbstractQueryAnalyzer( ABC ):

    @abstractmethod
    def analyze( self, query:str ) -> QueryAnalyzedType:
        pass


class QueryAnalyzerWithVectorizer( AbstractQueryAnalyzer):

    def __init__( self, preprocessor:Preprocessor, vectorizer:CountVectorizer|TfidfVectorizer ):
        super().__init__()
        self._preprocessor = preprocessor
        self._vectorizer = vectorizer

    def analyze( self, query:str ) -> QueryAnalyzedType:
        query_preprocessed = self._preprocessor.transform( [ query ] )
        query_terms = list( word_tokenize( query_preprocessed[ 0 ] ) )
        query_repr = self._vectorizer.transform( query_preprocessed )
        return { 'query': query, 'terms': query_terms, 'repr': query_repr }


class QueryAnalyzerWithPretrained( AbstractQueryAnalyzer):

    def __init__( self, preprocessor:Preprocessor, model:SentenceTransformer|GloveModel ):
        super().__init__()
        self._preprocessor = preprocessor
        self._model = model

    def analyze( self, query:str ) -> QueryAnalyzedType:
        query_preprocessed = self._preprocessor.transform( [ query ] )
        query_terms = list( word_tokenize( query_preprocessed[ 0 ] ) )
        query_repr = self._model.encode( query )
        return { 'query': query, 'terms': query_terms, 'repr': csr_matrix( query_repr ) } # convert to spmatrix


def queryAnalyzerFactory( option:str ) -> AbstractQueryAnalyzer:

    match option:

        case 'arxiv-stemm-single-count':
            from .datasets.arXiv.settings import pickle_paths
            preprocessor = StemmPreprocessor()
            vectorizer_descr = 'title-summary_lower-punct-specials-stops-stemm_single_count'
            vectorizer_filename = f"{pickle_paths[ 'vectorizers' ]}/{vectorizer_descr}.pkl"
            vectorizer = PickleLoader( vectorizer_filename ).load()
            return QueryAnalyzerWithVectorizer( preprocessor, vectorizer )

        case 'arxiv-lemm-single-tfidf':
            from .datasets.arXiv.settings import pickle_paths
            preprocessor = LemmPreprocessor()
            vectorizer_descr = 'title-summary_lower-punct-specials-stops-lemm_single_tfidf'
            vectorizer_filename = f"{pickle_paths[ 'vectorizers' ]}/{vectorizer_descr}.pkl"
            vectorizer = PickleLoader( vectorizer_filename ).load()
            return QueryAnalyzerWithVectorizer( preprocessor, vectorizer )

        case 'arxiv-lemm-2gram-tfidf':
            from .datasets.arXiv.settings import pickle_paths
            preprocessor = LemmPreprocessor()
            vectorizer_descr = 'title-summary_lower-punct-specials-stops-lemm_2gram_tfidf'
            vectorizer_filename = f"{pickle_paths[ 'vectorizers' ]}/{vectorizer_descr}.pkl"
            vectorizer = PickleLoader( vectorizer_filename ).load()
            return QueryAnalyzerWithVectorizer( preprocessor, vectorizer )

        case 'arxiv-naive-glove':
            preprocessor = NaivePreprocessor()
            model = gloveModelFactory( 'arxiv' )
            return QueryAnalyzerWithPretrained( preprocessor, model )

        case 'medical-lemm-single-tfidf':
            from .datasets.medical.settings import pickle_paths
            preprocessor = LemmPreprocessor()
            vectorizer_descr = 'title-summary_lower-punct-specials-stops-lemm_single_tfidf'
            vectorizer_filename = f"{pickle_paths[ 'vectorizers' ]}/{vectorizer_descr}.pkl"
            vectorizer = PickleLoader( vectorizer_filename ).load()
            return QueryAnalyzerWithVectorizer( preprocessor, vectorizer )

        case 'medical-lemm-2gram-tfidf':
            from .datasets.medical.settings import pickle_paths
            preprocessor = LemmPreprocessor()
            vectorizer_descr = 'title-summary_lower-punct-specials-stops-lemm_2gram_tfidf'
            vectorizer_filename = f"{pickle_paths[ 'vectorizers' ]}/{vectorizer_descr}.pkl"
            vectorizer = PickleLoader( vectorizer_filename ).load()
            return QueryAnalyzerWithVectorizer( preprocessor, vectorizer )

        case 'medical-lemm-single-jina':
            from .datasets.medical.settings import pickle_paths
            preprocessor = LemmPreprocessor()
            model = SentenceTransformer( "jinaai/jina-embeddings-v2-base-en", trust_remote_code=True, local_files_only=True )
            model.max_seq_length = 1024 # control your input sequence length up to 8192
            return QueryAnalyzerWithPretrained( preprocessor, model )

        case 'medical-naive-glove':
            preprocessor = NaivePreprocessor()
            model = gloveModelFactory( 'medical' )
            return QueryAnalyzerWithPretrained( preprocessor, model )

        case 'naive-bert':
            preprocessor = NaivePreprocessor()
            model = SentenceTransformer( 'all-MiniLM-L6-v2', trust_remote_code=True, local_files_only=True )
            return QueryAnalyzerWithPretrained( preprocessor, model )

        case 'naive-jina':
            preprocessor = NaivePreprocessor()
            model = SentenceTransformer( "jinaai/jina-embeddings-v2-base-en", trust_remote_code=True, local_files_only=True )
            model.max_seq_length = 1024 # control your input sequence length up to 8192
            return QueryAnalyzerWithPretrained( preprocessor, model )

        case _:
            raise Exception( 'queryAnalyzerFactory(): No valid option.' )


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

