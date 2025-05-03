import sys
from abc import ABC, abstractmethod
from typing import cast
from scipy.sparse import spmatrix

from .QueryAnalyzer import AbstractQueryAnalyzer, queryAnalyzerFactory
from .DocFilter import AbstractDocFilter, docFilterFactory
from .SimilarityEstimator import AbstractSimilarityEstimator, similarityEstimatorFactory
from .helpers.typing import QueryAnalyzedType

class AbstractSearchEngine( ABC ):

    @abstractmethod
    def search( self, query:str ):
        pass

class TermsSearchEngine( AbstractSearchEngine ):

    def __init__( self, queryAnalyzer:AbstractQueryAnalyzer, docFilter:AbstractDocFilter, similarityEstimator:AbstractSimilarityEstimator ):
        self._queryAnalyzer = queryAnalyzer
        self._docFilter = docFilter
        self._similarityEstimator = similarityEstimator

    def search( self, query:str ) -> list[tuple[str,float]]:

        # Analyze query into terms and vectors
        print( 'Analyze query...' )
        query_analyzed:QueryAnalyzedType = self._queryAnalyzer.analyze( query )

        # Filter documents based on terms, names, period
        print( 'Filter docs...' )
        filtered_docs = self._docFilter.filter( query_analyzed )
        if len( filtered_docs ) == 0:
            return []
        
        # Estimate similarities between query and document/sentence vectors
        print( 'Estimate similarities...' )
        query_repr = cast( spmatrix, query_analyzed[ 'repr' ] )
        return self._similarityEstimator.estimate( query_repr, filtered_docs )

class TermsNamesPeriodSearchEngine( AbstractSearchEngine ):

    def __init__( self, queryAnalyzer:AbstractQueryAnalyzer, docFilter:AbstractDocFilter, similarityEstimator:AbstractSimilarityEstimator ):
        self._queryAnalyzer = queryAnalyzer
        self._docFilter = docFilter
        self._similarityEstimator = similarityEstimator

    def search( self, query:str, names:list[str]|None=None, period:str|None=None ) -> list[tuple[str,float]]:

        # Analyze query into terms and vectors
        print( 'Analyze query...' )
        query_analyzed:QueryAnalyzedType = self._queryAnalyzer.analyze( query )

        # Filter documents based on terms, names, period
        print( 'Filter docs...' )
        filtered_docs = self._docFilter.filter( query_analyzed, names, period )
        if len( filtered_docs ) == 0:
            return []
        
        # Estimate similarities between query and document/sentence vectors
        print( 'Estimate similarities...' )
        query_repr = cast( spmatrix, query_analyzed[ 'repr' ] )
        return self._similarityEstimator.estimate( query_repr, filtered_docs )


def searchEngineFactory( option:str ):

    match option:

        case 'arxiv-lemm-single-tfidf':
            queryAnalyzer = queryAnalyzerFactory( option )
            docFilter = docFilterFactory( 'arxiv-lemm-single' )
            similarityEstimator = similarityEstimatorFactory( option )
            return TermsNamesPeriodSearchEngine( queryAnalyzer, docFilter, similarityEstimator )

        case 'medical-lemm-single-tfidf':
            queryAnalyzer = queryAnalyzerFactory( option )
            docFilter = docFilterFactory( 'medical-lemm-single' )
            similarityEstimator = similarityEstimatorFactory( option )
            return TermsSearchEngine( queryAnalyzer, docFilter, similarityEstimator )

        case 'arxiv-lemm-single-jina':
            queryAnalyzer = queryAnalyzerFactory( 'lemm-single-jina' )
            docFilter = docFilterFactory( 'arxiv-lemm-single' )
            similarityEstimator = similarityEstimatorFactory( 'arxiv-jina' )
            return TermsNamesPeriodSearchEngine( queryAnalyzer, docFilter, similarityEstimator )

        case 'arxiv-sentences-jina-kmeans':
            queryAnalyzer = queryAnalyzerFactory( 'lemm-single-jina' )
            docFilter = docFilterFactory( 'arxiv-sentences-jina-kmeans' )
            similarityEstimator = similarityEstimatorFactory( 'arxiv-jina' )
            return TermsNamesPeriodSearchEngine( queryAnalyzer, docFilter, similarityEstimator )

        case _:
            raise Exception( 'searchEngineFactory(): No valid option.' )


# RUN: python -m src.SearchEngine
if __name__ == "__main__": 

    option = None
    if len( sys.argv ) >= 2:
        option = sys.argv[ 1 ]

    query = "Are there any available papers about database management systems (both SQL and NoSQL), especially somehow relevant to semantics?"
    query = "Are there any available papers about database management systems (both SQL and NoSQL)?"
    if len( sys.argv ) >= 3:
        query = sys.argv[ 2 ]

    match option:
        case 'arxiv-lemm-single-tfidf':
            engine = searchEngineFactory( option )
            results = engine.search( query )

            from .datasets.arXiv.Dataset import Dataset
            corpus = Dataset().toList()
            for res in results[:10]:
                doc = corpus[ int(res[0]) ]
                print( f"{doc['id']} {doc['catg_ids']} {res[1]}" )

        case 'arxiv-lemm-single-jina':
            engine = searchEngineFactory( option )
            results = engine.search( query )

            from .datasets.arXiv.Dataset import Dataset
            corpus = Dataset().toList()
            for res in results[:10]:
                doc = corpus[ int(res[0]) ]
                print( f"{doc['id']} {doc['catg_ids']} {res[1]}" )

        case 'arxiv-sentences-jina-kmeans':
            engine = searchEngineFactory( option )
            results = engine.search( query )

            from .datasets.arXiv.Dataset import Dataset
            corpus = Dataset().toList()
            for res in results[:10]:
                doc = corpus[ int(res[0]) ]
                print( f"{doc['id']} {doc['catg_ids']} {res[1]}" )


        case _:
            raise Exception( 'No valid option.' )


