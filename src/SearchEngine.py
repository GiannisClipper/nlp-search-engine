import sys
from abc import ABC, abstractmethod
from typing import cast
from scipy.sparse import spmatrix

from .QueryAnalyzer import AbstractQueryAnalyzer, queryAnalyzerFactory
from .Retriever import AbstractRetriever, retrieverFactory
from .Ranker import AbstractRanker, rankerFactory
from .helpers.typing import QueryAnalyzedType

class AbstractSearchEngine( ABC ):

    def __init__( self, queryAnalyzer:AbstractQueryAnalyzer, retriever:AbstractRetriever, ranker:AbstractRanker ):
        self._queryAnalyzer = queryAnalyzer
        self._query_analyzed:QueryAnalyzedType

        self._retriever = retriever
        self._retrieved:list[str]

        self._ranker = ranker
        self._ranked = list[tuple[str,float]]

    @abstractmethod
    def _analyze( self, query:str ) -> None:
        pass

    @abstractmethod
    def _retrieve( self ) -> None:
        pass

    @abstractmethod
    def _rank( self ) -> None:
        pass

    @abstractmethod
    def search( self, query:str ) -> list[tuple[str,float]]:
        pass

class TermsSearchEngine( AbstractSearchEngine ):

    def _analyze( self, query:str ) -> None:

        # Analyze query into terms and vectors
        print( 'Analyze query...' )
        self._query_analyzed:QueryAnalyzedType = self._queryAnalyzer.analyze( query )

    def _retrieve( self ) -> None:

        # Retrieve documents or sentences based on terms
        print( 'Retrieve docs/sentences...' )
        self._retrieved = self._retriever.retrieve( query_analyzed=self._query_analyzed )

    def _rank( self ) -> None:
        
        # Rank based on similarity between query and document/sentence vectors
        print( f'Rank documents ({len(self._retrieved)})...' )
        query_repr = cast( spmatrix, self._query_analyzed[ 'repr' ] )
        self._ranked = self._ranker.rank( query_repr, self._retrieved )

    def search( self, query:str ) -> list[tuple[str,float]]:

        self._analyze( query )
        self._retrieve()
        if len( self._retrieved ) == 0:
            return []
        self._rank()
        return self._ranked


class PeriodNamesTermsSearchEngine( TermsSearchEngine ):

    def _retrieve( self, names:list[str]|None=None, period:str|None=None ) -> None:

        # Retrieve documents or sentences based on terms
        print( 'Retrieve docs/sentences...' )
        self._retrieved = self._retriever.retrieve( period=period, names=names, query_analyzed=self._query_analyzed )

    def _search( self, query:str, names:list[str]|None=None, period:str|None=None ) -> list[tuple[str,float]]:

        self._analyze( query )
        self._retrieve( period=period, names=names )
        if len( self._retrieved ) == 0:
            return []
        self._rank()
        return self._ranked


def searchEngineFactory( option:str ):

    match option:

        case 'arxiv-lemm-single-tfidf':
            queryAnalyzer = queryAnalyzerFactory( option )
            retriever = retrieverFactory( 'arxiv-lemm-single' )
            ranker = rankerFactory( option )
            return PeriodNamesTermsSearchEngine( queryAnalyzer, retriever, ranker )

        case 'medical-lemm-single-tfidf':
            queryAnalyzer = queryAnalyzerFactory( option )
            retriever = retrieverFactory( 'medical-lemm-single' )
            ranker = rankerFactory( option )
            return TermsSearchEngine( queryAnalyzer, retriever, ranker )

        case 'medical-lemm-single-jina':
            queryAnalyzer = queryAnalyzerFactory( option )
            retriever = retrieverFactory( 'medical-lemm-single' )
            ranker = rankerFactory( 'medical-jina' )
            return TermsSearchEngine( queryAnalyzer, retriever, ranker )

        case 'arxiv-sentences-jina-kmeans':
            queryAnalyzer = queryAnalyzerFactory( 'naive-jina' )
            retriever = retrieverFactory( 'arxiv-sentences-jina-kmeans' )
            ranker = rankerFactory( 'arxiv-jina' )
            return PeriodNamesTermsSearchEngine( queryAnalyzer, retriever, ranker )

        case 'medical-sentences-jina-kmeans':
            queryAnalyzer = queryAnalyzerFactory( 'naive-jina' )
            retriever = retrieverFactory( 'medical-sentences-jina-kmeans' )
            ranker = rankerFactory( 'medical-jina' )
            return TermsSearchEngine( queryAnalyzer, retriever, ranker )

        case 'arxiv-sentences-jina-b25':
            queryAnalyzer = queryAnalyzerFactory( 'naive-jina' )
            retriever = retrieverFactory( 'arxiv-sentences-b25' )
            ranker = rankerFactory( 'arxiv-jina' )
            return PeriodNamesTermsSearchEngine( queryAnalyzer, retriever, ranker )

        case 'medical-sentences-jina-b25':
            queryAnalyzer = queryAnalyzerFactory( 'naive-jina' )
            retriever = retrieverFactory( 'medical-sentences-b25' )
            ranker = rankerFactory( 'medical-jina' )
            return TermsSearchEngine( queryAnalyzer, retriever, ranker )

        case 'arxiv-sentences-jina-faiss':
            queryAnalyzer = queryAnalyzerFactory( 'naive-jina' )
            retriever = retrieverFactory( 'arxiv-sentences-jina-faiss' )
            ranker = rankerFactory( 'arxiv-jina' )
            return TermsSearchEngine( queryAnalyzer, retriever, ranker )

        case 'medical-sentences-jina-faiss':
            queryAnalyzer = queryAnalyzerFactory( 'naive-jina' )
            retriever = retrieverFactory( 'medical-sentences-jina-faiss' )
            ranker = rankerFactory( 'medical-jina' )
            return TermsSearchEngine( queryAnalyzer, retriever, ranker )

        case 'arxiv-sentences-bert-faiss':
            queryAnalyzer = queryAnalyzerFactory( 'naive-bert' )
            retriever = retrieverFactory( 'arxiv-sentences-bert-faiss' )
            ranker = rankerFactory( 'arxiv-bert' )
            return TermsSearchEngine( queryAnalyzer, retriever, ranker )

        case 'medical-sentences-bert-faiss':
            queryAnalyzer = queryAnalyzerFactory( 'naive-bert' )
            retriever = retrieverFactory( 'medical-sentences-bert-faiss' )
            ranker = rankerFactory( 'medical-bert' )
            return TermsSearchEngine( queryAnalyzer, retriever, ranker )

        case _:
            raise Exception( 'searchEngineFactory(): No valid option.' )


# RUN: python -m src.SearchEngine
if __name__ == "__main__": 

    option = None
    if len( sys.argv ) >= 2:
        option = sys.argv[ 1 ]

    query = "Are there any available papers about database management systems (both SQL and NoSQL), especially somehow relevant to semantics?"
    if len( sys.argv ) >= 3:
        query = sys.argv[ 2 ]

    match option:

        case 'arxiv-lemm-single-tfidf' |\
             'arxiv-lemm-single-jina' |\
             'arxiv-sentences-jina-kmeans' |\
             'arxiv-sentences-jina-b25' |\
             'arxiv-sentences-jina-faiss' |\
             'arxiv-sentences-bert-faiss':

            engine = searchEngineFactory( option )
            results = engine.search( query )

            from .datasets.arXiv.Dataset import Dataset
            corpus = Dataset().toList()
            for res in results:
                doc = corpus[ int(res[0]) ]
                print( f"{doc['id']} {doc['catg_ids']} {res[1]}" )

        case 'medical-sentences-jina-kmeans' |\
             'medical-sentences-jina-b25':

            engine = searchEngineFactory( option )
            results = engine.search( query )

            from .datasets.medical.Dataset import Dataset
            corpus = Dataset().toList()
            for res in results[:10]:
                doc = corpus[ int(res[0]) ]
                print( f"{doc['id']} {res[1]}" )

        case _:
            raise Exception( 'No valid option.' )


