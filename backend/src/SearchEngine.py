import sys
from abc import ABC, abstractmethod
from typing import cast
from scipy.sparse import spmatrix

from .QueryAnalyzer import AbstractQueryAnalyzer, queryAnalyzerFactory
from .Retriever import AbstractRetriever, retrieverFactory
from .Ranker import AbstractRanker, rankerFactory
from .Summarizer import summarizerFactory, AbstractSummarizer, NaiveSummarizer
from .helpers.typing import QueryAnalyzedType

# ------------------------------ #
# Code to compose search engines #
#------------------------------- #

class AbstractSearchEngine( ABC ):

    def __init__( 
        self, 
        queryAnalyzer:AbstractQueryAnalyzer, 
        retriever:AbstractRetriever, 
        ranker:AbstractRanker,
        summarizer:AbstractSummarizer,
        threshold:float=0.0
    ):
        self._queryAnalyzer = queryAnalyzer
        self._query_analyzed:QueryAnalyzedType

        self._retriever = retriever
        self._retrieved:list[str]

        self._ranker = ranker
        self._ranked = list[tuple[str,float]]

        self._summarizer = summarizer
        self._summarized = list[tuple[str,float,dict]]

        self._threshold = threshold

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
    def _summarize( self ) -> None:
        pass

    @abstractmethod
    def search( self, query:str ) -> list[tuple[str,float,dict]]:
        pass

class TermsSearchEngine( AbstractSearchEngine ):

    def _analyze( self, query:str ) -> None:

        # Analyze query into terms and vectors/embeddings
        print( 'Analyze query...' )
        self._query_analyzed:QueryAnalyzedType = self._queryAnalyzer.analyze( query )

    def _retrieve( self ) -> None:

        # Retrieve documents or sentences based on query
        print( 'Retrieve docs/sentences...' )
        self._retrieved = self._retriever.retrieve( query_analyzed=self._query_analyzed )

    def _rank( self ) -> None:
        
        # Rank based on similarity between query and document/sentence vectors
        print( f'Rank documents ({len(self._retrieved)})...' )
        query_repr = cast( spmatrix, self._query_analyzed[ 'repr' ] )
        self._ranked = self._ranker.rank( query_repr, self._retrieved )

    def _summarize( self ) -> None:

        self._summarized = []
        for idoc, rank in self._ranked:
            query_repr = cast( spmatrix, self._query_analyzed[ 'repr' ] )
            summarized = self._summarizer.summarize( int(idoc), query_repr )
            self._summarized.append( ( idoc, rank, summarized ) )

    def search( self, query:str ) -> list[tuple[str,float,dict]]:

        self._analyze( query )
        self._retrieve()
        if len( self._retrieved ) == 0:
            return []
        self._rank()

        # keep results regarding threshold
        self._ranked = [ r for r in self._ranked if r[1] >= self._threshold ]
        self._summarize()

        return self._summarized

class PeriodNamesTermsSearchEngine( TermsSearchEngine ):

    def _retrieve( self, names:list[str]|None=None, period:str|None=None ) -> None:

        # Retrieve documents or sentences based on query
        print( 'Retrieve docs/sentences...' )
        self._retrieved = self._retriever.retrieve( period=period, names=names, query_analyzed=self._query_analyzed )

    def search( self, query:str, names:list[str]|None=None, period:str|None=None ) -> list[tuple[str,float,dict]]:

        self._analyze( query )
        self._retrieve( period=period, names=names )
        if len( self._retrieved ) == 0:
            return []
        self._rank()

        # keep results regarding threshold
        self._ranked = [ r for r in self._ranked if r[1] >= self._threshold ]
        self._summarize()

        return self._summarized


def searchEngineFactory( option:str ):

    VEC_THRESHOLD = 0.25 # for tf, tfidf vectors
    EMB_THRESHOLD = 0.45 # for embeddings

    match option:

        case 'arxiv-stemm-single-count':
            queryAnalyzer = queryAnalyzerFactory( option )
            retriever = retrieverFactory( 'arxiv-stemm-single' )
            ranker = rankerFactory( option )
            summarizer = summarizerFactory( 'arxiv-naive' )
            return PeriodNamesTermsSearchEngine( queryAnalyzer, retriever, ranker, summarizer, threshold=VEC_THRESHOLD )

        case 'arxiv-lemm-single-tfidf':
            queryAnalyzer = queryAnalyzerFactory( option )
            retriever = retrieverFactory( 'arxiv-lemm-single' )
            ranker = rankerFactory( option )
            summarizer = summarizerFactory( 'arxiv-naive' )
            return PeriodNamesTermsSearchEngine( queryAnalyzer, retriever, ranker, summarizer, threshold=VEC_THRESHOLD )

        case 'arxiv-lemm-2gram-tfidf':
            queryAnalyzer = queryAnalyzerFactory( option )
            retriever = retrieverFactory( 'arxiv-lemm-2gram' )
            ranker = rankerFactory( option )
            summarizer = summarizerFactory( 'arxiv-naive' )
            return PeriodNamesTermsSearchEngine( queryAnalyzer, retriever, ranker, summarizer, threshold=VEC_THRESHOLD )

        case 'arxiv-sentences-glove-bm25':
            queryAnalyzer = queryAnalyzerFactory( 'arxiv-naive-glove' )
            retriever = retrieverFactory( 'arxiv-sentences-bm25' )
            ranker = rankerFactory( 'arxiv-glove' )
            summarizer = summarizerFactory( 'arxiv-naive' )
            return PeriodNamesTermsSearchEngine( queryAnalyzer, retriever, ranker, summarizer, threshold=EMB_THRESHOLD )

        case 'arxiv-sentences-glove-retrained-bm25':
            queryAnalyzer = queryAnalyzerFactory( 'arxiv-naive-glove-retrained' )
            retriever = retrieverFactory( 'arxiv-sentences-bm25' )
            ranker = rankerFactory( 'arxiv-glove-retrained' )
            summarizer = summarizerFactory( 'arxiv-glove-retrained-similarity' )
            return PeriodNamesTermsSearchEngine( queryAnalyzer, retriever, ranker, summarizer, threshold=EMB_THRESHOLD )

        case 'arxiv-sentences-jina-bm25':
            queryAnalyzer = queryAnalyzerFactory( 'naive-jina' )
            retriever = retrieverFactory( 'arxiv-sentences-bm25' )
            ranker = rankerFactory( 'arxiv-jina' )
            summarizer = summarizerFactory( 'arxiv-jina-similarity' )
            return PeriodNamesTermsSearchEngine( queryAnalyzer, retriever, ranker, summarizer, threshold=EMB_THRESHOLD )

        case 'arxiv-sentences-jina-kmeans':
            queryAnalyzer = queryAnalyzerFactory( 'naive-jina' )
            retriever = retrieverFactory( 'arxiv-sentences-jina-kmeans' )
            ranker = rankerFactory( 'arxiv-jina' )
            summarizer = summarizerFactory( 'arxiv-jina-similarity' )
            return PeriodNamesTermsSearchEngine( queryAnalyzer, retriever, ranker, summarizer, threshold=EMB_THRESHOLD )

        case 'arxiv-sentences-jina-faiss':
            queryAnalyzer = queryAnalyzerFactory( 'dummy-jina' )
            retriever = retrieverFactory( 'arxiv-sentences-jina-faiss' )
            ranker = rankerFactory( 'arxiv-jina' )
            summarizer = summarizerFactory( 'arxiv-jina-similarity' )
            return PeriodNamesTermsSearchEngine( queryAnalyzer, retriever, ranker, summarizer, threshold=EMB_THRESHOLD )

        case 'arxiv-sentences-bert-faiss':
            queryAnalyzer = queryAnalyzerFactory( 'dummy-bert' )
            retriever = retrieverFactory( 'arxiv-sentences-bert-faiss' )
            ranker = rankerFactory( 'arxiv-bert' )
            summarizer = summarizerFactory( 'arxiv-bert-similarity' )
            return PeriodNamesTermsSearchEngine( queryAnalyzer, retriever, ranker, summarizer, threshold=EMB_THRESHOLD )

        ###########
        # medical #
        ###########

        case 'medical-stemm-single-count':
            queryAnalyzer = queryAnalyzerFactory( option )
            retriever = retrieverFactory( 'medical-stemm-single' )
            ranker = rankerFactory( option )
            summarizer = summarizerFactory( 'medical-naive' )
            return TermsSearchEngine( queryAnalyzer, retriever, ranker, summarizer )

        case 'medical-lemm-single-tfidf':
            queryAnalyzer = queryAnalyzerFactory( option )
            retriever = retrieverFactory( 'medical-lemm-single' )
            ranker = rankerFactory( option )
            summarizer = summarizerFactory( 'medical-naive' )
            return TermsSearchEngine( queryAnalyzer, retriever, ranker, summarizer )

        case 'medical-lemm-2gram-tfidf':
            queryAnalyzer = queryAnalyzerFactory( option )
            retriever = retrieverFactory( 'medical-lemm-2gram' )
            ranker = rankerFactory( option )
            summarizer = summarizerFactory( 'medical-naive' )
            return TermsSearchEngine( queryAnalyzer, retriever, ranker, summarizer )

        case 'medical-sentences-glove-bm25':
            queryAnalyzer = queryAnalyzerFactory( 'medical-naive-glove' )
            retriever = retrieverFactory( 'medical-sentences-bm25' )
            ranker = rankerFactory( 'medical-glove' )
            summarizer = summarizerFactory( 'medical-naive' )
            return TermsSearchEngine( queryAnalyzer, retriever, ranker, summarizer )

        case 'medical-sentences-glove-retrained-bm25':
            queryAnalyzer = queryAnalyzerFactory( 'medical-naive-glove-retrained' )
            retriever = retrieverFactory( 'medical-sentences-bm25' )
            ranker = rankerFactory( 'medical-glove-retrained' )
            summarizer = summarizerFactory( 'medical-naive' )
            return TermsSearchEngine( queryAnalyzer, retriever, ranker, summarizer )

        # case 'medical-sentences-glove-faiss':
        #     queryAnalyzer = queryAnalyzerFactory( 'medical-naive-glove' )
        #     retriever = retrieverFactory( 'medical-sentences-glove-faiss' )
        #     ranker = rankerFactory( 'medical-glove' )
        #     summarizer = summarizerFactory( 'medical-naive' )
        #     return TermsSearchEngine( queryAnalyzer, retriever, ranker, summarizer )

        # case 'medical-sentences-glove-retrained-faiss':
        #     queryAnalyzer = queryAnalyzerFactory( 'medical-naive-glove-retrained' )
        #     retriever = retrieverFactory( 'medical-sentences-glove-retrained-faiss' )
        #     ranker = rankerFactory( 'medical-glove-retrained' )
        #     summarizer = summarizerFactory( 'medical-naive' )
        #     return TermsSearchEngine( queryAnalyzer, retriever, ranker, summarizer )

        case 'medical-sentences-jina-bm25':
            queryAnalyzer = queryAnalyzerFactory( 'naive-jina' )
            retriever = retrieverFactory( 'medical-sentences-bm25' )
            ranker = rankerFactory( 'medical-jina' )
            summarizer = summarizerFactory( 'medical-naive' )
            return TermsSearchEngine( queryAnalyzer, retriever, ranker, summarizer )

        case 'medical-sentences-jina-kmeans':
            queryAnalyzer = queryAnalyzerFactory( 'naive-jina' )
            retriever = retrieverFactory( 'medical-sentences-jina-kmeans' )
            ranker = rankerFactory( 'medical-jina' )
            summarizer = summarizerFactory( 'medical-naive' )
            return TermsSearchEngine( queryAnalyzer, retriever, ranker, summarizer )

        case 'medical-sentences-jina-faiss':
            queryAnalyzer = queryAnalyzerFactory( 'dummy-jina' )
            retriever = retrieverFactory( 'medical-sentences-jina-faiss' )
            ranker = rankerFactory( 'medical-jina' )
            summarizer = summarizerFactory( 'medical-naive' )
            return TermsSearchEngine( queryAnalyzer, retriever, ranker, summarizer )

        case 'medical-sentences-bert-faiss':
            queryAnalyzer = queryAnalyzerFactory( 'dummy-bert' )
            retriever = retrieverFactory( 'medical-sentences-bert-faiss' )
            ranker = rankerFactory( 'medical-bert' )
            summarizer = summarizerFactory( 'medical-naive' )
            return TermsSearchEngine( queryAnalyzer, retriever, ranker,summarizer )

        case 'medical-sentences-bert-retrained-faiss':
            queryAnalyzer = queryAnalyzerFactory( 'dummy-bert-retrained' )
            retriever = retrieverFactory( 'medical-sentences-bert-retrained-faiss' )
            ranker = rankerFactory( 'medical-bert-retrained' )
            summarizer = summarizerFactory( 'medical-naive' )
            return TermsSearchEngine( queryAnalyzer, retriever, ranker, summarizer )

        case _:
            raise Exception( 'searchEngineFactory(): No valid option.', option )


# +----------------------------------------+
# | For development and debugging purposes |
# +----------------------------------------+

# RUN: python -m src.SearchEngine
if __name__ == "__main__": 

    option = None
    if len( sys.argv ) >= 2:
        option = sys.argv[ 1 ]

    query = "Are there any available papers about database management systems (both SQL and NoSQL), especially somehow relevant to semantics?"
    if len( sys.argv ) >= 3:
        query = sys.argv[ 2 ]

    match option:

        case 'arxiv-stemm-single-count' |\
             'arxiv-lemm-single-tfidf' |\
             'arxiv-lemm-2gram-tfidf' |\
             'arxiv-sentences-glove-bm25' |\
             'arxiv-sentences-glove-retrained-bm25' |\
             'arxiv-sentences-jina-bm25' |\
             'arxiv-sentences-jina-kmeans' |\
             'arxiv-sentences-jina-faiss' |\
             'arxiv-sentences-bert-faiss' |\
             'arxiv-sentences-bert-retrained-faiss':

            engine = searchEngineFactory( option )
            results = engine.search( query )
            for res in results:
                print( res )

            # from .datasets.arXiv.Dataset import Dataset
            # corpus = Dataset().toList()
            # for res in results:
            #     doc = corpus[ int(res[0]) ]
            #     print( f"{doc['id']} {doc['catg_ids']} {res[1]}" )

        case 'medical-stemm-single-count' |\
             'medical-lemm-single-tfidf' |\
             'medical-lemm-2gram-tfidf' |\
             'medical-sentences-glove-bm25' |\
             'medical-sentences-glove-retrained-bm25' |\
             'medical-sentences-jina-bm25' |\
             'medical-sentences-jina-kmeans' |\
             'medical-sentences-jina-faiss' |\
             'medical-sentences-bert-faiss' |\
             'medical-sentences-bert-retrained-faiss':

            engine = searchEngineFactory( option )
            results = engine.search( query )
            for res in results:
                print( res )

            # from .datasets.medical.Dataset import Dataset
            # corpus = Dataset().toList()
            # for res in results[:10]:
            #     doc = corpus[ int(res[0]) ]
            #     print( f"{doc['id']} {res[1]}" )

        case _:
            raise Exception( 'No valid option.' )


