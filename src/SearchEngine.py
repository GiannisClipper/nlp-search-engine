from abc import ABC, abstractmethod

from .QueryAnalyzer import QueryAnalyzer, queryAnalyzerFactory
from .DocFilter import DocFilter, docFilterFactory
from .DocEstimator import DocEstimator, docEstimatorFactory

class SearchEngine( ABC ):

    def __init__( self, queryAnalyzer:QueryAnalyzer, docFilter:DocFilter, docEstimator:DocEstimator ):
        self._queryAnalyzer = queryAnalyzer
        self._docFilter = docFilter
        self._docEstimator = docEstimator

    @abstractmethod
    def search( self, query:str, names:list[str]|None=None, period:str|None=None ) -> list[tuple[str,float]]:
        pass


class ArxivTfidfSearchEngine( SearchEngine ):

    def __init__( self ):
        option = 'arxiv-lemm-single-tfidf'
        queryAnalyzer = queryAnalyzerFactory( option )
        docFilter = docFilterFactory( option )
        docEstimator = docEstimatorFactory( option )
        super().__init__( queryAnalyzer, docFilter, docEstimator )

    def search( self, query:str, names:list[str]|None=None, period:str|None=None ) -> list[tuple[str,float]]:

        # Analyze query into terms and vectors
        print( 'Analyze query...' )
        query_terms, query_repr = self._queryAnalyzer.analyze( query )

        # Filter documents based on terms, names, period
        print( 'Filter docs...' )
        filtered_docs = self._docFilter.filter( query_terms, names, period )
        if len( filtered_docs ) == 0:
            return []
        
        # Estimate similarities between query and document vectors
        print( 'Estimate similarities...' )
        return self._docEstimator.estimate( query_repr, filtered_docs )


# RUN: python -m src.SearchEngine
if __name__ == "__main__": 

    engine = ArxivTfidfSearchEngine()
    query = "Is there any available literature about databases (both SQL and NoSQL), especially somehow relevant to semantics?"
    # query = "Can you fetch some good texts about networks and communications?"
    results = engine.search( query )

    from .arXiv.Dataset import Dataset
    corpus = Dataset().toList()

    for res in results:
        doc = corpus[ int(res[0]) ]
        print( f"{doc['id']} {doc['catg_ids']} {res[1]}" )
