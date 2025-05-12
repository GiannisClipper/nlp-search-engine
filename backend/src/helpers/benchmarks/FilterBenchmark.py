import sys
from abc import ABC, abstractmethod
from src.QueryAnalyzer import AbstractQueryAnalyzer, queryAnalyzerFactory
from src.TermsFilter import AbstractTermsFilter, OccuredTermsFilter, WeightedTermsFilter
from src.TermsFilter import ClusteredTermsFilter, B25TermsFilter, FaissTermsFilter
from src.datasets.medical.Dataset import Dataset, Queries, QueriesResults
from src.datasets.medical.settings import pickle_paths
from src.helpers.Pickle import PickleLoader

class AbstractFilterBenchmark( ABC ):

    def __init__( self, queryAnalyzer:AbstractQueryAnalyzer, filter:AbstractTermsFilter ):
        self._queryAnalyzer = queryAnalyzer
        self._filter = filter

    @abstractmethod
    def conduct( self ):
        pass

class DocFilterBenchmark( AbstractFilterBenchmark ):

    def __init__( self, queryAnalyzer:AbstractQueryAnalyzer, filter:AbstractTermsFilter ):
        super().__init__( queryAnalyzer, filter  )
        self._queries = Queries().toList()
        self._queries_results = QueriesResults().toDict()
        self._corpus = Dataset().toList()

    def _convert_to_doc_ids( self, y_hat:list[str] ):
        y_hat = [ self._corpus[ int(idoc) ][ 'id' ] for idoc in y_hat ]
        return y_hat

    def conduct( self ):

        all_positives = []
        all_responses = []
        true_positives = []
        for i in range( 30 ):
            query = self._queries[ i ]
            y = self._queries_results[ query[ 'id' ] ]
            query_analyzed = self._queryAnalyzer.analyze(query[ 'query' ] ) 
            y_hat = self._filter.filter( query_analyzed )
            y_hat = self._convert_to_doc_ids( y_hat )
            all_responses.append( len( y_hat ) )
            true_positives.append( len( set( y ) & set( y_hat ) ) )
            all_positives.append( len( y ) )

        return {
            'all responses': sum( all_responses ),
            'true positives': sum( true_positives ),
            'all positives': sum( all_positives ),
        }


class SentenceFilterBenchmark( DocFilterBenchmark ):

    def __init__( self, queryAnalyzer:AbstractQueryAnalyzer, filter:AbstractTermsFilter ):
        super().__init__( queryAnalyzer, filter  )
        self._sentences, self._tags = Dataset().toSentences()

    def _convert_to_doc_ids( self, y_hat:list[str] ):
        y_hat = [ self._tags[ int(isent) ].split('.')[0] for isent in y_hat ]
        y_hat = list( set( [ self._corpus[ int(idoc) ][ 'id' ] for idoc in y_hat ] ) )
        return y_hat


# RUN: python -m src.helpers.FilterBenchmark [option]
if __name__ == "__main__": 

    option = ""
    if len( sys.argv ) >= 2:
        option = sys.argv[ 1 ]

    match option:

        case 'occured-index':
            queryAnalyzer = queryAnalyzerFactory( 'medical-lemm-single-tfidf' )
            index_descr = 'title-summary_lower-punct-specials-stops-lemm_single'
            index_filename = f"{pickle_paths[ 'indexes' ]}/{index_descr}.pkl"
            index = PickleLoader( index_filename ).load()
            filter = OccuredTermsFilter( index=index )
            benchmark = DocFilterBenchmark( queryAnalyzer, filter )
            print( benchmark.conduct() )

        case 'weighted-index':
            queryAnalyzer = queryAnalyzerFactory( 'medical-lemm-single-tfidf' )
            index_descr = 'title-summary_lower-punct-specials-stops-lemm_single'
            index_filename = f"{pickle_paths[ 'indexes' ]}/{index_descr}.pkl"
            index = PickleLoader( index_filename ).load()
            filter = WeightedTermsFilter( index=index, corpus=Dataset().toList() )
            benchmark = DocFilterBenchmark( queryAnalyzer, filter )
            print( benchmark.conduct() )

        case 'jina-kmeans':
            queryAnalyzer = queryAnalyzerFactory( 'naive-jina' )
            clusters_descr = 'sentences-jina-kmeans'
            clusters_filename = f"{pickle_paths[ 'clusters' ]}/{clusters_descr}.pkl"
            clustering_model = PickleLoader( clusters_filename ).load()
            index_descr = 'title-summary_lower-punct-specials-stops-lemm_single'
            index_filename = f"{pickle_paths[ 'indexes' ]}/{index_descr}.pkl"
            index = PickleLoader( index_filename ).load()
            filter = ClusteredTermsFilter( model=clustering_model )
            benchmark = SentenceFilterBenchmark( queryAnalyzer, filter )
            print( benchmark.conduct() )

        case 'b25':
            queryAnalyzer = queryAnalyzerFactory( 'naive-jina' )
            sentences, tags = Dataset().toSentences()
            filter = B25TermsFilter( corpus=sentences )
            benchmark = SentenceFilterBenchmark( queryAnalyzer, filter )
            print( benchmark.conduct() )

        case 'bert-feiss':
            queryAnalyzer = queryAnalyzerFactory( 'naive-bert' )
            descr = 'sentences-bert'
            filename = f"{pickle_paths[ 'corpus_repr' ]}/{descr}.pkl"
            embeddings = PickleLoader( filename ).load()
            filter = FaissTermsFilter( corpus_embeddings=embeddings )
            benchmark = SentenceFilterBenchmark( queryAnalyzer, filter )
            print( benchmark.conduct() )

        case _:
            raise Exception( 'No valid option.' )


