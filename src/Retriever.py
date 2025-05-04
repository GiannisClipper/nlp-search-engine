import sys
from abc import ABC, abstractmethod
from typing import cast
from .helpers.typing import QueryAnalyzedType

from .TermsFilter import AbstractTermsFilter, IndexedTermsFilter, ClusteredTermsFilter, B25TermsFilter
from .NameFilter import NamesFilter
from .PeriodFilter import PeriodFilter
from .helpers.Pickle import PickleLoader
from .helpers.DocViewer import DocViewer

class AbstractRetriever( ABC ):

    @abstractmethod
    def filter( self, query_analyzed:QueryAnalyzedType|None=None, names:list[str]|None=None, period:str|None=None ) -> list[str]:
        pass

class TermsRetriever( AbstractRetriever ):

    def __init__( self, termsFilter:AbstractTermsFilter ):
        self.termsFilter = termsFilter

    def filter( self, query_analyzed:QueryAnalyzedType ) -> list[str]:

        filtered_docs = self.termsFilter.filter( query_analyzed )
        filtered_docs = [ str(i) for i in filtered_docs ]
        return filtered_docs

class TermsNamesPeriodRetriever( AbstractRetriever ):

    def __init__( self, termsFilter:AbstractTermsFilter, namesFilter:NamesFilter, periodFilter:PeriodFilter ):
        self.termsFilter = termsFilter
        self.namesFilter = namesFilter
        self.periodFilter = periodFilter

    def filter( self, query_analyzed:QueryAnalyzedType|None=None, names:list[str]|None=None, period:str|None=None ) -> list[str]:

        # Filter by period
        period_filtered_docs = self.periodFilter.tags # e.g. tags -> '234', '235', ...
        if period:
            period_filtered_docs = self.periodFilter( period )
         
        if len( period_filtered_docs ) == 0: # No doc match the date filter
            return period_filtered_docs

        # Filter by names
        names_filtered_docs = list( set( [ t.split('.')[0] for t in self.namesFilter.tags ] ) ) # e.g. tags -> '234.0', '234.1', '235.0', ...
        if names:
            names_filtered_docs = list( self.namesFilter( names ) )
 
        if len( names_filtered_docs ) == 0: # No doc match the names filter
            return names_filtered_docs
        
        # Intersect period, names filters
        period_names_filtered_docs = list( set( period_filtered_docs ) & set( names_filtered_docs ) )
 
        if len( period_names_filtered_docs ) == 0: # No doc match both period, names filters
            return period_names_filtered_docs

        # Filter by terms
        if not query_analyzed:
            return period_names_filtered_docs

        terms_filtered_docs = self.termsFilter.filter( query_analyzed )
        terms_filtered_docs = [ str(t) for t in terms_filtered_docs ]

        # Intersect period, names, terms filters
        filtered_docs = list( set( period_names_filtered_docs ) & set( terms_filtered_docs ) )

        return filtered_docs


def retrieverFactory( option:str ) -> AbstractRetriever:

    match option:

        case 'arxiv-lemm-single':
            from .datasets.arXiv.Dataset import Dataset
            from .datasets.arXiv.settings import pickle_paths
            ds = Dataset()

            dates, tags = ds.toPublished()
            periodFilter = PeriodFilter( dates=dates, tags=tags )

            names, tags = ds.toAuthors()
            namesFilter = NamesFilter( names=names, tags=tags )

            index_descr = 'title-summary_lower-punct-specials-stops-lemm_single'
            index_filename = f"{pickle_paths[ 'indexes' ]}/{index_descr}.pkl"
            index = PickleLoader( index_filename ).load()
            corpus = ds.toList()
            termsFilter = IndexedTermsFilter( index=index, corpus=corpus )

            return TermsNamesPeriodRetriever( termsFilter=termsFilter, namesFilter=namesFilter, periodFilter=periodFilter )

        case 'medical-lemm-single':
            from .datasets.medical.Dataset import Dataset
            from .datasets.medical.settings import pickle_paths
            ds = Dataset()

            index_descr = 'title-summary_lower-punct-specials-stops-lemm_single'
            index_filename = f"{pickle_paths[ 'indexes' ]}/{index_descr}.pkl"
            index = PickleLoader( index_filename ).load()
            corpus = ds.toList()
            termsFilter = IndexedTermsFilter( index=index, corpus=corpus )

            return TermsRetriever( termsFilter=termsFilter )

        case 'arxiv-sentences-jina-kmeans':
            from .datasets.arXiv.Dataset import Dataset
            from .datasets.arXiv.settings import pickle_paths
            ds = Dataset()

            dates, tags = ds.toPublished()
            periodFilter = PeriodFilter( dates=dates, tags=tags )

            names, tags = ds.toAuthors()
            namesFilter = NamesFilter( names=names, tags=tags )

            clusters_descr = 'sentences-jina-kmeans'
            clusters_filename = f"{pickle_paths[ 'clusters' ]}/{clusters_descr}.pkl"
            clustering_model = PickleLoader( clusters_filename ).load()
            termsFilter = ClusteredTermsFilter( model=clustering_model )

            return TermsNamesPeriodRetriever( termsFilter=termsFilter, namesFilter=namesFilter, periodFilter=periodFilter )

        case 'medical-sentences-jina-kmeans':
            from .datasets.medical.Dataset import Dataset
            from .datasets.medical.settings import pickle_paths
            ds = Dataset()

            clusters_descr = 'sentences-jina-kmeans'
            clusters_filename = f"{pickle_paths[ 'clusters' ]}/{clusters_descr}.pkl"
            clustering_model = PickleLoader( clusters_filename ).load()
            termsFilter = ClusteredTermsFilter( model=clustering_model )

            return TermsRetriever( termsFilter=termsFilter )

        case 'medical-sentences-b25':
            from .datasets.medical.Dataset import Dataset
            sentences, tags = Dataset().toSentences()
            termsFilter = B25TermsFilter( corpus=sentences )
            return TermsRetriever( termsFilter=termsFilter )

        case _:
            raise Exception( 'retrieverFactory(): No valid option.' )



# RUN: python -m src.Retriever
if __name__ == "__main__": 

    # initialize involved instances

    docFilter = retrieverFactory( 'arxiv-lemm-single' )

    from .datasets.arXiv.Dataset import Dataset
    ds = Dataset()
    corpus = ds.toList()
    docViewer = DocViewer( corpus=corpus )

    # set testing parameters

    params:dict[str,str|None] = {
        'terms': None,
        'names': None,
        'period': None
    }
    for i in range( 1, len( sys.argv ) ):
        key, value = sys.argv[ i ].split( '=' )
        params[ key ] = value

    if len( sys.argv ) == 1:
        params[ 'terms' ] = 'probability,geometric,poisson,distribution'
        params[ 'names' ] = 'michael jordan'
        params[ 'period' ] = '2012-01-01,2012-12-31'

    # perform filterings

    terms = None
    if params[ 'terms' ]:
        terms = params[ 'terms' ].split(',')
        query_analyzed = cast( QueryAnalyzedType, { 'terms': terms } )
        result = docFilter.filter( query_analyzed=query_analyzed )
        print( '-------------------------------------------------------------' )
        print( terms, len( result ), result[:5] )
        for res in result[:5]:
            docViewer.view( int( res ) )

    names = None
    if params[ 'names' ]:
        names = params[ 'names' ].split(',')
        result = docFilter.filter( names=names )
        print( '-------------------------------------------------------------' )
        print( names, len( result ), result[:5] )
        for res in result[:5]:
            docViewer.view( int( res ) )

    period = None
    if params[ 'period' ]:
        period = params[ 'period' ]
        result = docFilter.filter( period=period )
        print( '-------------------------------------------------------------' )
        print( period, len( result ), result[:5] )
        for res in result[:5]:
            docViewer.view( int( res ) )

    query_analyzed = cast( QueryAnalyzedType, { 'terms': terms } )
    result = docFilter.filter( query_analyzed=query_analyzed, names=names, period=period )
    print( '-------------------------------------------------------------' )
    print( terms, names, period, len( result ), result[:5] )
    for res in result[:5]:
        docViewer.view( int( res ) )
