import sys

from .TermsFilter import TermsFilter
from .NameFilter import NameFilter
from .PeriodFilter import PeriodFilter
from .DocViewer import DocViewer
from .helpers.Pickle import PickleLoader

class DocFilter:

    def __init__( self, termsFilter:TermsFilter, nameFilter:NameFilter, periodFilter:PeriodFilter ):
        self.termsFilter = termsFilter
        self.nameFilter = nameFilter
        self.periodFilter = periodFilter

    def _names_period_filter( self, terms:list[str]|None=None, names:list[str]|None=None, period:str|None=None ) -> list[str]:

        # Filter by period
        # ----------------

        period_filtered_docs = self.periodFilter.tags # e.g. tags -> '234', '235', ...
        if period:
            period_filtered_docs = self.periodFilter( period )

        # No doc match the date filter 
        if len( period_filtered_docs ) == 0:
            return period_filtered_docs

        # Filter by names
        # ---------------

        names_filtered_doc = list( set( [ t.split('.')[0] for t in self.nameFilter.tags ] ) ) # e.g. tags -> '234.0', '234.1', '235.0', ...
        if names:
            single_results = []
            for name in names: 
                single_results.append( self.nameFilter( name ) )

            # Leave doc index only, remove name position
            for i, res in enumerate( single_results ):
                single_results[ i ] = set( [ r.split('.')[0] for r in res ] )

            # Get the instersection from single name results
            names_filtered_doc = single_results[ 0 ]
            for i in range( 1, len( single_results ) ):
                names_filtered_doc = names_filtered_doc & single_results[ i ]
            names_filtered_doc = list( names_filtered_doc )

        # No doc match the names filter 
        if len( names_filtered_doc ) == 0:
            return names_filtered_doc
        
        # Intersect period, names filters
        # -------------------------------

        period_names_filtered_docs = list( set( period_filtered_docs ) & set( names_filtered_doc ) )

        # No doc match both period, names filters 
        if len( period_names_filtered_docs ) == 0:
            return period_names_filtered_docs

        # Filter by terms
        # ---------------

        if not terms:
            return period_names_filtered_docs

        terms_filtered_docs = self.termsFilter.filter( terms, threshold=0.0 )
        terms_filtered_docs = [ str(t) for t in terms_filtered_docs ]

        # Intersect period, names, terms filters
        # --------------------------------------

        filtered_docs = list( set( period_names_filtered_docs ) & set( terms_filtered_docs ) )

        return filtered_docs

    def _terms_filter( self, terms:list[str] ) -> list[str]:

        terms_filtered_doc = self.termsFilter.filter( terms, threshold=0.5 )
        terms_filtered_doc = [ str(t) for t in terms_filtered_doc ]
        return terms_filtered_doc

    def filter( self, terms:list[str]|None=None, names:list[str]|None=None, period:str|None=None ) -> list[str]:

        if names or period:
            return self._names_period_filter( terms=terms, names=names, period=period )

        if terms:
            return self._terms_filter( terms=terms )
        
        return []


def docFilterFactory( option:str ):

    match option:

        case 'arxiv-lemm-single':
            from .arXiv.Dataset import Dataset
            from .arXiv.settings import pickle_paths
            ds = Dataset()

            dates, tags = ds.toPublished()
            periodFilter = PeriodFilter( dates=dates, tags=tags )

            names, tags = ds.toAuthors()
            nameFilter = NameFilter( names=names, tags=tags )

            corpus = ds.toList()
            index_descr = 'title-summary_lower-punct-specials-stops-lemm_single'
            index_filename = f"{pickle_paths[ 'indexes' ]}/{index_descr}.pkl"
            index = PickleLoader( index_filename ).load()
            termsFilter = TermsFilter( corpus=corpus, index=index )

            return DocFilter( termsFilter=termsFilter, nameFilter=nameFilter, periodFilter=periodFilter )

        case _:
            raise Exception( 'docFilterFactory(): No valid option.' )



# RUN: python -m src.DocFilter [option]
if __name__ == "__main__": 

    # initialize involved instances

    docFilter = docFilterFactory( 'arxiv-lemm-single' )

    from .arXiv.Dataset import Dataset
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
        params[ 'names' ] = 'jonathan'
        params[ 'period' ] = '2024-01-01,2024-12-31'

    # perform filterings

    terms = None
    if params[ 'terms' ]:
        terms = params[ 'terms' ].split(',')
        result = docFilter.filter( terms=terms )
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

    result = docFilter.filter( terms=terms, names=names, period=period )
    print( '-------------------------------------------------------------' )
    print( terms, names, period, len( result ), result[:5] )
    for res in result[:5]:
        docViewer.view( int( res ) )
