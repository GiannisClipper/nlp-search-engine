import sys

from .TermsFilter import TermsFilter
from .NameFilter import NameFilter
from .PeriodFilter import PeriodFilter

from .helpers.Pickle import PickleLoader

class DocFilter:

    def __init__( self, termsFilter:TermsFilter, nameFilter:NameFilter, periodFilter:PeriodFilter ):
        self.termsFilter = termsFilter
        self.nameFilter = nameFilter
        self.periodFilter = periodFilter

    def _names_period_filter( self, names:list[str]|None=None, period:str|None=None ) -> list[str]:

        # Filter by date
        # --------------

        period_filtered_docs = self.periodFilter.tags # e.g. tags -> '234', '235', ...
        if period:
            period_filtered_docs = self.periodFilter( period )

        # No doc match the date filter 
        if len( period_filtered_docs ) == 0:
            return period_filtered_docs

        # Filter by names
        # ---------------

        names_filtered_doc = set( [ t.split('.')[0] for t in self.nameFilter.tags ] ) # e.g. tags -> '234.0', '234.1', '235.0', ...
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
        
        # Intersect date and names filters
        # --------------------------------

        period_names_filtered_docs = list( set( period_filtered_docs ) & set( names_filtered_doc ) )
        return period_names_filtered_docs

    def _terms_filter( self, terms:list[str] ) -> list[str]:

        terms_filtered_doc = self.termsFilter.filter( terms )
        terms_filtered_doc = [ str(t) for t in terms_filtered_doc ]
        return terms_filtered_doc

    def filter( self, terms:list[str]|None=None, names:list[str]|None=None, period:str|None=None ) -> list[str]:

        if names or period:
            return self._names_period_filter( names=names, period=period )

        if terms:
            return self._terms_filter( terms=terms )
        
        return []


# RUN: python -m src.DocFilter [option]
if __name__ == "__main__": 

    from .arXiv.Dataset import Dataset
    from .arXiv.settings import pickle_paths

    params = dict()
    for i in range( 1, len( sys.argv ) ):
        key, value = sys.argv[ i ].split( ':' )
        params[ key ] = value

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

    docFilter = DocFilter( termsFilter=termsFilter, nameFilter=nameFilter, periodFilter=periodFilter )

    result = docFilter.filter( terms=[ 'gradient', 'descent' ] )
    print( len( result ), result[:10] )

    result = docFilter.filter( names=[ 'taylor' ] )
    print( len( result ), result[:10] )

    result = docFilter.filter( period='2025-01-07,2025-01-08' )
    print( len( result ), result[:10] )
