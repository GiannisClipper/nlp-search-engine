import sys
from .datasets.arXiv.Dataset import Dataset

class PeriodFilter:

    def __init__( self, dates:list[str], tags:list[str] ):

        self._dates = dates
        self._tags = tags

    def __call__( self, text:str="" ) -> list[str]:

        text = text if ',' in text else text + ','
        from_date, to_date = text.split( ',' )

        results = []
        for date, tag in zip( self._dates, self._tags ):
            if ( from_date == '' or from_date <= date ) and ( to_date == '' or to_date >= date ):
                results.append( tag )

        return results
    
    @property
    def tags( self ):
        return self._tags


# RUN: python -m src.PeriodFilter [from_date,to_date]
if __name__ == "__main__":

    option = None
    if len( sys.argv ) > 1:
        text = sys.argv[ 1 ]
        ds = Dataset()
        dates, tags = ds.toPublished()
        filter = PeriodFilter( dates, tags )
        print( filter( text ) )

    else:
        raise Exception( 'No text passed to filter.' )


