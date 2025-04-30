import sys
from .arXiv.Dataset import Dataset

class DateFilter:

    def __init__( self, dates:list[str], tags:list[str] ):

        self.dates = dates
        self.tags = tags

    def __call__( self, text:str="" ) -> list[str]:

        text = text if ',' in text else text + ','
        from_date, to_date = text.split( ',' )

        results = []
        for date, tag in zip( self.dates, self.tags ):
            if ( from_date == '' or from_date <= date ) and ( to_date == '' or to_date >= date ):
                results.append( tag )

        return results


# RUN: python -m src.DateFilter [from_date,to_date]
if __name__ == "__main__":

    option = None
    if len( sys.argv ) > 1:
        text = sys.argv[ 1 ]
        ds = Dataset()
        dates, tags = ds.toPublished()
        filter = DateFilter( dates, tags )
        print( filter( text ) )

    else:
        raise Exception( 'No text passed to filter.' )


