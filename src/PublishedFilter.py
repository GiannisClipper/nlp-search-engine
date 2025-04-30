import sys
from .arXiv.Dataset import Dataset

class PublishedFilter:

    def __init__( self, records:list[dict] ):

        self.records = records

    def __call__( self, text:str="" ) -> list[str]:

        text = text if ',' in text else text + ','
        from_date, to_date = text.split( ',' )

        results = []
        for i, doc in enumerate( self.records ):
            date = doc[ 'published' ]
            if ( from_date == '' or from_date <= date ) and ( to_date == '' or to_date >= date ):
                results.append( str( i ) )

        return results


# RUN: python -m src.PublishedFilter [from_date,to_date]
if __name__ == "__main__":

    option = None
    if len( sys.argv ) > 1:
        text = sys.argv[ 1 ]
        ds = Dataset()
        records = ds.toList()
        filter = PublishedFilter( records )
        print( filter( text ) )

    else:
        raise Exception( 'No text passed to filter.' )


