import sys
from .arXiv.Dataset import Dataset

class NameFilter:

    def __init__( self, names:list[str], tags:list[str] ):

        self._names = [ self.__preprocess( a ) for a in names ]
        self._tags = tags

    def __preprocess( self, name:str ) -> str:
        no_dots = ' '.join( name.split( '.' ) )
        no_dashes = ' '.join( name.split( '-' ) )
        lower = no_dashes.lower()
        return ' ' + lower + ' ' 

    def __call__( self, name:str ) -> list[str]:

        # apply the same transformation as data representation 
        name = self.__preprocess( name )
        subnames = [ n.strip() for n in name.split( ' ' ) ]

        # get all subnames longer than 1 char in reverse order (surname first) 
        subnames1 = [ n for n in subnames if len( n ) > 1 ]
        subnames1.reverse()

        # get all subnames with 1 char only 
        subnames2 = [ n for n in subnames if len( n ) == 1 ]

        # merge the separated subnames 
        subnames = subnames1 + subnames2

        results = []
        for subname in subnames:
            results.append( [ tag for name, tag in zip( self._names, self._tags ) if ' '+subname+' ' in name ] )

        final = []
        for result in results:
            commons = list( set( final ) & set( result ) )

            # if common results
            if len( commons ) > 0:
                final = commons
                continue

            # if no common results
            final = final if len( final ) > 0 else result

        return final

    @property
    def tags( self ):
        return self._tags


# RUN: python -m src.NameFilter [text]
if __name__ == "__main__":

    option = None
    if len( sys.argv ) > 1:
        text = sys.argv[ 1 ]
        ds = Dataset()
        authors, tags = ds.toAuthors()
        filter = NameFilter( authors, tags )
        print( filter( text ) )
        # print( [ a for a in authors if 'taylor' in a.lower() ] )

    else:
        raise Exception( 'No text passed to filter.' )


