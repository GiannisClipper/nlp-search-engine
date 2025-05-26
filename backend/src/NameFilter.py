import sys
from .datasets.arXiv.Dataset import Dataset

# ------------------------------------------- #
# Code to filter docs matching specific names #
# ------------------------------------------- #

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

        # Apply the same transformation as data representation 
        name = self.__preprocess( name )
        subnames = [ n.strip() for n in name.split( ' ' ) ]

        # Get all subnames longer than 1 char in reverse order (surname first) 
        subnames1 = [ n for n in subnames if len( n ) > 1 ]
        subnames1.reverse()

        # Get all subnames with 1 char only 
        subnames2 = [ n for n in subnames if len( n ) == 1 ]

        # Merge the separated subnames 
        subnames = subnames1 + subnames2

        results = []
        for subname in subnames:
            results.append( [ tag for name, tag in zip( self._names, self._tags ) if ' '+subname+' ' in name ] )

        final = []
        for result in results:
            commons = list( set( final ) & set( result ) )

            # If common results
            if len( commons ) > 0:
                final = commons
                continue

            # If no common results
            final = final if len( final ) > 0 else result

        return final

    @property
    def tags( self ):
        return self._tags


class NamesFilter( NameFilter ):

    def __call__( self, names:list[str] ) -> list[str]:

        # Get results for each single name
        single_results = []
        for name in names: 
            single_results.append( super().__call__( name ) )

        # Leave doc index only, remove name position
        for i, res in enumerate( single_results ):
            single_results[ i ] = set( [ r.split('.')[0] for r in res ] )

        # Get the instersection from single name results
        results = single_results[ 0 ]
        for i in range( 1, len( single_results ) ):
            results = results & single_results[ i ]

        return list( results )


# +----------------------------------------+
# | For development and debugging purposes |
# +----------------------------------------+

# RUN: python -m src.NameFilter [text]
if __name__ == "__main__":

    names = "taylor,mendez".split(',')
    if len( sys.argv ) > 1:
        names = sys.argv[ 1 ].split(',')

    authors, tags = Dataset().toAuthors()
    # print( [ a for a in authors if 'taylor' in a.lower() ] )

    filter = NameFilter( authors, tags )
    for name in names:
        print( f"{name}:{filter(name)}" )

    filter = NamesFilter( authors, tags )
    print( f"{','.join(names)}:{filter(names)}" ) # type: ignore
