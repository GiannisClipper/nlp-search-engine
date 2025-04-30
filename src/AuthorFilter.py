import sys
from .arXiv.Dataset import Dataset

class AuthorFilter:

    def __init__( self, authors:list[str], tags:list[str] ):

        self.authors = [ self.__preprocess( a ) for a in authors ]
        self.tags = tags

    def __preprocess( self, author:str ) -> str:
        return ''.join( author.lower().split( '.' ) )

    def __call__( self, author:str ) -> list[str]:

        # apply the same transformation as data representation 
        author = self.__preprocess( author )
        names = [ a.strip() for a in author.split( ' ' ) ]

        # get all subnames longer than 1 char in reverse order (surname first) 
        names1 = [ n for n in names if len( n ) > 1 ]
        names1.reverse()

        # get all subnames with 1 char only 
        names2 = [ n for n in names if len( n ) == 1 ]

        # merge the separated subnames 
        names = names1 + names2

        results = []
        for name in names:
            results.append( [ t for a, t in zip( self.authors, self.tags ) if name in a ] )

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


# RUN: python -m src.AuthorFilter [text]
if __name__ == "__main__":

    option = None
    if len( sys.argv ) > 1:
        text = sys.argv[ 1 ]
        ds = Dataset()
        authors, tags = ds.toAuthors()
        filter = AuthorFilter( authors, tags )
        print( filter( text ) )

    else:
        raise Exception( 'No text passed to filter.' )


