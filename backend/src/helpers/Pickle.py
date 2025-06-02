import pickle

class PickleSaver:

    def __init__( self, filename ):
        self._filename = filename

    def save( self, content ):
        print( f'Saving {self._filename} in disk...' )
        with open( f"{self._filename}", 'wb' ) as f:
            pickle.dump( content, f )

    def __str__( self ):
        return self.__class__.__name__


class PickleLoader:

    def __init__( self, filename ):
        self._filename = filename

    def load( self ):
        print( f'Loading {self._filename} from disk...' )
        with open( f"{self._filename}", 'rb' ) as f:
           result = pickle.load( f )
        return result

    def __str__( self ):
        return self.__class__.__name__


class CachedPickleLoader( object ):

    _cache = {}

    def __new__( cls, *args, **kwargs ):

        obj = super( CachedPickleLoader, cls ).__new__( cls )
        obj.__dict__ = cls._cache
        return obj

    def __init__( self, filename ):
        self._filename = filename

    def load( self ):
        if self._filename in self._cache.keys():
            print( f'Loading {self._filename} from cache...' )
            return self._cache[ self._filename ]
        
        print( f'Loading {self._filename} from disk...' )
        with open( f"{self._filename}", 'rb' ) as f:
            result = pickle.load( f )

        self._cache[ self._filename ] = result
        return result
