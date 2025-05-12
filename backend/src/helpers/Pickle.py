import pickle

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

class PickleSaver:

    def __init__( self, filename ):
        self._filename = filename

    def save( self, content ):
        print( f'Saving {self._filename} in disk...' )
        with open( f"{self._filename}", 'wb' ) as f:
            pickle.dump( content, f )

    def __str__( self ):
        return self.__class__.__name__
