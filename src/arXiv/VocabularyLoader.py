import pickle
from .settings import pickle_paths

class VocabularyLoader:
    def __init__( self, descr ):
        self._descr = descr

    def load( self ) -> list[str]:
        with open( f"{pickle_paths[ 'vocabularies' ]}/{self._descr}.pkl", 'rb' ) as f:
           vocabulary = pickle.load( f )
        return vocabulary

    def __str__( self ):
        return self.__class__.__name__
