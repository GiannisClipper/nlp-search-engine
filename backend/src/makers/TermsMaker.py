from abc import ABC, abstractmethod
from .Tokenizer import SingleTokenizer, TwogramTokenizer

# --------------------------------------------------------------------- #
# Classes to extract uniques terms (single terms or 2grams) from a text #
# --------------------------------------------------------------------- #

# abstract class 
class AbstractTermsMaker( ABC ):

    def __init__( self, limit:int=0 ):
        self._limit = limit

    @abstractmethod
    def make( self, text:str ) -> list[str]:
        pass

    def __str__( self ):
        return self.__class__.__name__


class SingleTermsMaker( AbstractTermsMaker ):

    def make( self, text:str ) -> list[str]:
        tokens = SingleTokenizer().tokenize( text )
        singles = {}
        for token in tokens:
            singles[ token ] = singles.get( token, 0 ) + 1
        singles = list( singles.items() )
        singles.sort( key=lambda x: x[1], reverse=True )
        return [ x[0] for x in singles ][:self._limit if self._limit > 0 else None]


class TwogramTermsMaker( AbstractTermsMaker ):

    def make( self, text:str ) -> list[str]:
        tokens = TwogramTokenizer().tokenize( text )
        twograms = {}
        for token in tokens:
            twograms[ token ] = twograms.get( token, 0 ) + 1
        twograms = list( twograms.items() )
        twograms.sort( key=lambda x: x[1], reverse=True )
        return [ x[0] for x in twograms ][:self._limit if self._limit > 0 else None]


if __name__ == '__main__':

    text = 'This is a test from a collection of test examples but a test comprehensive enough'

    print( SingleTermsMaker().make( text ) )

    print( TwogramTermsMaker().make( text ) )