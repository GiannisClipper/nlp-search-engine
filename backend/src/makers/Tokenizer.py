from abc import ABC, abstractmethod
import nltk
nltk.download( 'punkt_tab' ) # required by word_tokenize()
from nltk.tokenize import word_tokenize
from nltk import ngrams

# --------------------------------------------------------------- #
# Code to split text into tokens (single terms or 2grams or both) #
# --------------------------------------------------------------- #

# abstract class 
class AbstractTokenizer( ABC ):

    @abstractmethod
    def tokenize( self, text:str ) -> list[str]:
        pass

    def __str__( self ):
        return self.__class__.__name__


class SingleTokenizer( AbstractTokenizer ):

    def tokenize( self, text:str ) -> list[str]:
        return word_tokenize( text )


class TwogramTokenizer( AbstractTokenizer ):

    def tokenize( self, text:str ) -> list[str]:
        tokens = word_tokenize( text )
        return [  x + ' ' + y for x, y in  ngrams( tokens, 2 ) ]


class SingleAndTwogramTokenizer( AbstractTokenizer ):

    def tokenize( self, text:str ) -> list[str]:
        tokens = word_tokenize( text )
        twograms = [  x + ' ' + y for x, y in  ngrams( tokens, 2 ) ]
        return tokens + twograms
    

if __name__ == '__main__':

    text = 'This is a test from a collection of test examples but a test comprehensive enough'

    print( SingleTokenizer().tokenize( text ) )

    print( TwogramTokenizer().tokenize( text ) )

    print( SingleAndTwogramTokenizer().tokenize( text ) )
