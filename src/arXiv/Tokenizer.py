from abc import ABC, abstractmethod

import nltk
nltk.download( 'punkt_tab' ) # required by word_tokenize()

from nltk.tokenize import word_tokenize
from nltk import ngrams

# abstract class 
class Tokenizer( ABC ):
    def __init__( self, text:str ):
        self._tokens = tuple( word_tokenize( text ) )

    @property
    def tokens( self ) -> tuple[str]:
        return self._tokens

    def __str__( self ):
        return self.__class__.__name__

class SingleTokenizer( Tokenizer ):
    def __init__( self, text:str ):
        super().__init__( text )
        singles = {}
        for token in self._tokens:
            singles[ token ] = singles.get( token, 0 ) + 1
        singles = list( singles.items() )
        singles.sort( key=lambda x: x[1], reverse=True )
        self._tokens = tuple( x[0] for x in singles )

class TwogrammTokenizer( Tokenizer ):
    def __init__( self, text:str ):
        super().__init__( text )
        self._tokens = tuple( x + ' ' + y for x, y in ngrams( self._tokens, 2 ) )
        twograms = {}
        for token in self._tokens:
            twograms[ token ] = twograms.get( token, 0 ) + 1
        twograms = list( twograms.items() )
        twograms.sort( key=lambda x: x[1], reverse=True )
        self._tokens =  tuple( x[0] for x in twograms )
