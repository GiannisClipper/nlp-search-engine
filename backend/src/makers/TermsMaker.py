from abc import ABC, abstractmethod

import nltk
nltk.download( 'punkt_tab' ) # required by word_tokenize()

from nltk.tokenize import word_tokenize
from nltk import ngrams

# abstract class 
class TermsMaker( ABC ):

    def __init__( self, limit:int=0 ):
        self._limit = limit

    @abstractmethod
    def make( self, text:str ) -> tuple[str,...]:
        self._tokens = tuple( word_tokenize( text ) )

    @property
    def tokens( self ) -> tuple[str,...]:
        return self._tokens

    def __str__( self ):
        return self.__class__.__name__


class SingleTermsMaker( TermsMaker ):

    def make( self, text:str ) -> tuple[str,...]:
        super().make( text )
        singles = {}
        for token in self._tokens:
            singles[ token ] = singles.get( token, 0 ) + 1
        singles = list( singles.items() )
        singles.sort( key=lambda x: x[1], reverse=True )
        self._tokens = tuple( x[0] for x in singles )[:self._limit if self._limit > 0 else None]
        return self._tokens


class TwogramTermsMaker( TermsMaker ):

    def make( self, text:str ) -> tuple[str,...]:
        super().make( text )
        self._tokens = tuple( x + ' ' + y for x, y in ngrams( self._tokens, 2 ) )
        twograms = {}
        for token in self._tokens:
            twograms[ token ] = twograms.get( token, 0 ) + 1
        twograms = list( twograms.items() )
        twograms.sort( key=lambda x: x[1], reverse=True )
        self._tokens =  tuple( x[0] for x in twograms )[:self._limit if self._limit > 0 else None]
        return self._tokens

