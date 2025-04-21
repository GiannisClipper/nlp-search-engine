from abc import ABC, abstractmethod

from .Dataset import Dataset


# abstract class 
class CorpusLoader( ABC ):
    def __init__( self ):
        pass

    @abstractmethod
    def load( self ) -> list[str]:
        pass

    def __str__( self ):
        return self.__class__.__name__


# concatenates titles & summaries
class TitleSummaryCorpusLoader( CorpusLoader ):

    def load( self ) -> list[str]:
        ds = Dataset()
        records = ds.toDictList()
        corpus = []
        for i in range( len( records ) ):
            corpus.append( records[ i ][ 'title' ] + '-' + records[ i ][ 'summary' ] )
        return corpus
