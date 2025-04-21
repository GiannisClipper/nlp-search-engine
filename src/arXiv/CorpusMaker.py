from abc import ABC, abstractmethod

from .Dataset import Dataset


# abstract class 
class CorpusMaker( ABC ):
    def __init__( self ):
        pass

    @abstractmethod
    def make( self ) -> list[str]:
        pass

    def __str__( self ):
        return self.__class__.__name__


# concatenates titles & summaries
class TitleSummaryCorpusMaker( CorpusMaker ):

    def make( self ) -> list[str]:
        ds = Dataset()
        records = ds.toDictList()
        corpus = []
        for i in range( len( records ) ):
            corpus.append( records[ i ][ 'title' ] + '-' + records[ i ][ 'summary' ] )
        return corpus
