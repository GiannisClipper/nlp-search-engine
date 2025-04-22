from abc import ABC, abstractmethod

from .Dataset import Dataset as ArxivDataset

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
        ds = ArxivDataset()
        records = ds.toList()
        corpus = []
        for i in range( len( records ) ):
            corpus.append( records[ i ][ 'title' ] + '-' + records[ i ][ 'summary' ] )
        return corpus


if __name__ == "__main__":
    cl = TitleSummaryCorpusLoader()
    corpus = cl.load()
    print( corpus[:3] )
