from abc import ABC, abstractmethod

from .Dataset import Dataset as MedicalDataset

# abstract class 
class CorpusLoader( ABC ):
    def __init__( self ):
        pass

    @abstractmethod
    def load( self ) -> list[str]:
        pass

    def __str__( self ):
        return self.__class__.__name__

# concatenates titles & abstracts
class TitleAbstractCorpusLoader( CorpusLoader ):

    def load( self ) -> list[str]:
        ds = MedicalDataset()
        records = ds.toList()
        corpus = []
        for i in range( len( records ) ):
            corpus.append( records[ i ][ 'title' ] + '-' + records[ i ][ 'abstract' ] )
        return corpus


if __name__ == "__main__":
    cl = TitleAbstractCorpusLoader()
    corpus = cl.load()
    print( corpus[:3] )
