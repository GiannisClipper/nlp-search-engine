from abc import ABC, abstractmethod

class AbstractSummarizer( ABC ):

    @abstractmethod
    def summarize( self, idoc:int ):
        pass

# UNUSED code
# class DummySummarizer( AbstractSummarizer ):

#     def __init__( self, corpus:list[dict] ):
#         self._corpus = corpus

#     def summarize( self, idoc: int ):
#         title = self._corpus[ idoc ][ 'title' ]
#         summarized = self._corpus[ idoc ][ 'summary' ]
#         return { 'title': title, 'summarized': summarized }


class NaiveSummarizer( AbstractSummarizer ):

    def __init__( self, corpus:list[dict], limit:int=50 ):
        self._corpus = corpus
        self._limit = limit

    def summarize( self, idoc: int ):
        title = self._corpus[ idoc ][ 'title' ]
        summarized = self._corpus[ idoc ][ 'summary' ].split( ' ' )
        summarized = [ s for s in summarized if len(s)>0 ]
        dots = '...' if len( summarized ) > self._limit else '' 
        summarized = ' '.join( summarized[:self._limit] ) + dots
        return { 'title': title, 'summarized': summarized }


def summarizerFactory( option:str ) -> AbstractSummarizer:

    match option:

        case 'arxiv-naive':
            from .datasets.arXiv.Dataset import Dataset
            corpus = Dataset().toList()
            return NaiveSummarizer( corpus )

        case 'medical-dummy':
            from .datasets.medical.Dataset import Dataset
            corpus = Dataset().toList()
            for record in corpus:
                record[ 'summary' ] = record[ 'abstract' ]
            return NaiveSummarizer( corpus )

        case _:
            raise Exception( 'summarizerFactory(): No valid option.' )
