from abc import ABC, abstractmethod
from scipy.sparse import spmatrix
from sklearn.metrics.pairwise import cosine_similarity
from .helpers.Pickle import PickleLoader
from .helpers.computators import compute_similarities0

class AbstractSummarizer( ABC ):

    @abstractmethod
    def summarize( self, idoc:int, query_repr:spmatrix|None=None ):
        pass

# UNUSED code
# class DummySummarizer( AbstractSummarizer ):

#     def __init__( self, corpus:list[dict] ):
#         self._corpus = corpus

#     def summarize( self, idoc: int ) -> dict:
#         title = self._corpus[ idoc ][ 'title' ]
#         summarized = self._corpus[ idoc ][ 'summary' ]
#         return { 'title': title, 'summarized': summarized }


class NaiveSummarizer( AbstractSummarizer ):

    def __init__( self, corpus:list[dict], limit:int=50 ):
        self._corpus = corpus
        self._limit = limit

    def summarize( self, idoc:int ) -> dict:
        title = self._corpus[ idoc ][ 'title' ]
        summarized = self._corpus[ idoc ][ 'summary' ].split( ' ' )
        summarized = [ s for s in summarized if len(s)>0 ]
        dots = '...' if len( summarized ) > self._limit else '' 
        summarized = ' '.join( summarized[:self._limit] ) + dots
        return { 'title': title, 'summarized': summarized }


class SimilaritySummarizer( AbstractSummarizer ):

    def __init__( self, corpus:list[dict], sentences:list[tuple[str,str]], sent_repr:spmatrix, limit:int=50 ):
        self._corpus = corpus
        self._sentences = sentences
        self._sent_repr = sent_repr
        self._limit = limit

    def summarize( self, idoc:int, query_repr:spmatrix ) -> dict:

        # get sentences' tags, compute similarities, sort
        results = []
        for isent, sentence in enumerate( self._sentences ):
            tag = sentence[ 1 ].split( '.' )
            if str( idoc ) == tag[ 0 ] and tag[ 1 ] != '0': # sentence 0 is the title
                similarity = cosine_similarity( query_repr.reshape(1, -1), self._sent_repr[ isent ].reshape(1, -1) ) # type: ignore
                results.append( ( isent, similarity ) )
        results.sort( key=lambda x: x[1], reverse=True )

        # get sentences' texts
        summarized = [ self._sentences[isent][0] for isent, _ in results ]
        summarized = ' '.join( summarized )

        # prepare result
        title = self._corpus[ idoc ][ 'title' ]
        summarized = summarized.split( ' ' )
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

        case 'arxiv-glove-retrained-similarity':
            from .datasets.arXiv.Dataset import Dataset
            from .datasets.arXiv.settings import pickle_paths
            ds = Dataset()
            corpus = ds.toList()
            texts, tags = ds.toSentences()
            sentences = [ ( text, tag ) for text, tag in zip( texts, tags ) ] 
            representations_descr = 'sentences-glove-retrained'
            representations_filename = f"{pickle_paths[ 'corpus_repr' ]}/{representations_descr}.pkl"
            sent_repr = PickleLoader( representations_filename ).load()
            return SimilaritySummarizer( corpus, sentences, sent_repr )

        case 'arxiv-jina-similarity':
            from .datasets.arXiv.Dataset import Dataset
            from .datasets.arXiv.settings import pickle_paths
            ds = Dataset()
            corpus = ds.toList()
            texts, tags = ds.toSentences()
            sentences = [ ( text, tag ) for text, tag in zip( texts, tags ) ] 
            representations_descr = 'sentences-jina'
            representations_filename = f"{pickle_paths[ 'corpus_repr' ]}/{representations_descr}.pkl"
            sent_repr = PickleLoader( representations_filename ).load()
            return SimilaritySummarizer( corpus, sentences, sent_repr )

        case 'arxiv-bert-similarity':
            from .datasets.arXiv.Dataset import Dataset
            from .datasets.arXiv.settings import pickle_paths
            ds = Dataset()
            corpus = ds.toList()
            texts, tags = ds.toSentences()
            sentences = [ ( text, tag ) for text, tag in zip( texts, tags ) ] 
            representations_descr = 'sentences-bert'
            representations_filename = f"{pickle_paths[ 'corpus_repr' ]}/{representations_descr}.pkl"
            sent_repr = PickleLoader( representations_filename ).load()
            return SimilaritySummarizer( corpus, sentences, sent_repr )

        case 'medical-naive':
            from .datasets.medical.Dataset import Dataset
            corpus = Dataset().toList()
            for record in corpus:
                record[ 'summary' ] = record[ 'abstract' ]
            return NaiveSummarizer( corpus )

        case _:
            raise Exception( 'summarizerFactory(): No valid option.' )

# if __name__ == '__main__':
#     summarizerFactory( 'arxiv-glove-retrained-similarity' )