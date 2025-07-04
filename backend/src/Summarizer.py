from abc import ABC, abstractmethod
from scipy.sparse import spmatrix
from sklearn.metrics.pairwise import cosine_similarity
from .helpers.Pickle import PickleLoader, CachedPickleLoader
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
        summarized = self._corpus[ idoc ][ 'summary' ].split( ' ' )
        summarized = [ s for s in summarized if len(s)>0 ]
        dots = '...' if len( summarized ) > self._limit else '' 
        summarized = ' '.join( summarized[:self._limit] ) + dots
        return { 
            'idoc': idoc,
            'id': self._corpus[ idoc ][ 'id' ],
            'catg_ids': self._corpus[ idoc ][ 'catg_ids' ],
            'authors': self._corpus[ idoc ][ 'authors' ],
            'published': self._corpus[ idoc ][ 'published' ],
            'title': self._corpus[ idoc ][ 'title' ], 
            'summarized': summarized 
        }


class SimilaritySummarizer( AbstractSummarizer ):

    def __init__( self, corpus:list[dict], sentences:list[tuple[str,str]], sent_repr:spmatrix, limit:int=50 ):
        self._corpus = corpus
        self._sentences = sentences
        self._sent_repr = sent_repr
        self._limit = limit
        self._index = {}
        idoc = ''
        for isent, sentence in enumerate( sentences ):
            tag = sentence[ 1 ].split( '.' )
            if idoc != tag[0]:
                idoc = tag[0]
                self._index[ idoc ] = isent

    def summarize( self, idoc:int, query_repr:spmatrix ) -> dict:

        # get sentences and compute similarities
        results = []
        isent = self._index[ str(idoc) ]
        while True:
            tag = self._sentences[ isent ][ 1 ].split( '.' )
            if tag[ 0 ] != str(idoc):
                break

            if tag[ 1 ] != '0': # to omit the title
                similarity = cosine_similarity( query_repr.reshape(1, -1), self._sent_repr[ isent ].reshape(1, -1) ) # type: ignore
                results.append( [ self._sentences[isent][0], similarity ] )

            isent += 1
            if isent >= len( self._sentences ):
                break

        # summarizing process
        summarized = ''
        while True:

            # compose a summarized paragraph
            summarized = [ sent for sent, _ in results ]
            summarized = ' '.join( summarized )
            summarized = summarized.split( ' ' )
            summarized = [ s for s in summarized if len(s)>0 ]

            # check the current state and decide
            if len( summarized ) <= self._limit or len( results ) <= 1:
                if len( summarized ) > self._limit:
                    summarized = summarized[:self._limit] 
                    summarized.append( ' (...)' )
                summarized = ' '.join( summarized )
                break
            
            # remove the sentence with the minimum similarity
            imin = 0
            min_sim = results[0][1]
            for i, ( _, sim ) in enumerate( results ):
                if sim <= min_sim:
                    imin = i
            if imin == 0 and results[1][0][0:6] != '(...) ':
                results[1][0] = '(...) ' + results[1][0] 
            if imin > 0 and results[imin-1][0][-6:] != ' (...)':
                results[imin-1][0] = results[imin-1][0] + ' (...)'
            results.pop( imin )

        return { 
            'idoc': idoc,
            'id': self._corpus[ idoc ][ 'id' ],
            'catg_ids': self._corpus[ idoc ][ 'catg_ids' ],
            'authors': self._corpus[ idoc ][ 'authors' ],
            'published': self._corpus[ idoc ][ 'published' ],
            'title': self._corpus[ idoc ][ 'title' ], 
            'summarized': summarized 
        }


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
            sent_repr = CachedPickleLoader( representations_filename ).load()
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
            sent_repr = CachedPickleLoader( representations_filename ).load()
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
            sent_repr = CachedPickleLoader( representations_filename ).load()
            return SimilaritySummarizer( corpus, sentences, sent_repr )

        case 'medical-naive':
            from .datasets.medical.Dataset import Dataset
            corpus = Dataset().toList()
            for record in corpus:
                record[ 'id' ] = None # temporary solution to match both datasets
                record[ 'catg_ids' ] = None # temporary solution to match both datasets
                record[ 'authors' ] = None # temporary solution to match both datasets
                record[ 'published' ] = None # temporary solution to match both datasets
                record[ 'summary' ] = record[ 'abstract' ] # temporary solution to match both datasets
            return NaiveSummarizer( corpus )

        case _:
            raise Exception( 'summarizerFactory(): No valid option.' )

# if __name__ == '__main__':
#     summarizerFactory( 'arxiv-glove-retrained-similarity' )