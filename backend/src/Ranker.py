from abc import ABC, abstractmethod
import sys
import numpy as np
from scipy.sparse import spmatrix
from .helpers.Pickle import PickleLoader
from .helpers.computators import compute_similarities0

class AbstractRanker( ABC ):

    def __init__( self, representations:spmatrix ):
        self._representations = representations

    @property
    def representations( self ):
        return self._representations

    @abstractmethod
    def rank( self, query_repr:spmatrix, filtered:list[str] ) -> list[tuple[str,float]]:
        pass

class DocRanker( AbstractRanker ):

    def __init__( self, representations:spmatrix ):
        super().__init__( representations )

    def rank( self, query_repr:spmatrix, filtered_idoc:list[str] ) -> list[tuple[str,float]]:

        # Get the corresponding corpus representations
        filtered_repr = np.array( [ self._representations[ int( idoc ) ] for idoc in filtered_idoc ] ) # type: ignore
        filtered_repr.reshape( len( filtered_idoc ), -1 )

        # Compute the similarities
        similarities = compute_similarities0( query_repr, filtered_repr )

        # Put together idocs and similarities
        results = [ ( idoc, round( sim, 4 ) ) for idoc, sim in zip( filtered_idoc, similarities ) ]

        # Put greater similarities at the top
        results.sort( key=lambda x: x[1], reverse=True )

        return results


class SentRanker( AbstractRanker ):

    def __init__( self, representations:spmatrix, tags:list[str] ):
        super().__init__( representations )
        self._tags = tags

    def rank( self, query_repr:spmatrix, filtered_isent:list[str] ) -> list[tuple[str,float]]:

        # Get the corresponding sentence representations
        filtered_repr = []
        filtered_tags = []
        for isent in filtered_isent:
            filtered_repr.append( self._representations[ int(isent) ] ) # type: ignore
            filtered_tags.append( self._tags[ int(isent) ] ) # type: ignore
        filtered_repr = np.array( filtered_repr )

        # Compute the similarities
        similarities = compute_similarities0( query_repr, filtered_repr )

        # Aggregate similarities (per document)
        doc_result = {}
        for tag, similarity in zip( filtered_tags, similarities ):
            idoc = int( tag.split('.')[0] )
            if not idoc in doc_result:
                doc_result[ idoc ] = similarity
            elif doc_result[ idoc ] < similarity:
                doc_result[ idoc ] = similarity

        # Put together idocs and similarities
        results = [ ( idoc, round( sim, 4 ) ) for idoc, sim in doc_result.items() ]

        # Put greater similarities at the top
        results.sort( key=lambda x: x[1], reverse=True )

        return results


def rankerFactory( option:str ):

    match option:

        case 'arxiv-stemm-single-count':
            from .datasets.arXiv.settings import pickle_paths
            vectorizer_descr = 'title-summary_lower-punct-specials-stops-stemm_single_count'
            corpus_repr_filename = f"{pickle_paths[ 'corpus_repr' ]}/{vectorizer_descr}.pkl"
            corpus_repr = PickleLoader( corpus_repr_filename ).load()
            return DocRanker( corpus_repr )

        case 'arxiv-lemm-single-tfidf':
            from .datasets.arXiv.settings import pickle_paths
            vectorizer_descr = 'title-summary_lower-punct-specials-stops-lemm_single_tfidf'
            corpus_repr_filename = f"{pickle_paths[ 'corpus_repr' ]}/{vectorizer_descr}.pkl"
            corpus_repr = PickleLoader( corpus_repr_filename ).load()
            return DocRanker( corpus_repr )

        case 'arxiv-lemm-2gram-tfidf':
            from .datasets.arXiv.settings import pickle_paths
            vectorizer_descr = 'title-summary_lower-punct-specials-stops-lemm_2gram_tfidf'
            corpus_repr_filename = f"{pickle_paths[ 'corpus_repr' ]}/{vectorizer_descr}.pkl"
            corpus_repr = PickleLoader( corpus_repr_filename ).load()
            return DocRanker( corpus_repr )

        case 'arxiv-glove':
            from .datasets.arXiv.Dataset import Dataset
            from .datasets.arXiv.settings import pickle_paths
            representations_descr = 'sentences-glove'
            representations_filename = f"{pickle_paths[ 'corpus_repr' ]}/{representations_descr}.pkl"
            representations = PickleLoader( representations_filename ).load()
            sentences, tags = Dataset().toSentences()
            return SentRanker( representations, tags )

        case 'arxiv-glove-retrained':
            from .datasets.arXiv.Dataset import Dataset
            from .datasets.arXiv.settings import pickle_paths
            representations_descr = 'sentences-glove-retrained'
            representations_filename = f"{pickle_paths[ 'corpus_repr' ]}/{representations_descr}.pkl"
            representations = PickleLoader( representations_filename ).load()
            sentences, tags = Dataset().toSentences()
            return SentRanker( representations, tags )

        case 'arxiv-bert':
            from .datasets.arXiv.Dataset import Dataset
            from .datasets.arXiv.settings import pickle_paths
            representations_descr = 'sentences-bert'
            representations_filename = f"{pickle_paths[ 'corpus_repr' ]}/{representations_descr}.pkl"
            representations = PickleLoader( representations_filename ).load()
            sentences, tags = Dataset().toSentences()
            return SentRanker( representations, tags )

        case 'arxiv-jina':
            from .datasets.arXiv.Dataset import Dataset
            from .datasets.arXiv.settings import pickle_paths
            representations_descr = 'sentences-jina'
            representations_filename = f"{pickle_paths[ 'corpus_repr' ]}/{representations_descr}.pkl"
            representations = PickleLoader( representations_filename ).load()
            sentences, tags = Dataset().toSentences()
            return SentRanker( representations, tags )

        ###########
        # medical #
        ###########

        case 'medical-stemm-single-count':
            from .datasets.medical.settings import pickle_paths
            vectorizer_descr = 'title-summary_lower-punct-specials-stops-stemm_single_count'
            corpus_repr_filename = f"{pickle_paths[ 'corpus_repr' ]}/{vectorizer_descr}.pkl"
            corpus_repr = PickleLoader( corpus_repr_filename ).load()
            return DocRanker( corpus_repr )

        case 'medical-lemm-single-tfidf':
            from .datasets.medical.settings import pickle_paths
            vectorizer_descr = 'title-summary_lower-punct-specials-stops-lemm_single_tfidf'
            corpus_repr_filename = f"{pickle_paths[ 'corpus_repr' ]}/{vectorizer_descr}.pkl"
            corpus_repr = PickleLoader( corpus_repr_filename ).load()
            return DocRanker( corpus_repr )

        case 'medical-lemm-2gram-tfidf':
            from .datasets.medical.settings import pickle_paths
            vectorizer_descr = 'title-summary_lower-punct-specials-stops-lemm_2gram_tfidf'
            corpus_repr_filename = f"{pickle_paths[ 'corpus_repr' ]}/{vectorizer_descr}.pkl"
            corpus_repr = PickleLoader( corpus_repr_filename ).load()
            return DocRanker( corpus_repr )

        case 'medical-glove':
            from .datasets.medical.Dataset import Dataset
            from .datasets.medical.settings import pickle_paths
            representations_descr = 'sentences-glove'
            representations_filename = f"{pickle_paths[ 'corpus_repr' ]}/{representations_descr}.pkl"
            representations = PickleLoader( representations_filename ).load()
            sentences, tags = Dataset().toSentences()
            return SentRanker( representations, tags )

        case 'medical-glove-retrained':
            from .datasets.medical.Dataset import Dataset
            from .datasets.medical.settings import pickle_paths
            representations_descr = 'sentences-glove-retrained'
            representations_filename = f"{pickle_paths[ 'corpus_repr' ]}/{representations_descr}.pkl"
            representations = PickleLoader( representations_filename ).load()
            sentences, tags = Dataset().toSentences()
            return SentRanker( representations, tags )

        case 'medical-jina':
            from .datasets.medical.Dataset import Dataset
            from .datasets.medical.settings import pickle_paths
            representations_descr = 'sentences-jina'
            representations_filename = f"{pickle_paths[ 'corpus_repr' ]}/{representations_descr}.pkl"
            representations = PickleLoader( representations_filename ).load()
            sentences, tags = Dataset().toSentences()
            return SentRanker( representations, tags )

        case 'medical-bert':
            from .datasets.medical.Dataset import Dataset
            from .datasets.medical.settings import pickle_paths
            representations_descr = 'sentences-bert'
            representations_filename = f"{pickle_paths[ 'corpus_repr' ]}/{representations_descr}.pkl"
            representations = PickleLoader( representations_filename ).load()
            sentences, tags = Dataset().toSentences()
            return SentRanker( representations, tags )

        case 'medical-bert-retrained':
            from .datasets.medical.Dataset import Dataset
            from .datasets.medical.settings import pickle_paths
            representations_descr = 'sentences-bert-retrained'
            representations_filename = f"{pickle_paths[ 'corpus_repr' ]}/{representations_descr}.pkl"
            representations = PickleLoader( representations_filename ).load()
            sentences, tags = Dataset().toSentences()
            return SentRanker( representations, tags )

        case _:
            raise Exception( 'rankerFactory(): No valid option.' )


##########################################
# for development and debugging purposes #
##########################################

# RUN: python -m src.DocEstimator [option]
if __name__ == "__main__": 

    option = None
    if len( sys.argv ) >= 2:
        option = sys.argv[ 1 ]

    match option:

        case 'arxiv-stemm-single-count':
            ranker = rankerFactory( option )
            query_repr = ranker.representations[ 3 ] # type: ignore
            print( ranker.rank( query_repr, [ '0', '1', '2', '3', '4' ] ) )

        case 'arxiv-lemm-single-tfidf':
            ranker = rankerFactory( option )
            query_repr = ranker.representations[ 3 ] # type: ignore
            print( ranker.rank( query_repr, [ '0', '1', '2', '3', '4' ] ) )

        case 'arxiv-jina':
            ranker = rankerFactory( option )
            query_repr = ranker.representations[ 0 ] # type: ignore
            print( ranker.rank( query_repr, [ '0', '1', '2', '3', '4' ] ) )

        case 'medical-lemm-single-tfidf':
            ranker = rankerFactory( option )
            query_repr = ranker.representations[ 3 ] # type: ignore
            print( ranker.rank( query_repr, [ '0', '1', '2', '3', '4' ] ) )

        case _:
            raise Exception( 'No valid option.' )

