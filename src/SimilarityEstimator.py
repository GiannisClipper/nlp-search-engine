from abc import ABC, abstractmethod
import sys
import numpy as np
from scipy.sparse import spmatrix
from .helpers.Pickle import PickleLoader
from .helpers.computators import compute_similarities0

class AbstractSimilarityEstimator( ABC ):

    def __init__( self, representations:spmatrix ):
        self._representations = representations

    @property
    def representations( self ):
        return self._representations

    @abstractmethod
    def estimate( self, query_repr:spmatrix, filtered_docs:list[str] ) -> list[tuple[str,float]]:
        pass

class DocSimilarityEstimator( AbstractSimilarityEstimator ):

    def __init__( self, representations:spmatrix ):
        super().__init__( representations )

    def estimate( self, query_repr:spmatrix, filtered_docs:list[str] ) -> list[tuple[str,float]]:

        # Get the corresponding corpus representations
        filtered_repr = np.array( [ self._representations[ int( idoc ) ] for idoc in filtered_docs ] ) # type: ignore
        filtered_repr.reshape( len( filtered_docs ), -1 )

        # Compute the similarities
        similarities = compute_similarities0( query_repr, filtered_repr )

        # Put together idocs and similarities
        results = [ ( idoc, round( sim, 4 ) ) for idoc, sim in zip( filtered_docs, similarities ) ]

        # Put greater similarities at the top
        results.sort( key=lambda x: x[1], reverse=True )

        return results


class SentSimilarityEstimator( AbstractSimilarityEstimator ):

    def __init__( self, representations:spmatrix, tags:list[str] ):
        super().__init__( representations )
        self._tags = tags

    def estimate( self, query_repr:spmatrix, filtered_docs:list[str] ) -> list[tuple[str,float]]:

        # Get the corresponding sentence representations
        filtered_repr = []
        filtered_tags = []
        for idoc in filtered_docs:
            doc_sentences = [ isent for isent, tag in zip( range( len( self._tags ) ), self._tags ) if tag.split('.')[0] == str( idoc ) ]
            for j, isent in enumerate( doc_sentences ):
                filtered_repr.append( self._representations[ isent ] ) # type: ignore
                filtered_tags.append( f'{idoc}.{j}' )
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


def similarityEstimatorFactory( option:str ):

    match option:

        case 'arxiv-stemm-single-count':
            from .arXiv.settings import pickle_paths
            vectorizer_descr = 'title-summary_lower-punct-specials-stops-stemm_single_count'
            corpus_repr_filename = f"{pickle_paths[ 'corpus_repr' ]}/{vectorizer_descr}.pkl"
            corpus_repr = PickleLoader( corpus_repr_filename ).load()
            return DocSimilarityEstimator( corpus_repr )

        case 'arxiv-lemm-single-tfidf':
            from .arXiv.settings import pickle_paths
            vectorizer_descr = 'title-summary_lower-punct-specials-stops-lemm_single_tfidf'
            corpus_repr_filename = f"{pickle_paths[ 'corpus_repr' ]}/{vectorizer_descr}.pkl"
            corpus_repr = PickleLoader( corpus_repr_filename ).load()
            return DocSimilarityEstimator( corpus_repr )

        case 'arxiv-jina':
            from .arXiv.Dataset import Dataset
            from .arXiv.settings import pickle_paths
            representations_descr = 'jina-embeddings-sentences'
            representations_filename = f"{pickle_paths[ 'corpus_repr' ]}/{representations_descr}.pkl"
            representations = PickleLoader( representations_filename ).load()
            sentences, tags = Dataset().toSentences()
            return SentSimilarityEstimator( representations, tags )

        case 'medical-lemm-single-tfidf':
            from .medical.settings import pickle_paths
            vectorizer_descr = 'title-summary_lower-punct-specials-stops-lemm_single_tfidf'
            corpus_repr_filename = f"{pickle_paths[ 'corpus_repr' ]}/{vectorizer_descr}.pkl"
            corpus_repr = PickleLoader( corpus_repr_filename ).load()
            return DocSimilarityEstimator( corpus_repr )

        case _:
            raise Exception( 'similarityEstimatorFactory(): No valid option.' )


# RUN: python -m src.DocEstimator [option]
if __name__ == "__main__": 

    option = None
    if len( sys.argv ) >= 2:
        option = sys.argv[ 1 ]

    match option:

        case 'arxiv-stemm-single-count':
            estimator = similarityEstimatorFactory( option )
            query_repr = estimator.representations[ 3 ] # type: ignore
            print( estimator.estimate( query_repr, [ '0', '1', '2', '3', '4' ] ) )

        case 'arxiv-lemm-single-tfidf':
            estimator = similarityEstimatorFactory( option )
            query_repr = estimator.representations[ 3 ] # type: ignore
            print( estimator.estimate( query_repr, [ '0', '1', '2', '3', '4' ] ) )

        case 'arxiv-jina':
            estimator = similarityEstimatorFactory( option )
            query_repr = estimator.representations[ 0 ] # type: ignore
            print( estimator.estimate( query_repr, [ '0', '1', '2', '3', '4' ] ) )

        case 'medical-lemm-single-tfidf':
            estimator = similarityEstimatorFactory( option )
            query_repr = estimator.representations[ 3 ] # type: ignore
            print( estimator.estimate( query_repr, [ '0', '1', '2', '3', '4' ] ) )

        case _:
            raise Exception( 'No valid option.' )

