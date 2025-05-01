import sys

import numpy as np
from scipy.sparse import spmatrix
from .helpers.Pickle import PickleLoader
from .helpers.computators import compute_similarities0

class DocEstimator:

    def __init__( self, corpus_repr:spmatrix ):
        self._corpus_repr = corpus_repr

    @property
    def corpus_repr( self ):
        return self._corpus_repr

    def estimate( self, query_repr:spmatrix, filtered_docs:list[str] ) -> list[tuple[str,float]]:

        # get the corresponding corpus representations
        filtered_corpus_repr = np.array( [ self._corpus_repr[ int( idoc ) ] for idoc in filtered_docs ] ) # type: ignore
        filtered_corpus_repr.reshape( len( filtered_docs ), -1 )

        # compute the similarities
        similarities = compute_similarities0( query_repr, filtered_corpus_repr )

        # put together idocs and similarities
        results = [ ( idoc, round( sim, 4 ) ) for idoc, sim in zip( filtered_docs, similarities ) ]

        # put greater similarities at the top
        results.sort( key=lambda x: x[1], reverse=True )

        return results


def docEstimatorFactory( option:str ):

    match option:

        case 'arxiv-stemm-single-count':
            from .arXiv.settings import pickle_paths
            vectorizer_descr = 'title-summary_lower-punct-specials-stops-stemm_single_count'
            corpus_repr_filename = f"{pickle_paths[ 'corpus_repr' ]}/{vectorizer_descr}.pkl"
            corpus_repr = PickleLoader( corpus_repr_filename ).load()
            return DocEstimator( corpus_repr )

        case 'arxiv-lemm-single-tfidf':
            from .arXiv.settings import pickle_paths
            vectorizer_descr = 'title-summary_lower-punct-specials-stops-lemm_single_tfidf'
            corpus_repr_filename = f"{pickle_paths[ 'corpus_repr' ]}/{vectorizer_descr}.pkl"
            corpus_repr = PickleLoader( corpus_repr_filename ).load()
            return DocEstimator( corpus_repr )

        case 'medical-lemm-single-tfidf':
            from .medical.settings import pickle_paths
            vectorizer_descr = 'title-summary_lower-punct-specials-stops-lemm_single_tfidf'
            corpus_repr_filename = f"{pickle_paths[ 'corpus_repr' ]}/{vectorizer_descr}.pkl"
            corpus_repr = PickleLoader( corpus_repr_filename ).load()
            return DocEstimator( corpus_repr )

        case _:
            raise Exception( 'docEstimatorFactory(): No valid option.' )


# RUN: python -m src.DocEstimator [option]
if __name__ == "__main__": 

    option = None
    if len( sys.argv ) >= 2:
        option = sys.argv[ 1 ]

    match option:

        case 'arxiv-stemm-single-count':
            estimator = docEstimatorFactory( option )
            query_repr = estimator.corpus_repr[ 3 ] # type: ignore
            print( estimator.estimate( query_repr, [ '0', '1', '2', '3', '4' ] ) )

        case 'arxiv-lemm-single-tfidf':
            estimator = docEstimatorFactory( option )
            query_repr = estimator.corpus_repr[ 3 ] # type: ignore
            print( estimator.estimate( query_repr, [ '0', '1', '2', '3', '4' ] ) )

        case 'medical-lemm-single-tfidf':
            estimator = docEstimatorFactory( option )
            query_repr = estimator.corpus_repr[ 3 ] # type: ignore
            print( estimator.estimate( query_repr, [ '0', '1', '2', '3', '4' ] ) )

        case _:
            raise Exception( 'No valid option.' )

