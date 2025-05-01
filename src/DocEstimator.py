import sys

import numpy as np
from scipy.sparse import spmatrix
from .helpers.Pickle import PickleLoader
from .helpers.computators import compute_similarities0

class DocEstimator:

    def __init__( self, query_repr:spmatrix, corpus_repr:spmatrix ):
        self._query_repr = query_repr
        self._corpus_repr = corpus_repr

    def estimate( self, filtered_docs:list[str] ) -> list[tuple[str,float]]:

        # get the corresponding corpus representations
        filtered_corpus_repr = np.array( [ self._corpus_repr[ int( idoc ) ] for idoc in filtered_docs ] ) # type: ignore
        filtered_corpus_repr.reshape( len( filtered_docs ), -1 )

        # compute the similarities
        similarities = compute_similarities0( self._query_repr, filtered_corpus_repr )

        # put together idocs and similarities
        results = [ ( idoc, round( sim, 4 ) ) for idoc, sim in zip( filtered_docs, similarities ) ]

        # put greater similarities at the top
        results.sort( key=lambda x: x[1], reverse=True )

        return results


# RUN: python -m src.DocEstimator [option]
if __name__ == "__main__": 

    from .arXiv.Dataset import Dataset
    from .arXiv.settings import pickle_paths

    option = None
    if len( sys.argv ) >= 2:
        option = sys.argv[ 1 ]

    match option:

        case 'arxiv-stemm-single-count':
            from .arXiv.settings import pickle_paths
            vectorizer_descr = 'title-summary_lower-punct-specials-stops-stemm_single_count'
            corpus_repr_filename = f"{pickle_paths[ 'corpus_repr' ]}/{vectorizer_descr}.pkl"
            corpus_repr = PickleLoader( corpus_repr_filename ).load()
            estimator = DocEstimator( corpus_repr[ 3 ], corpus_repr )
            print( estimator.estimate( [ '0', '1', '2', '3', '4' ] ) )

        case 'arxiv-lemm-single-tfidf':
            from .arXiv.settings import pickle_paths
            vectorizer_descr = 'title-summary_lower-punct-specials-stops-lemm_single_tfidf'
            corpus_repr_filename = f"{pickle_paths[ 'corpus_repr' ]}/{vectorizer_descr}.pkl"
            corpus_repr = PickleLoader( corpus_repr_filename ).load()
            estimator = DocEstimator( corpus_repr[ 3 ], corpus_repr )
            print( estimator.estimate( [ '0', '1', '2', '3', '4' ] ) )

        case 'medical-lemm-single-tfidf':
            from .medical.settings import pickle_paths
            vectorizer_descr = 'title-summary_lower-punct-specials-stops-lemm_single_tfidf'
            corpus_repr_filename = f"{pickle_paths[ 'corpus_repr' ]}/{vectorizer_descr}.pkl"
            corpus_repr = PickleLoader( corpus_repr_filename ).load()
            estimator = DocEstimator( corpus_repr[ 3 ], corpus_repr )
            print( estimator.estimate( [ '0', '1', '2', '3', '4' ] ) )

        case _:
            raise Exception( 'No valid option.' )

