import pickle
import random
import math

from .Dataset import Dataset
from .Preprocessor import LemmPreprocessor

from .settings import pickle_paths

from .helpers.decorators import with_time_counter
from .helpers.computators import compute_similarities0, compute_similarities1

# +-------------+
# | Load corpus |
# +-------------+

print( f'\nLoading corpus...' )
ds = Dataset()
corpus = ds.toList()

# +-------------------+
# | Init preprocessor |
# +-------------------+

preprocessor = LemmPreprocessor()

# +-----------------+
# | Load vectorizer |
# +-----------------+

vec_descr = 'title-summary_lower-punct-specials-stops-stemm_single_count'
# vec_descr = 'title-summary_lower-punct-specials-stops-lemm_single_count'

print( f'\nLoading vectorizer {vec_descr}...' )
with open( f"{pickle_paths[ 'vectorizers' ]}/{vec_descr}.pkl", 'rb' ) as f:
    vectorizer = pickle.load( f )

vocabulary = vectorizer.get_feature_names_out()
step = len( vocabulary ) // 50 if len( vocabulary ) > 50 else 1 
print( f'\nVocabulary[::{step}]:', len( vocabulary ), vocabulary[::step] )

# +---------------------------+
# | Load the tfidf vectorizer |
# +---------------------------+

# print( f'\nLoading tfidf vectorizer...' )
# with open( pickle_filenames[ 'tfidf' ][ 'vectorizer' ], 'rb' ) as f:
#     vectorizer = pickle.load( f )

# vocabulary = vectorizer.get_feature_names_out()
# print( 'Vocabulary[::500]:', len( vocabulary ), vocabulary[::500] )

# +----------------------------+
# | Load the count corpus repr |
# +----------------------------+

print( f'\nLoad count corpus representations...' )
with open( f"{pickle_paths[ 'corpus_repr' ]}/{vec_descr}.pkl", 'rb' ) as f:
    corpus_repr = pickle.load( f )
corpus_repr = corpus_repr.toarray()
print( f'Dimensions: {corpus_repr.shape}' )

# +----------------------------+
# | Load the tfidf corpus repr |
# +----------------------------+

# print( f'\nLoad tfidf corpus representations...' )
# corpus_repr = None
# with open( pickle_filenames[ 'tfidf' ][ 'corpus_repr' ], 'rb' ) as f:
#     corpus_repr = pickle.load( f )
# corpus_repr = corpus_repr.toarray()
# print( f'Dimensions: {corpus_repr.shape}' )

# +-------------------+
# | Prepare the query |
# +-------------------+

print( '\nPrepare query...' )
# random_record = random.randint( 0, corpus_repr.shape[0]-1 )
# query = corpus_repr[ random_record ]
query = "Available literature about databases (both SQL and NoSQL), especially somehow relevant to semantics?"
query_preprocessed = preprocessor.transform( [ query ] )
query_repr = vectorizer.transform( query_preprocessed )

# +----------------------+
# | Compute similarities |
# +----------------------+

@with_time_counter
def compute_similarities( message=None, *args, **kwargs ):
    similarities = compute_similarities0( query_repr, corpus_repr )

    similarities = [ ( x[0][ 'id' ], x[0][ 'catg_ids' ], x[1] ) for x in zip( corpus, similarities ) ]
    similarities.sort( key=lambda x: x[2], reverse=True )
    return similarities

similarities = compute_similarities( 'Computing similarities...' )
print()
for i in range( 20 ):
    print( similarities[ i ] )

# with open( './tests/test_sim_results_1.pkl', 'wb' ) as f:
#     pickle.dump( [ s[0] for s in similarities ][:10], f )

# +-----------------+
# | Compute metrics |
# +-----------------+

gains = [ s[2] for s in similarities[:20] ]
random.shuffle( gains )

def cumulative_gain( gains:list[float] ):
    counter = 0
    result = []
    for i in range( len( gains ) ):
        counter += gains[ i ]
        result.append( counter )
    return result

cg = cumulative_gain( gains )
# print( 'CG:', cg )

def discounted_cumulative_gain( gains:list[float] ):
    counter = 0.0
    result = []
    for i, rel in enumerate( gains ):
        rank = i + 1  # ranks start at 1
        counter += rel / math.log2( rank + 1 )  # discount factor: log2(rank+1)
        result.append( counter )
    return result

dcg = discounted_cumulative_gain( gains )
# print( 'DCG:', dcg )

def ideal_discounted_cumulative_gain( gains:list[float] ):
    # sort gains in descending order for ideal ranking
    sorted_gains = sorted( gains, reverse=True )
    return discounted_cumulative_gain( sorted_gains )

idcg = ideal_discounted_cumulative_gain( gains )
# print( 'IDCG:', idcg )

def normalized_discounted_cumulative_gain( gains:list[float] ):
    # sort gains in descending order for ideal ranking
    dcg = discounted_cumulative_gain( gains )
    idcg = ideal_discounted_cumulative_gain( gains )
    return [ x[0] / x[1] for x in zip( dcg, idcg ) ]

ndcg = normalized_discounted_cumulative_gain( gains )
# print( 'NDCG:', ndcg )

print()
print( "------", "------", "------", "------", "------","------" )
print( "RANK  ", "REL   ", "CG    ", "DCG   ", "IDCG  ","NDCG  " )
print( "------", "------", "------", "------", "------","------" )
for i, g0, g1, g2, g3, g4 in zip( range(1, len(cg)+1), gains, cg, dcg, idcg, ndcg ):
    print( f' {i:3} {g0:6.2} {g1:6.2} {g2:6.2} {g3:6.2} {g4:6.2}')
