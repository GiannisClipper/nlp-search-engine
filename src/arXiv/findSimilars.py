import pickle
import time
import random
import math
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity # type: ignore

from Dataset import Dataset

print( f'\nLoad corpus...' )
ds = Dataset()
corpus = ds.toDictList()

print( f'\nLoad preprocessor...' )
preprocessor = None
with open( 'pickles/preprocessor.pkl', 'rb' ) as f:
    preprocessor = pickle.load( f )
print( f'Transformers: {str( preprocessor )}' )

print( f'\nLoad vectorizer...' )
vectorizer = None
with open( 'pickles/vectorizer.pkl', 'rb' ) as f:
    vectorizer = pickle.load( f )

vocabulary = vectorizer.get_feature_names_out()
print( 'Vocabulary[::500]:', len( vocabulary ), vocabulary[::500] )

print( f'\nLoad corpus representations...' )
corpus_repr = None
with open( 'pickles/corpus_repr.pkl', 'rb' ) as f:
    corpus_repr = pickle.load( f )
corpus_repr = corpus_repr.toarray()
print( f'Dimensions: {corpus_repr.shape}' )

print( '\nCompute similarities...', end=' ' )
def compute_similarities( single_repr, corpus_repr ):
    similarities = []
    for i in range( corpus_repr.shape[ 0 ] ):
        sim = cosine_similarity( single_repr.reshape(1,-1), corpus_repr[i].reshape(1,-1) )
        similarities.append( sim[0][0] )
    return similarities

# about 3 times slower: ~8 secs (compute_similarities) vs ~24 secs (compute_similarities2)
# def compute_similarities2( single_repr, corpus_repr ):

#     similarities = []

#     # keep in dict only pos with non zero values
#     single_repr_d = dict( [ ( i, r ) for i, r in enumerate( single_repr ) if r != 0 ] )
#     for i in range( corpus_repr.shape[ 0 ] ):

#         # keep in dict only pos with non zero values
#         text_repr_d = dict( [ ( i, r ) for i, r in enumerate( corpus_repr[i] ) if r != 0 ] )

#         # find the common positions
#         union = set( list( single_repr_d.keys() ) + list( text_repr_d.keys() ) )

#         # get the values from the common positions only
#         x = np.array( [ single_repr[ j ] for j in union ] ).reshape(1,-1)
#         y = np.array([ corpus_repr[ i ][ j ] for j in union ] ).reshape(1,-1)

#         sim = cosine_similarity( x, y )
#         similarities.append( sim[0][0] )

#     return similarities

random_record = random.randint( 0, corpus_repr.shape[0]-1 )
start_time = time.time()
random_repr = corpus_repr[ random_record ]

random_repr = preprocessor.transform( [ "Literature about databases (both SQL and NoSQL) and if they are somehow related to semantics?" ] )
random_repr = vectorizer.transform( random_repr )

similarities = compute_similarities( random_repr, corpus_repr )
print( f'({round(time.time()-start_time, 1)} secs)' )

similarities = [ ( x[0][ 'id' ], x[0][ 'catg_ids' ], x[1] ) for x in zip( corpus, similarities ) ]
# print( random_record, similarities[ random_record ] )
similarities.sort( key=lambda x: x[2], reverse=True )
for i in range( 20 ):
    print( similarities[ i ] )

# #####################################################

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
