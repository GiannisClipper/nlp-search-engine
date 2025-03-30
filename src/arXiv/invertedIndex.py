import pickle
import time
import pytrie

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

def build_trie_index( vocabulary, corpus_repr ):

    trie = pytrie.StringTrie()

    for i, record_repr in enumerate( corpus_repr ):

        for j in range( len( record_repr ) ):

          if record_repr[ j ] != 0:
            if not vocabulary[ j ] in trie:
                trie[ vocabulary[ j ] ] = []
            trie[ vocabulary[ j ] ].append( i ) # type: ignore

    return trie

print( 'Build trie index...', end=' ' )
start_time = time.time()
trie = build_trie_index( vocabulary, corpus_repr )
print( f'({round(time.time()-start_time, 1)} secs)' )

print()
for term, doc_indexes in trie.items()[:10]:
    print( term, len( doc_indexes ) ) # type: ignore

arr = [ ( x[0], len( x[1] ) ) for x in trie.items() ] # type: ignore
arr.sort( key=lambda x: x[1], reverse=True )

print()
for term, occurences in arr[:20]:
    print( term, occurences ) # type: ignore

