import sys
import numpy as np

from gensim.models import KeyedVectors, Word2Vec

import nltk
nltk.download( 'punkt_tab' ) # required by word_tokenize()
from nltk.tokenize import word_tokenize, sent_tokenize

from .settings import pretrained_models
from .helpers.decorators import with_time_counter

from .datasets.medical.Dataset import Dataset, Queries, ResultMetrics
from .datasets.medical.settings import pickle_paths
from .Preprocessor import NaivePreprocessor
from .helpers.Pickle import PickleSaver, PickleLoader
from .helpers.computators import compute_similarities0


def load_pretrained_model():

    @with_time_counter
    def load( message=None, *args, **kwargs ):
        # model = KeyedVectors.load_word2vec_format( 'https://drive.google.com/file/d/1iBepVrIhjmQuJBJYuyDahC_VNqn_SVyG/view?usp=sharing', binary=True )
        model = KeyedVectors.load_word2vec_format( pretrained_models[ 'googlenews' ], binary=True )
        # Get the vector for a specific word
        # vector = model['example']
        # Find most similar words
        # similar_words = model.most_similar( 'database' )
        # print( 'SIMILARS', similar_words )
        # Perform word analogy
        # analogy_result = model.most_similar(positive=['king', 'woman'], negative=['man'])
        # print( 'ANALOGY', analogy_result )
        return model

    print( 'Loading googlenews...' )
    return load()


def retrain_model():

    print( 'RETRAIN MODEL' )

    model = load_pretrained_model()

    @with_time_counter
    def retrain( message=None, *args, **kwargs ):

        # vector_size = size of the word embedding that it would output
        # min_count = minimum frequency for the words to be a part of the model: i.e. it ignores all words with count less than min_count
        # window = maximum distance between the current and predicted word within a sentence
        # workers = threads to train the model, this can be adjusted as per the number of cores your system has
        # iter = number of iterations for training the model
        new_model = Word2Vec( vector_size=300, min_count=1, window=2, workers=4, batch_words=100 )

        print( 'Updating vocabulary...' )
        # build vocab from the original model
        # new_model.build_vocab( [ list( model.key_to_index.keys() ) ], update=False )
        new_model.wv.vectors = model.vectors
        new_model.wv.key_to_index = model.key_to_index
        new_model.wv.index_to_key = model.index_to_key

        print( 'Retraining...' )
        new_sentences = kwargs[ 'new_sentences' ]
        new_sentences = [ word_tokenize( s ) for s in new_sentences[:100] ]
        # add ne words in vocab from new sentences
        new_model.build_vocab( new_sentences, update=True )
        # train on the new sentences
        new_model.train( new_sentences, total_examples=len( new_sentences ), epochs=1 )
        return new_model

    corpus = Dataset().toListTitlesAbstracts()
    sentences = []
    for doc in corpus:
        doc_sentences = sent_tokenize( doc )
        for sentence in doc_sentences:
            sentences.append( sentence )
    sentences = NaivePreprocessor().transform( sentences )

    print( 'Retraining googlenews...' )
    new_model = retrain( new_sentences=sentences )

    print( 'Saving retrained model...' )
    vectorizer_descr = 'retrained-googlenews-sentences'
    vectorizer_filename = f"{pickle_paths[ 'vectorizers' ]}/{vectorizer_descr}.pkl"
    PickleSaver( vectorizer_filename ).save( new_model )


def sentence_to_vec( model, words ):

    vectors = [ model[ w ] for w in words if w in model ]

    if len( vectors ) == 0:
        print( words )
        # return np.zeros( 300 )
        return None

    # How to build a search engine with word embeddings
    # https://dev.to/mage_ai/how-to-build-a-search-engine-with-word-embeddings-56jd
    return np.concatenate( [ np.mean( vectors, axis=0 ), np.amax( vectors, axis=0 ) ] )


def make_vectorizer():

    print( 'MAKE VECTORIZER' )

    model = load_pretrained_model()

    @with_time_counter
    def vectorize_corpus( message=None, *args, **kwargs ):

        corpus = Dataset().toListTitlesAbstracts()

        sentences = []
        tags = []
        for i, doc in enumerate( corpus ):
            some_sentences = sent_tokenize( doc )
            for j, sentence in enumerate( some_sentences ):
                sentences.append( sentence )
                tags.append( f'{i}.{j}' )

        sentences = NaivePreprocessor().transform( sentences )

        sentences_repr = []
        tags_new = []
        for sentence, tag in zip( sentences, tags ):
            vector = sentence_to_vec( model, word_tokenize( sentence ) )
            if not vector is None:
                sentences_repr.append( vector )
                tags_new.append( tag )

        return sentences_repr, tags_new

    print( 'Vectorizing corpus (as sentences)...' )
    sentences, tags = vectorize_corpus()

    print( 'Saving vectorizer...' )
    vectorizer_descr = 'googlenews-sentences'
    vectorizer_filename = f"{pickle_paths[ 'vectorizers' ]}/{vectorizer_descr}.pkl"
    PickleSaver( vectorizer_filename ).save( ( sentences, tags ) )


def ask_queries():

    print( 'ASK QUERIES' )

    model = load_pretrained_model()

    corpus = Dataset().toList()
    queries = Queries().toList()[:10]
    results= []
    for q in queries:

        query_preprocessed = NaivePreprocessor().transform( [ q[ 'query' ] ] )
        query_repr = sentence_to_vec( model, word_tokenize( query_preprocessed[ 0 ] ) )

        if query_repr is None: # e.g. {'id': 'PLAIN-1007', 'query': 'ddt'}
            results.append( None )
            continue
    
        vectorizer_descr = 'googlenews-sentences'
        vectorizer_filename = f"{pickle_paths[ 'vectorizers' ]}/{vectorizer_descr}.pkl"
        sentences_repr, tags = PickleLoader( vectorizer_filename ).load()

        sentences_repr = np.array( sentences_repr )
        sentences_repr.reshape( len( tags ), -1 ) # type: ignore

        # print( q )
        # print( 'query_repr', query_repr )
        # print( 'sentences_repr', sentences_repr )

        # compute the similarities
        similarities = compute_similarities0( query_repr, sentences_repr )

        # put together document ids and similarities
        result = []
        for tag, similarity in zip( tags, similarities ):
            i = int( tag.split('.')[0] )
            result.append( ( corpus[ i ], round( float( similarity ), 4 ) ) ) # type: ignore

        # greater similarities at the top
        result.sort( key=lambda x: x[1], reverse=True )

        result = [ ( r[0]['id'], r[1] ) for r in result ]
        result = result[:150]
        # print( result )

        result = [ r[0] for r in result ]
        results.append( result )

    print( queries, results )
    joined = zip( queries, results )
    joined = [ (q, r) for q, r in joined if not r is None ]
    queries = [ q for q, r in joined ]
    results = [ r for q, r in joined ]
    print()
    print( queries, results )

    print( '\n\n' )
    resultMetrics = ResultMetrics()
    resultMetrics.compute( queries, results )
    resultMetrics.show()

if __name__ == "__main__":

    option = None
    if len( sys.argv ) >= 2:
        option = sys.argv[ 1 ]

    match option:

        case 'retrain-model':
            retrain_model()

        case 'make-vectorizer':
            make_vectorizer()

        case 'ask-queries':
            ask_queries()

        case _:
            raise Exception( 'No valid option.' )
