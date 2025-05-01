import sys
import os
import time
import numpy as np

from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

import nltk
nltk.download( 'punkt_tab' ) # required by word_tokenize()
from nltk.tokenize import word_tokenize, sent_tokenize

from .settings import pretrained_models
from .helpers.decorators import with_time_counter

from .medical.Dataset import Dataset, Queries, QueriesResults, ResultMetrics
from .medical.settings import pickle_paths
from .Preprocessor import NaivePreprocessor, LemmPreprocessor
from .helpers.Pickle import PickleSaver, PickleLoader
from .TermsFilter import TermsFilter
from .helpers.computators import compute_similarities0

def load_pretrained_model():

    @with_time_counter
    def load( message=None, *args, **kwargs ):
        model = SentenceTransformer( "jinaai/jina-embeddings-v2-base-en", trust_remote_code=True, local_files_only=True )
        model.max_seq_length = 1024 # control your input sequence length up to 8192
        return model

    print( 'Loading jina-embeddings model...' )
    return load()

def load_sentence_embeddings():

    @with_time_counter
    def load( message=None, *args, **kwargs ):
        vectorizer_descr = 'jina-embeddings-sentences'
        filename = f"{pickle_paths[ 'corpus_repr' ]}/{vectorizer_descr}.pkl"
        embeddings = PickleLoader( filename ).load()
        return embeddings

    print( 'Loading sentence embeddings...' )
    return load()

def make_vectorizers():

    print( 'MAKE VECTORIZERS' )

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

        sentences = LemmPreprocessor().transform( sentences )
        print( 'Total sentences:', len( sentences ) )

        vectorizer_descr = 'jina-embeddings-sentences'
        for i in range( 0, len( sentences ), 1000 ):
            vectorizer_filename = f"{pickle_paths[ 'corpus_repr' ]}/{vectorizer_descr}-{i}-{i+1000-1}.pkl"
            if os.path.exists( vectorizer_filename ):
                print( f'{vectorizer_filename} already exists.' )
                continue

            sentences_repr = []
            for j in range( i, min( i+1000, len( sentences ) ), 100 ):
                print( 'Encoding:', j, '-', j+100-1 )
                sentences_repr.append( model.encode( sentences[j:j+100] ) ) # embeddings
            sentences_repr = np.concatenate( sentences_repr )
            PickleSaver( vectorizer_filename ).save( sentences_repr )

            # Delay to handle machine's temperature
            if i + 1000 < len( sentences ): time.sleep( 60 )


        return sentences_repr, tags

    print( 'Vectorizing corpus (as sentences)...' )
    sentences_repr, tags = vectorize_corpus()

    print( 'Saving vectorizer...' )

def merge_vectorizers():

    print( 'MERGE VECTORIZERS' )

    print( 'Counting sentences...' )
    corpus = Dataset().toListTitlesAbstracts()
    counter = 0
    for i, doc in enumerate( corpus ):
        some_sentences = sent_tokenize( doc )
        counter += len( some_sentences )

    print( 'Reading vectorizers...' )
    vectorizer_descr = 'jina-embeddings-sentences'
    vectorizers = []
    for i in range( 0, counter, 1000 ):
        filename = f"{pickle_paths[ 'corpus_repr' ]}/{vectorizer_descr}-{i}-{i+1000-1}.pkl"
        vectorizers.append( PickleLoader( filename ).load() )

    print( 'Saving...' )
    filename = f"{pickle_paths[ 'corpus_repr' ]}/{vectorizer_descr}.pkl"
    PickleSaver( filename ).save( np.concatenate( vectorizers ) )

def read_merged():
    vectorizer_descr = 'jina-embeddings-sentences'
    filename = f"{pickle_paths[ 'corpus_repr' ]}/{vectorizer_descr}.pkl"
    embeddings = PickleLoader( filename ).load()
    print( type(embeddings) )
    print( embeddings.shape )
    print( type(embeddings[0]) )
    print( embeddings[0] )

def ask_queries():

    print( 'ASK QUERIES' )

    model = load_pretrained_model()
    embeddings = load_sentence_embeddings()

    # init sentences
    # --------------
    sentences, tags = Dataset().toSentences()
    sentences = LemmPreprocessor().transform( sentences )
    # REMARK: Same results either using origin dentences or preprocessed
    # it was checked against the 1st and 3rd queries.    

    queries = Queries().toList()[:10]
    results= []
    for q in queries:

        @with_time_counter
        def ask( message=None, *args, **kwargs ):

            # prepare query
            # ------------- 
            query_preprocessed = LemmPreprocessor().transform( [ q[ 'query' ] ] )[ 0 ]
            query_terms = tuple( word_tokenize( query_preprocessed ) )

            # query_repr = model.encode( [ q[ 'query' ] ] )
            query_repr = model.encode( [ query_preprocessed ] )
            # REMARK: Same results either using origin dentences or preprocessed
            # it was checked against the 1st and 3rd queries.    

            # select docs
            # ----------- 
            corpus = Dataset().toList()
            index_descr = 'title-summary_lower-punct-specials-stops-lemm_single'
            index_filename = f"{pickle_paths[ 'indexes' ]}/{index_descr}.pkl"
            index = PickleLoader( index_filename ).load()
            doc_selection = TermsFilter( corpus, index ).filter( query_terms )
            # print( 'Selected docs:', len( doc_selection ), doc_selection )

            #-------------------------------------------------------------
            # CHECK HOW MANY SELECTED DOCS MATCH TO QUERY RESULTS
            #-------------------------------------------------------------
            y = set( QueriesResults().toDict()[ q[ 'id' ] ] )
            y_hat = set( [ corpus[int(i)]['id'] for i in doc_selection ] )
            tp = len( y & y_hat )  # intersection of correct and retrieved
            fp = len( y_hat - y )  # retrieved but not correct
            fn = len( y - y_hat )  # correct but not retrieved
            print( f'selected:{len(y_hat)}, tp:{tp}, fp:{fp}, fn:{fn}')
            # print( 'y:', y )
            # print( 'y_hat:', y_hat )
            # exit()
            #-------------------------------------------------------------

            if len( doc_selection ) == 0:
                # results.append( [] )
                return []

            # get sentences' embeddings
            # -------------------------
            filtered_repr = []
            filtered_tags = []
            for idoc in doc_selection:
                doc_sentences = [ isent for isent, tag in zip( range( len( sentences ) ), tags ) if tag.split('.')[0] == str( idoc ) ]
                # print( f'Doc:{idoc}, Sentences:{len(doc_sentences)}' )
                for j, isent in enumerate( doc_sentences ):
                    filtered_repr.append( embeddings[ isent ] )
                    filtered_tags.append( f'{idoc}.{j}' )
            filtered_repr = np.array( filtered_repr )

            # compute similarities
            # --------------------
            similarities = compute_similarities0( np.array( query_repr ), filtered_repr )

            # aggregate similarities (per document)
            # -------------------------------------
            similarities = compute_similarities0( np.array( query_repr ), filtered_repr )
            doc_result = {}
            for tag, similarity in zip( filtered_tags, similarities ):
                i = int( tag.split('.')[0] )
                if not i in doc_result:
                    doc_result[ i ] = similarity
                elif doc_result[ i ] < similarity:
                    doc_result[ i ] = similarity

            # put together documents and similarities
            # ---------------------------------------
            result = []
            for i, similarity in doc_result.items():
                result.append( ( corpus[ i ], round( float( similarity ), 4 ) ) )

            # greater similarities at the top
            result.sort( key=lambda x: x[1], reverse=True )

            result = [ ( r[0]['id'], r[1] ) for r in result ]
            result = result[:10]
            # print( result )

            result = [ r[0] for r in result ]
            return result
        
        print( 'QUERY:', q ) 
        result = ask()
        # if result is None:
        #     continue
        results.append( result )

    # compute and show metrics
    # ------------------------
    print( '\n\n' )
    resultMetrics = ResultMetrics()
    resultMetrics.compute( queries, results )
    resultMetrics.show()


if __name__ == "__main__":

    option = None
    if len( sys.argv ) >= 2:
        option = sys.argv[ 1 ]

    match option:

        case 'make-vectorizers':
            make_vectorizers()

        case 'merge-vectorizers':
            merge_vectorizers()

        case 'read-merged':
            read_merged()

        case 'ask-queries':
            ask_queries()

        case _:
            raise Exception( 'No valid option.' )
