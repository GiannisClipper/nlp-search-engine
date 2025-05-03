from gensim.models.doc2vec import Doc2Vec, TaggedDocument

import nltk
nltk.download( 'punkt_tab' ) # required by word_tokenize()
from nltk.tokenize import word_tokenize, sent_tokenize

import numpy as np

from .settings import pretrained_models
from .datasets.medical.Dataset import Dataset, Queries, ResultMetrics
from .datasets.medical.settings import pickle_paths
from .Preprocessor import LemmPreprocessor, NaivePreprocessor
from .TermsFilter import TermsFilter

from .helpers.Pickle import PickleLoader
from .helpers.computators import compute_similarities0

corpus = Dataset().toListTitlesAbstracts()
# all_sentences = []
# tags = []
# for i, doc in enumerate( corpus ):
#     sentences = sent_tokenize( doc )
#     for j, sentence in enumerate( sentences ):
#         all_sentences.append( sentence )
#         tags.append( f'{i}.{j}' )
# corpus = all_sentences
corpus = LemmPreprocessor().transform( corpus )

documents = [ TaggedDocument( words=word_tokenize( doc ), tags=[ str(i)] ) for i, doc in enumerate( corpus ) ]
# documents = [ TaggedDocument( words=word_tokenize( doc ), tags=[ tag ] ) for doc, tag in zip( corpus, tags ) ]

# Initialize the Doc2Vec model
model = Doc2Vec(vector_size=250,  # Dimensionality of the document vectors
                window=4,         # Maximum distance between the current and predicted word within a sentence
                min_count=1,      # Ignores all words with total frequency lower than this
                workers=4,        # Number of CPU cores to use for training
                epochs=20)        # Number of training epochs

# Build the vocabulary
model.build_vocab(documents)

# Train the model
model.train(documents, total_examples=model.corpus_count, epochs=model.epochs)

# Infer a vector for a new document
# inferred_vector = model.infer_vector(['your', 'new', 'document', 'text'])
# print( 'inferred_vector', inferred_vector )

# Get the vector for an existing document by tag
# existing_document_vector = model.dv[5]
# print( 'existing_document_vector', existing_document_vector)

# -------------------------------------------------------------------------
# q = Queries().toList()[5]
# query_preprocessed = LemmPreprocessor().transform( [ q[ 'query' ] ] )
# query_terms = tuple( word_tokenize( query_preprocessed[ 0 ] ) )
# query_repr = model.infer_vector( query_terms )

# filtered_corpus_repr = np.array( [ model.dv['5'] ] )
# filtered_corpus_repr.reshape( 1, -1 )

# similarities = compute_similarities0( query_repr, filtered_corpus_repr )
# print( similarities )
# print( model.dv.most_similar('5') )
# exit()
# -------------------------------------------------------------------------

# q = Queries().toDict()[ 'PLAIN-1' ]
queries = Queries().toList()[:10]
results= []
for q in queries:

    query_preprocessed = LemmPreprocessor().transform( [ q[ 'query' ] ] )
    query_terms = tuple( word_tokenize( query_preprocessed[ 0 ] ) )
    query_repr = model.infer_vector( query_terms )

    # -------------------------------------------------------------------------
    # similars = model.dv.most_similar( [ query_repr ], topn=50 )
    # print( similars )
    # result = []
    # corpus = Dataset().toList()
    # for tag, similarity in similars:
    #     i, j = tag.split( '.' )
    #     result.append( ( corpus[ int(i) ], round( float( similarity ), 4 ) ) ) # type: ignore
    # result = [ ( r[0]['id'], r[1] ) for r in result ]
    # result = [ r[0] for r in result ]
    # results.append( result )
    # break
    # -------------------------------------------------------------------------

    # greater similarities at the top
    # result.sort( key=lambda x: x[1], reverse=True )

    # result = [ ( r[0]['id'], r[1] ) for r in result ]
    # result = result[:50]
    # # print( result )

    # result = [ r[0] for r in result ]
    # results.append( result )



    corpus = Dataset().toList()
    index_descr = 'title-summary_lower-punct-specials-stops-lemm_single'
    index_filename = f"{pickle_paths[ 'indexes' ]}/{index_descr}.pkl"
    index = PickleLoader( index_filename ).load()
    doc_selection = TermsFilter( corpus, index ).filter( query_terms )
    # doc_selection = [ i for i, _ in enumerate( corpus ) ]
    if len( doc_selection ) == 0:
        results.append( [] )
        continue

    filtered_corpus_repr = np.array( [ model.dv[ str(key) ] for key in doc_selection ] )
    filtered_corpus_repr.reshape( len( doc_selection ), -1 )

    # compute the similarities
    similarities = compute_similarities0( query_repr, filtered_corpus_repr )

    # put together document ids and similarities
    result = []
    for key, similarity in zip( doc_selection, similarities ):
        result.append( ( corpus[ key ], round( float( similarity ), 4 ) ) ) # type: ignore

    # greater similarities at the top
    result.sort( key=lambda x: x[1], reverse=True )

    result = [ ( r[0]['id'], r[1] ) for r in result ]
    result = result[:150]
    # print( result )

    result = [ r[0] for r in result ]
    results.append( result )

print( '\n\n' )
resultMetrics = ResultMetrics()
resultMetrics.compute( queries, results )
resultMetrics.show()
