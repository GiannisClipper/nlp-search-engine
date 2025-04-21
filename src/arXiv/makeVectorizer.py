import pickle
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from .Dataset import Dataset
from .Preprocessor import Preprocessor
from .Preprocessor import LowerConverter, PunctRemover, SpecialsRemover
from .Preprocessor import StopwordsRemover, Stemmer, Lemmatizer
from .Tokenizer import SingleTokenizer
from .settings import pickle_filenames

from .helpers.decorators import with_time_counter

# +------------------+
# | Load the dataset |
# +------------------+

print( f'\nLoad corpus...' )
ds = Dataset()
records = ds.toDictList()
corpus = []
for i in range( len( records ) ):
    corpus.append( records[ i ][ 'title' ] + '-' + records[ i ][ 'summary' ] )

# +---------------------------+
# | Perform the preprocessing |
# +---------------------------+

print( f'\nPreprocessing...' )
preprocessor = Preprocessor( [
    LowerConverter(), 
    PunctRemover(), 
    SpecialsRemover(),
    StopwordsRemover(), 
    Stemmer(),
    # Lemmatizer()
] )

corpus = preprocessor.transform( corpus )
print( 'corpus[::1000]:', corpus[::1000] )
print()

print( 'Saving preprocessor in disk...' )
with open( pickle_filenames[ 'preprocessor' ], 'wb' ) as f:
    pickle.dump( preprocessor, f )

# +-----------------------+
# | Create the vocabulary |
# +-----------------------+

@with_time_counter
def create_vocabulary( message=None, *args, **kwargs ):
    return SingleTokenizer( ' '.join( corpus ) ).tokens

vocabulary = create_vocabulary( '\nCreating vocabulary...' )

print( '\nVocabulary[::500]:', len( vocabulary ), vocabulary[::500] )

print( 'Saving vocabulary in disk...' )
with open( pickle_filenames[ 'vocabulary' ], 'wb' ) as f:
    pickle.dump( vocabulary, f )

# +---------------------------+
# | Create a count vectorizer |
# +---------------------------+

@with_time_counter
def create_count_vectorizer( message=None, *args, **kwargs ):
    vectorizer = CountVectorizer( vocabulary=vocabulary )
    corpus_repr = vectorizer.fit_transform( corpus )
    return vectorizer, corpus_repr

vectorizer, corpus_repr = create_count_vectorizer( '\nCreating count vectorizer...' )

print( '\nSaving count vectorizer in disk...' )
with open( pickle_filenames[ 'count' ][ 'vectorizer' ], 'wb' ) as f:
    pickle.dump( vectorizer, f )

print( 'Saving corpus representations in disk...' )
with open( pickle_filenames[ 'count' ][ 'corpus_repr' ], 'wb' ) as f:
    pickle.dump( corpus_repr, f )

# +---------------------------+
# | Create a tfidf vectorizer |
# +---------------------------+

@with_time_counter
def create_tfidf_vectorizer( message=None, *args, **kwargs ):
    # vectorizer = TfidfVectorizer( max_features=5000 )
    vectorizer = TfidfVectorizer( vocabulary=vocabulary )
    corpus_repr = vectorizer.fit_transform( corpus )
    return vectorizer, corpus_repr

vectorizer, corpus_repr = create_tfidf_vectorizer( '\nCreating tfidf vectorizer...' )

print( '\nSaving tfidf vectorizer in disk...' )
with open( pickle_filenames[ 'tfidf' ][ 'vectorizer' ], 'wb' ) as f:
    pickle.dump( vectorizer, f )

print( 'Saving corpus representations in disk...' )
with open( pickle_filenames[ 'tfidf' ][ 'corpus_repr' ], 'wb' ) as f:
    pickle.dump( corpus_repr, f )

# extract the vocabulary
# vocabulary = vectorizer.get_feature_names_out()
# print( 'Vocabulary[::500]:', len( vocabulary ), vocabulary[::500] )
# print()
