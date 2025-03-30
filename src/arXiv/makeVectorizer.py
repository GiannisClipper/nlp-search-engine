import time
import pickle
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer # type: ignore

import nltk # type: ignore
nltk.download( 'stopwords' )
nltk.download( 'wordnet' )

from nltk.corpus import stopwords # type: ignore
from nltk.stem import SnowballStemmer # type: ignore
from nltk.stem import WordNetLemmatizer # type: ignore

from Dataset import Dataset
from Preprocessor import Preprocessor
from Preprocessor import LowerConverter, PunctRemover, SpecialsRemover
from Preprocessor import StopwordsRemover, Stemmer, Lemmatizer

print( f'Load corpus...' )
ds = Dataset()
records = ds.toDictList()
corpus = []
for i in range( len( records ) ):
    corpus.append( records[ i ][ 'title' ] + '-' + records[ i ][ 'summary' ] )

print( f'Preprocessing...' )
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

print( f'Make vectorizer...', end=' ' )
start_time = time.time()
vectorizer = TfidfVectorizer( max_features=500000 )
corpus_repr = vectorizer.fit_transform( corpus )
print( f'({round(time.time()-start_time, 1)} secs)' )

vocabulary = vectorizer.get_feature_names_out()
print( 'Vocabulary[::500]:', len( vocabulary ), vocabulary[::500] )
print()

print( 'Save preprocessor...' )
with open( 'pickles/preprocessor.pkl', 'wb' ) as f:
    pickle.dump( preprocessor, f )

print( 'Save vectorizer...' )
with open( 'pickles/vectorizer.pkl', 'wb' ) as f:
    pickle.dump( vectorizer, f )

print( 'Save corpus representations...' )
with open( 'pickles/corpus_repr.pkl', 'wb' ) as f:
    pickle.dump( corpus_repr, f )