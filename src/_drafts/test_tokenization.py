import nltk
nltk.download( 'stopwords' )
nltk.download( 'wordnet' )
nltk.download( 'punkt_tab' ) # required by word_tokenize()

# from nltk.corpus import stopwords
# from nltk.stem import SnowballStemmer
# from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import ngrams

from arXiv.Dataset import Dataset
from arXiv.Preprocessor import Preprocessor
from arXiv.Preprocessor import LowerConverter, PunctRemover, SpecialsRemover
from arXiv.Preprocessor import StopwordsRemover, Stemmer, Lemmatizer

# load the dataset

print( f'Load corpus...' )
ds = Dataset()
records = ds.toList()
corpus = []
for i in range( len( records ) ):
    corpus.append( records[ i ][ 'title' ] + '-' + records[ i ][ 'summary' ] )

# perform the preprocessing

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
# print( 'corpus[::1000]:', corpus[::1000] )
# print()

# extract words

tokens = tuple( word_tokenize( ' '.join( corpus ) ) )
singles = {}
for token in tokens:
    singles[ token ] = singles.get( token, 0 ) + 1
singles = list( singles.items() )
singles.sort( key=lambda x: x[1], reverse=True )

print( '\nNumber of single words:', len( singles ), singles[:100], '...' )

# extract 2grams

tokens2 = tuple( x + ' ' + y for x, y in ngrams( tokens, 2 ) )
twograms = {}
for token in tokens2:
    twograms[ token ] = twograms.get( token, 0 ) + 1
twograms = list( twograms.items() )
twograms.sort( key=lambda x: x[1], reverse=True )

print( '\nNumber of 2grams:', len( twograms ), twograms[:100], '...' )

