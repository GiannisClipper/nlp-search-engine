from arXiv.Dataset import Dataset
from arXiv.Preprocessor import Preprocessor
from arXiv.Preprocessor import LowerConverter, PunctRemover, SpecialsRemover
from arXiv.Preprocessor import StopwordsRemover, Stemmer, Lemmatizer
from arXiv.Tokenizer import SingleTokenizer, TwogrammTokenizer

# load the dataset

print( f'Load corpus...' )
ds = Dataset()
records = ds.toDictList()
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

singles = SingleTokenizer( ' '.join( corpus ) ).tokens
print( '\nNumber of single words:', len( singles ), singles[:100], '...' )

# extract 2grams

twograms = TwogrammTokenizer( ' '.join( corpus ) ).tokens
print( '\nNumber of 2grams:', len( twograms ), twograms[:100], '...' )

