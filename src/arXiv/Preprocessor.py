import re
import time
from abc import ABC, abstractmethod

import nltk # type: ignore
nltk.download( 'stopwords' )
nltk.download( 'wordnet' )

from nltk.corpus import stopwords # type: ignore
from nltk.stem import SnowballStemmer # type: ignore
from nltk.stem import WordNetLemmatizer # type: ignore

# abstract class 
class Transformer( ABC ):
    def __init__( self ):
        pass

    @abstractmethod
    def __call__( self , corpus:list[str] ) -> list[str]:
        pass

    def __str__( self ):
        return self.__class__.__name__

# converts into lowercase
class LowerConverter( Transformer ):
    def __call__( self , corpus:list[str] ) -> list[str]:
        return [ text.lower() for text in corpus ]

# removes punctuation marks
class PunctRemover( Transformer ):
    def __call__( self , corpus:list[str] ) -> list[str]:
        return [ re.sub(r"[.,;:!\?\"'`]", " ", text ) for text in corpus ]

# removes special characters
class SpecialsRemover( Transformer ):
    def __call__( self , corpus:list[str] ) -> list[str]:
        return [ re.sub(r"[@#\$%^&*\(\)\\/\+-_=\[\]\{\}<>]", " ", text ) for text in corpus ]

# removes stopwords
class StopwordsRemover( Transformer ):
    def __call__( self , corpus:list[str] ) -> list[str]:
        stop_words = stopwords.words( "english" )
        return [ ' '.join( word for word in text.split() if word not in stop_words ) for text in corpus ]

# applies stemming
class Stemmer( Transformer ):
    def __call__( self , corpus:list[str] ) -> list[str]:
        stemmer = SnowballStemmer( 'english' )
        return [ ' '.join( stemmer.stem( word ) for word in text.split() ) for text in corpus ]

# applies lemmatization
class Lemmatizer( Transformer ):
    def __call__( self , corpus:list[str] ) -> list[str]:
        wnl = WordNetLemmatizer()
        return [ ' '.join( wnl.lemmatize( word, "v" ) for word in text.split() ) for text in corpus ]

# performs preprocessing
class Preprocessor:

    def __init__( self, transformers:list[Transformer] ):
        self._transformers = transformers
    
    def transform( self, corpus:list[str] ) -> list[str]:

        for transformer in self._transformers:
            start_time = time.time()
            print( f'Preprocess {transformer.__class__.__name__}...', end=' ' )
            corpus = transformer( corpus )
            end_time = time.time()
            print( f'({round(end_time-start_time, 1)} secs)' )

        return corpus

    def __str__( self ):
        # return ', '.join( [ t.__class__.__name__ for t in self._transformers ] )
        return ', '.join( [ str( t ) for t in self._transformers ] )
         
