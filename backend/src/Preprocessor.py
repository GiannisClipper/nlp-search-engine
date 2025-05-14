import re
import time
from abc import ABC, abstractmethod

import nltk
nltk.download( 'stopwords' )
nltk.download( 'wordnet' )

from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.stem import WordNetLemmatizer

# abstract class 
class Transformation( ABC ):
    def __init__( self ):
        pass

    @abstractmethod
    def __call__( self, corpus:list[str] ) -> list[str]:
        pass

    def __str__( self ):
        return self.__class__.__name__

# converts into lowercase
class LowerConverter( Transformation ):
    def __call__( self, corpus:list[str] ) -> list[str]:
        return [ text.lower() for text in corpus ]

# removes punctuation marks
class PunctRemover( Transformation ):
    def __call__( self, corpus:list[str] ) -> list[str]:
        return [ re.sub(r"[.,;:!\?\"'`]", " ", text ) for text in corpus ]

# removes special characters
class SpecialsRemover( Transformation ):
    def __call__( self, corpus:list[str] ) -> list[str]:
        return [ re.sub(r"[@#\$%^&*\(\)\\/\+-_=\[\]\{\}<>]", " ", text ) for text in corpus ]

# removes stopwords
class StopwordsRemover( Transformation ):
    def __call__( self, corpus:list[str] ) -> list[str]:
        stop_words = stopwords.words( "english" )
        return [ ' '.join( word for word in text.split() if word not in stop_words ) for text in corpus ]

# applies stemming
class Stemmer( Transformation ):
    def __call__( self, corpus:list[str] ) -> list[str]:
        stemmer = SnowballStemmer( 'english' )
        return [ ' '.join( stemmer.stem( word ) for word in text.split() ) for text in corpus ]

# applies lemmatization
class Lemmatizer( Transformation ):
    def __call__( self, corpus:list[str] ) -> list[str]:
        wnl = WordNetLemmatizer()
        return [ ' '.join( wnl.lemmatize( word, "v" ) for word in text.split() ) for text in corpus ]

# performs preprocessing
class Preprocessor:

    def __init__( self, transformations:list[Transformation] ):
        self._transformations = transformations
    
    def transform( self, corpus:list[str] ) -> list[str]:

        for transformation in self._transformations:
            start_time = time.time()
            print( f'Preprocess {transformation.__class__.__name__}...', end=' ' )
            corpus = transformation( corpus )
            end_time = time.time()
            print( f'({round(end_time-start_time, 1)} secs)' )

        return corpus

    def __str__( self ):
        # return ', '.join( [ t.__class__.__name__ for t in self._transformations ] )
        return ', '.join( [ str( t ) for t in self._transformations ] )


class LowerWordsPreprocessor( Preprocessor ):

    def __init__( self ):
        super().__init__( [
            LowerConverter(),
            PunctRemover(), 
        ] )


class NaivePreprocessor( Preprocessor ):

    def __init__( self ):
        super().__init__( [
            LowerConverter(),
            PunctRemover(), 
            SpecialsRemover(),
            StopwordsRemover(), 
        ] )


class StemmPreprocessor( Preprocessor ):

    def __init__( self ):
        super().__init__( [
            LowerConverter(), 
            PunctRemover(), 
            SpecialsRemover(),
            StopwordsRemover(), 
            Stemmer()
        ] )


class LemmPreprocessor( Preprocessor ):

    def __init__( self ):
        super().__init__( [
            LowerConverter(), 
            PunctRemover(), 
            SpecialsRemover(),
            StopwordsRemover(), 
            Lemmatizer()
        ] )
