# from nltk.tokenize.stanford import StanfordTokenizer 
# NOTICE: NLTK was unable to find stanford-postagger.jar! Set the CLASSPATH

import numpy as np
from ..Preprocessor import LowerWordsPreprocessor
from ..makers.Tokenizer import SingleTokenizer
from ..settings import pretrained_models

# +----------------------------------------+
# | Code to make use of a GloVe embeddings |
# +----------------------------------------+

class GloveModel:   

    def __init__( self, corpus:list[str], embeddings_filename:str, embedding_dim:int=300 ):

        self._corpus = corpus
        self._embeddings_filename = embeddings_filename
        self._embedding_dim = embedding_dim
        self._preprocessor = LowerWordsPreprocessor()
        self._tokenizer = SingleTokenizer()

        # make the vocabulary from corpus
        corpus_preprocessed = self._preprocessor.transform( self._corpus )
        words = self._tokenizer.tokenize( ' '.join( corpus_preprocessed ) )
        words = list( set( words ) )
        self._word_index = { w:idx for idx, w in enumerate( words ) }
        vocab_size = len( words ) + 1 # add 1 due to reserved 0 index

        # read the word embeddings from file
        self._embeddings = np.zeros( ( vocab_size, self._embedding_dim ) )
        with open( self._embeddings_filename, encoding="utf8" ) as f:
            for line in f:
                word, *vector = line.split()
                if word in self._word_index:
                    idx = self._word_index[ word ]
                    self._embeddings[ idx ] = np.array( vector, dtype=np.float32 )[:self._embedding_dim]

    # def _tokenize( self, text:str ):
    #     return self._termsMaker.make( text.lower() )

    def _encode_one( self, sentence ):
        # words = self._tokenize( sentence )
        sentence_preprocessed = self._preprocessor.transform( sentence )
        words = self._tokenizer.tokenize( ''.join( sentence_preprocessed ) )
        embeddings = [ self._embeddings[ self._word_index[ word ] ] for word in words if word in self._word_index ]
        if len( embeddings ) == 0:
            return np.zeros( 1 * self._embedding_dim, dtype=np.float32 )
        
        # concatanate the mean value for each dimension + the max value for each dimension
        # return np.round( np.concatenate( [ np.mean( embeddings, axis=0 ), np.amax( embeddings, axis=0 ) ], dtype=np.float32 ), decimals=8 )

        # the mean value for each dimension
        return np.round( np.mean( embeddings, axis=0, dtype=np.float32 ), decimals=8 )

    def encode( self, sentences:str|list[str] ):

        # in case of single str
        if isinstance( sentences, str ):
            sentence = sentences
            sentence = self._preprocessor.transform( [ sentence ] )[0]
            return self._encode_one( sentence )

        # in case of list of str
        result = []
        sentences = self._preprocessor.transform( sentences )
        for sentence in sentences:
            result.append( self._encode_one( sentence ) )
        return result
 

def gloveModelFactory( option:str ):

    match option:

        case 'arxiv':
            from ..datasets.arXiv.Dataset import Dataset
            corpus = Dataset().toListTitlesSummaries()
            embeddings_filename = pretrained_models[ 'glove' ]
            model = GloveModel( corpus, embeddings_filename )
            return model

        case 'arxiv-retrained':
            from ..datasets.arXiv.Dataset import Dataset
            from ..datasets.arXiv.settings import pickle_paths
            corpus = Dataset().toListTitlesSummaries()
            embeddings_filename = f"{pickle_paths[ 'corpus_repr' ]}/glove-retrained.txt"
            model = GloveModel( corpus, embeddings_filename )
            return model

        case 'medical':
            from ..datasets.medical.Dataset import Dataset
            corpus = Dataset().toListTitlesAbstracts()
            embeddings_filename = pretrained_models[ 'glove' ]
            model = GloveModel( corpus, embeddings_filename )
            return model

        case 'medical-retrained':
            from ..datasets.medical.Dataset import Dataset
            from ..datasets.medical.settings import pickle_paths
            corpus = Dataset().toListTitlesAbstracts()
            embeddings_filename = f"{pickle_paths[ 'corpus_repr' ]}/glove-retrained.txt"
            model = GloveModel( corpus, embeddings_filename )
            return model

        case _:
            raise Exception( 'gloveModelFactory(): No valid option.' )


if __name__ == '__main__':
    print( 'loading...' )
    model = gloveModelFactory( 'arxiv' )
    sentence = 'What about a simple test?'
    result = model.encode( [ sentence ] )
    print( result, len(result), len(result[0]) )

    # from ..datasets.arXiv.settings import pickle_paths
    # from ..helpers.Pickle import PickleLoader
    # filename = f"{pickle_paths[ 'corpus_repr' ]}/sentences-glove.pkl"
    # emb = PickleLoader( filename ).load()
    # print( type(emb), emb.shape, emb[1,1], type(emb[1,1]) )

    # filename = f"{pickle_paths[ 'corpus_repr' ]}/sentences-bert.pkl"
    # emb = PickleLoader( filename ).load()
    # print( type(emb), emb.shape, emb[1,1], type(emb[1,1]) )

    # filename = f"{pickle_paths[ 'corpus_repr' ]}/sentences-jina.pkl"
    # emb = PickleLoader( filename ).load()
    # print( type(emb), emb.shape, emb[1,1], type(emb[1,1]) )
