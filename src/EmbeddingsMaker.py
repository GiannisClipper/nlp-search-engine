import sys
import os
import time

import numpy as np
from sentence_transformers import SentenceTransformer
from .Preprocessor import Preprocessor, LemmPreprocessor, StemmPreprocessor
from .helpers.Pickle import PickleSaver, PickleLoader

class EmbeddingsMaker:

    def __init__( self, model:SentenceTransformer, sentences:list[str], embeddings_filename_base:str, embeddings_filename_merged:str ):
        self._model = model
        self._sentences = sentences
        self._embeddings_filename_base = embeddings_filename_base
        self._embeddings_filename_merged = embeddings_filename_merged

    def make( self ):

        # Encode and save embeddings in blocks of 1000 sentences
        # ------------------------------------------------------

        print( 'Total sentences:', len( self._sentences ) )
        print( 'Create and save sentence embeddings...' )
        for i in range( 0, len( self._sentences ), 1000 ):

            # Check if embeddings' block file already exists
            embeddings_filename_block = f"{self._embeddings_filename_base}-{i}-{i+1000-1}.pkl"
            if os.path.exists( embeddings_filename_block ):
                print( f'{embeddings_filename_block} already exists.' )
                continue

            # Create embeddings' block (per 100 sentences to show progress)
            sentences_repr = []
            for j in range( i, min( i+1000, len( self._sentences ) ), 100 ):
                print( 'Encoding:', j, '-', j+100-1 )
                sentences_repr.append( self._model.encode( self._sentences[j:j+100] ) ) # embeddings
            sentences_repr = np.concatenate( sentences_repr )

            # Save embeddings' block
            PickleSaver( embeddings_filename_block ).save( sentences_repr )

            # Delay to handle machine's temperature
            if i + 1000 < len( self._sentences ): time.sleep( 60 )

        # Collect and merge the embeddings' blocks
        # ----------------------------------------

        # Check if merged file already exists
        if os.path.exists( self._embeddings_filename_merged ):
            print( f'{self._embeddings_filename_merged} already exists.' )
            return

        # Retrieve embeddings' block
        print( 'Retrieving the blocks of embeddings...' )
        embeddings = []
        for i in range( 0, len( self._sentences ), 1000 ):
            embeddings_filename_block = f"{self._embeddings_filename_base}-{i}-{i+1000-1}.pkl"
            embeddings.append( PickleLoader( embeddings_filename_block ).load() )

        # Save merged embeddings
        print( 'Saving as merged file...' )
        PickleSaver( self._embeddings_filename_merged ).save( np.concatenate( embeddings ) )

    def __str__( self ):
        return self.__class__    


def embeddingsMakerFactory( option:str ):

    match option:

        case 'arxiv-sentences-jina':
            from .arXiv.Dataset import Dataset
            from .arXiv.settings import pickle_paths
            model = SentenceTransformer( "jinaai/jina-embeddings-v2-base-en", trust_remote_code=True, local_files_only=True )
            model.max_seq_length = 1024 # control your input sequence length up to 8192
            sentences, tags = Dataset().toSentences()
            sentences = LemmPreprocessor().transform( sentences )
            embeddings_filename_base = f"{pickle_paths[ 'temp' ]}/{embeddings_descr}"
            embeddings_filename_merged = f"{pickle_paths[ 'corpus_repr' ]}/{embeddings_descr}.pkl"
            return EmbeddingsMaker( model, sentences, embeddings_filename_base, embeddings_filename_merged )

        case 'medical-sentences-jina':
            from .medical.Dataset import Dataset
            from .medical.settings import pickle_paths
            model = SentenceTransformer( "jinaai/jina-embeddings-v2-base-en", trust_remote_code=True, local_files_only=True )
            model.max_seq_length = 1024 # control your input sequence length up to 8192
            sentences, tags = Dataset().toSentences()
            sentences = LemmPreprocessor().transform( sentences )
            embeddings_filename_base = f"{pickle_paths[ 'temp' ]}/{embeddings_descr}"
            embeddings_filename_merged = f"{pickle_paths[ 'corpus_repr' ]}/{embeddings_descr}.pkl"
            return EmbeddingsMaker( model, sentences, embeddings_filename_base, embeddings_filename_merged )

        case _:
            raise Exception( 'EmbeddingsMakerFactory(): No valid option.' )


# RUN: python -m src.EmbeddingsMaker [option]
if __name__ == "__main__": 

    embeddings_descr = 'jina-embeddings-sentences'

    option = None
    if len( sys.argv ) >= 2:
        option = sys.argv[ 1 ]

    match option:

        case 'arxiv-sentences-jina':
            maker = embeddingsMakerFactory( option )
            maker.make()

        case 'medical-sentences-jina':
            maker = embeddingsMakerFactory( option )
            maker.make()

        case _:
            raise Exception( 'No valid option.' )
