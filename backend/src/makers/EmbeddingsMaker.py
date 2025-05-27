import sys
import os
import time

import numpy as np
from sentence_transformers import SentenceTransformer
from ..helpers.Pickle import PickleSaver, PickleLoader
from ..models.GloveModel import GloveModel, gloveModelFactory

# ---------------------------------------------------------------------- #
# Code to create sentence embeddings regarding various pretrained models #
# ---------------------------------------------------------------------- #

class EmbeddingsMaker:

    def __init__( self, model:SentenceTransformer|GloveModel, sentences:list[str], embeddings_filename_base:str, embeddings_filename_merged:str ):
        self._model = model
        self._sentences = sentences
        self._embeddings_filename_base = embeddings_filename_base
        self._embeddings_filename_merged = embeddings_filename_merged

    def make( self ):

        # Encode embeddings in blocks of 1000 sentences
        # ---------------------------------------------

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

            # Delay to handle machine temperature
            if i + 1000 < len( self._sentences ): 
                secs = 10 if i % 10000 == 0 else 1
                print( f'Waiting for {secs} secs to control machine temperature...' )
                time.sleep( secs )

        # Retrieve and merge the embeddings' blocks
        # -----------------------------------------

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

        case 'arxiv-sentences-glove':
            from ..datasets.arXiv.Dataset import Dataset
            from ..datasets.arXiv.settings import pickle_paths
            model = gloveModelFactory( 'arxiv' )
            sentences, tags = Dataset().toSentences()
            embeddings_descr = 'sentences-glove'
            embeddings_filename_base = f"{pickle_paths[ 'temp' ]}/{embeddings_descr}"
            embeddings_filename_merged = f"{pickle_paths[ 'corpus_repr' ]}/{embeddings_descr}.pkl"
            return EmbeddingsMaker( model, sentences, embeddings_filename_base, embeddings_filename_merged )

        case 'arxiv-sentences-glove-retrained':
            from ..datasets.arXiv.Dataset import Dataset
            from ..datasets.arXiv.settings import pickle_paths
            model = gloveModelFactory( 'arxiv-retrained' )
            sentences, tags = Dataset().toSentences()
            embeddings_descr = 'sentences-glove-retrained'
            embeddings_filename_base = f"{pickle_paths[ 'temp' ]}/{embeddings_descr}"
            embeddings_filename_merged = f"{pickle_paths[ 'corpus_repr' ]}/{embeddings_descr}.pkl"
            return EmbeddingsMaker( model, sentences, embeddings_filename_base, embeddings_filename_merged )

        case 'arxiv-sentences-bert':
            from ..datasets.arXiv.Dataset import Dataset
            from ..datasets.arXiv.settings import pickle_paths
            model = SentenceTransformer( 'all-MiniLM-L6-v2', trust_remote_code=True, local_files_only=True )
            sentences, tags = Dataset().toSentences()
            embeddings_descr = 'sentences-bert'
            embeddings_filename_base = f"{pickle_paths[ 'temp' ]}/{embeddings_descr}"
            embeddings_filename_merged = f"{pickle_paths[ 'corpus_repr' ]}/{embeddings_descr}.pkl"
            return EmbeddingsMaker( model, sentences, embeddings_filename_base, embeddings_filename_merged )

        case 'arxiv-sentences-jina':
            from ..datasets.arXiv.Dataset import Dataset
            from ..datasets.arXiv.settings import pickle_paths
            model = SentenceTransformer( "jinaai/jina-embeddings-v2-base-en", trust_remote_code=True, local_files_only=True )
            model.max_seq_length = 1024 # control your input sequence length up to 8192
            sentences, tags = Dataset().toSentences()
            embeddings_descr = 'sentences-jina'
            embeddings_filename_base = f"{pickle_paths[ 'temp' ]}/{embeddings_descr}"
            embeddings_filename_merged = f"{pickle_paths[ 'corpus_repr' ]}/{embeddings_descr}.pkl"
            return EmbeddingsMaker( model, sentences, embeddings_filename_base, embeddings_filename_merged )

        case 'medical-sentences-glove':
            from ..datasets.medical.Dataset import Dataset
            from ..datasets.medical.settings import pickle_paths
            model = gloveModelFactory( 'medical' )
            sentences, tags = Dataset().toSentences()
            embeddings_descr = 'sentences-glove'
            embeddings_filename_base = f"{pickle_paths[ 'temp' ]}/{embeddings_descr}"
            embeddings_filename_merged = f"{pickle_paths[ 'corpus_repr' ]}/{embeddings_descr}.pkl"
            return EmbeddingsMaker( model, sentences, embeddings_filename_base, embeddings_filename_merged )

        case 'medical-sentences-glove-retrained':
            from ..datasets.medical.Dataset import Dataset
            from ..datasets.medical.settings import pickle_paths
            model = gloveModelFactory( 'medical-retrained' )
            sentences, tags = Dataset().toSentences()
            embeddings_descr = 'sentences-glove-retrained'
            embeddings_filename_base = f"{pickle_paths[ 'temp' ]}/{embeddings_descr}"
            embeddings_filename_merged = f"{pickle_paths[ 'corpus_repr' ]}/{embeddings_descr}.pkl"
            return EmbeddingsMaker( model, sentences, embeddings_filename_base, embeddings_filename_merged )

        case 'medical-sentences-jina':
            from ..datasets.medical.Dataset import Dataset
            from ..datasets.medical.settings import pickle_paths
            model = SentenceTransformer( "jinaai/jina-embeddings-v2-base-en", trust_remote_code=True, local_files_only=True )
            model.max_seq_length = 1024 # control your input sequence length up to 8192
            sentences, tags = Dataset().toSentences()
            embeddings_descr = 'sentences-jina'
            embeddings_filename_base = f"{pickle_paths[ 'temp' ]}/{embeddings_descr}"
            embeddings_filename_merged = f"{pickle_paths[ 'corpus_repr' ]}/{embeddings_descr}.pkl"
            return EmbeddingsMaker( model, sentences, embeddings_filename_base, embeddings_filename_merged )

        case 'medical-sentences-bert':
            from ..datasets.medical.Dataset import Dataset
            from ..datasets.medical.settings import pickle_paths
            model = SentenceTransformer( 'all-MiniLM-L6-v2', trust_remote_code=True, local_files_only=True )
            sentences, tags = Dataset().toSentences()
            embeddings_descr = 'sentences-bert'
            embeddings_filename_base = f"{pickle_paths[ 'temp' ]}/{embeddings_descr}"
            embeddings_filename_merged = f"{pickle_paths[ 'corpus_repr' ]}/{embeddings_descr}.pkl"
            return EmbeddingsMaker( model, sentences, embeddings_filename_base, embeddings_filename_merged )

        case 'medical-sentences-bert-retrained':
            from ..datasets.medical.Dataset import Dataset
            from ..datasets.medical.settings import pickle_paths
            model_folder = f"{pickle_paths[ 'corpus_repr' ]}/bert-retrained"
            model = SentenceTransformer( model_folder, trust_remote_code=True, local_files_only=True )
            sentences, tags = Dataset().toSentences()
            embeddings_descr = 'sentences-bert-retrained'
            embeddings_filename_base = f"{pickle_paths[ 'temp' ]}/{embeddings_descr}"
            embeddings_filename_merged = f"{pickle_paths[ 'corpus_repr' ]}/{embeddings_descr}.pkl"
            return EmbeddingsMaker( model, sentences, embeddings_filename_base, embeddings_filename_merged )

        case _:
            raise Exception( 'EmbeddingsMakerFactory(): No valid option.' )


# RUN: python -m src.EmbeddingsMaker [option]
if __name__ == "__main__": 

    option = None
    if len( sys.argv ) >= 2:
        option = sys.argv[ 1 ]

    match option:

        case 'arxiv-sentences-glove' |\
             'arxiv-sentences-glove-retrained' |\
             'arxiv-sentences-jina' |\
             'arxiv-sentences-bert' |\
             'medical-sentences-glove' |\
             'medical-sentences-glove-retrained' |\
             'medical-sentences-jina' |\
             'medical-sentences-bert' |\
             'medical-sentences-bert-retrained':
            maker = embeddingsMakerFactory( option )
            maker.make()

        case _:
            raise Exception( 'No valid option.' )
