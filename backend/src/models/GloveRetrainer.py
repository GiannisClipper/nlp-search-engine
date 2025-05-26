import sys
from gensim.models import Word2Vec, KeyedVectors
from ..settings import pretrained_models
from ..Preprocessor import NaivePreprocessor
from ..makers.TermsMaker import SingleTermsMaker
from ..helpers.Timer import Timer

# ----------------------------------------------------------- #
# Code to retrain GloVe embeddings regarding corpus sentences #
# ----------------------------------------------------------- #

class GloveRetrainer:

    def __init__( self, sentences:list[str], output_filename:str ):
        self._sentences = sentences
        self._output_filename = output_filename

    def retrain( self ):

        # Read pretrained glove
        timer = Timer( start=True )
        print( f"Reading pretrained GloVe..." )
        filename = pretrained_models[ 'glove' ]
        timer = Timer( start=True )
        glove_model = KeyedVectors.load_word2vec_format( filename, binary=False, no_header=True )
        print( glove_model )
        print( f'(passed {timer.stop()} secs)' )

        # Preprocess sentences
        timer = Timer( start=True )
        print( f"Preprocessing sentences..." )
        sentences_preprocessed = NaivePreprocessor().transform( sentences )
        tkzr = SingleTermsMaker()
        sentences_tokenized = [ tkzr.make( s ) for s in sentences_preprocessed ]
        print( f'(passed {timer.stop()} secs)' )

        # Initialize a Word2Vec model
        print( f"Initializing Word2Vec model...")
        w2v_model = Word2Vec(
            vector_size=300,
            window=5,
            min_count=1,
            sg=1, # Skip-gram
            workers=4
        )

        # Build vocabulary
        timer = Timer( start=True )
        print( f"Building vocabulary..." )
        w2v_model.build_vocab( sentences_tokenized )
        # words = list( set( [ w for s in sentences_tokenized for w in s ] ) )
        counter = 0
        for word in w2v_model.wv.key_to_index:
            if word in glove_model:
                w2v_model.wv[ word ] = glove_model[ word ]
                counter += 1
        print( f"Initialized {counter} words from pretrained GloVe. " )
        print( 'w2v_model:', w2v_model, 'w2v_model.corpus_count:', w2v_model.corpus_count )
        print( f'(passed {timer.stop()} secs)' )

        # Retrain the model and update the embeddings
        timer = Timer( start=True )
        print( f"Retraining model/ updating embeddings..." )
        w2v_model.train(
            sentences_tokenized,
            total_examples=w2v_model.corpus_count,
            epochs=30
        )
        print( f'(passed {timer.stop()} secs)' )

        # Save the fine-tuned embeddings
        print( f"Saving fine-tuned embeddings...")
        w2v_model.wv.save_word2vec_format( output_filename, write_header= False )


# RUN: python -m src.models.GloveRetrainer [option]
if __name__ == "__main__": 

    option = None
    if len( sys.argv ) >= 2:
        option = sys.argv[ 1 ]

    match option:

        case 'arxiv':
            from ..datasets.arXiv.Dataset import Dataset
            from ..datasets.arXiv.settings import pickle_paths
            sentences, tags = Dataset().toSentences()
            output_filename = f"{pickle_paths[ 'corpus_repr' ]}/glove-retrained.txt"
            GloveRetrainer( sentences, output_filename ).retrain()

        case 'medical':
            from ..datasets.medical.Dataset import Dataset
            from ..datasets.medical.settings import pickle_paths
            sentences, tags = Dataset().toSentences()
            output_filename = f"{pickle_paths[ 'corpus_repr' ]}/glove-retrained.txt"
            GloveRetrainer( sentences, output_filename ).retrain()

        case _:
            raise Exception( 'No valid option.' )

