import sys
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
from ..helpers.Timer import Timer
from ..settings import pretrained_models

# ---------------------------------------------------------------- #
# Class to retrain BERT-like embeddings regarding corpus sentences #
# ---------------------------------------------------------------- #

class BertRetrainer:

    def __init__( self, sentences:list[str], output_folder:str ):
        self._sentences = sentences
        self._output_folder = output_folder

    def retrain( self ):

        # Load the pre-trained model
        timer = Timer( start=True )
        print( f"Loading pretrained model..." )
        model = SentenceTransformer( 'all-MiniLM-L6-v2', trust_remote_code=True ) #, local_files_only=True
        print( f'(passed {timer.stop()} secs)' )

        # Create training examples (SimCSE-style: same sentence twice)
        train_examples = [ InputExample( texts=[sent, sent] ) for sent in self._sentences ]

        # Create DataLoader
        train_dataloader = DataLoader( train_examples, shuffle=True, batch_size=4, pin_memory=False, num_workers=0 ) # type: ignore
        # pinned memory enables faster host-to-device (CPU -> GPU) transfers via DMA (Direct Memory Access)
        # when pin_memory=True on CPU-only runs wastes memory and may slightly hurt performance

        # Define SimCSE-style contrastive loss
        train_loss = losses.MultipleNegativesRankingLoss( model )

        # Fine-tune the model
        print( f"Fine tuning model..." )
        timer = Timer( start=True )
        model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=1,
            warmup_steps=1000, # (sentences/batch_size)*epochs*0.1
            show_progress_bar=True
        )
        print( f'(passed {timer.stop()} secs)' )

        # Save the fine-tuned model
        print( f"Saving fine-tuned model...")
        model.save( self._output_folder )


# RUN: python -m src.models.BertRetrainer [option]
if __name__ == "__main__": 

    option = None
    if len( sys.argv ) >= 2:
        option = sys.argv[ 1 ]

    match option:

        case 'arxiv':
            from ..datasets.arXiv.Dataset import Dataset
            from ..datasets.arXiv.settings import pickle_paths
            sentences, tags = Dataset().toSentences()
            output_folder = f"{pickle_paths[ 'corpus_repr' ]}/bert-retrained"
            BertRetrainer( sentences, output_folder ).retrain()

        case 'medical':
            from ..datasets.medical.Dataset import Dataset
            from ..datasets.medical.settings import pickle_paths
            sentences, tags = Dataset().toSentences()
            output_folder = f"{pickle_paths[ 'corpus_repr' ]}/bert-retrained"
            BertRetrainer( sentences, output_folder ).retrain()

        case _:
            raise Exception( 'No valid option.' )

