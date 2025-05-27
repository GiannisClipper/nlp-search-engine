import os
from sentence_transformers import SentenceTransformer
from ..settings import pretrained_models

# +----------------------------------+
# | Code to download BERT-like model |
# +----------------------------------+

class BertDownloader:

    def __init__( self, model_name:str, folder:str ):
        self._model_name = model_name
        self._folder = folder

    def download( self ) -> None:

        print( f'Download {self._model_name}...' )
        if os.path.exists( self._folder ):
            print( f'{self._folder} already exists.' )
        else:
            model = SentenceTransformer( self._model_name )
            model.save( self._folder )
        

# RUN: python -m src.models.BertDownloader
if __name__ == "__main__":

    model_name = 'all-MiniLM-L6-v2'
    folder = pretrained_models[ 'bert' ]

    downloader = BertDownloader( model_name, folder )
    downloader.download()
