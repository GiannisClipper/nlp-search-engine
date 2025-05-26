import os
import subprocess
from ..settings import pretrained_models

# +-----------------------------------+
# | Code to download GloVe embeddings |
# +-----------------------------------+

class GloveDownloader:

    def __init__( self, url:str, zip_filename:str, filename:str ):
        self._url = url
        self._zip_filename = zip_filename
        self._filename = filename

    def download( self ) -> None:

        print( f'Download {self._zip_filename}...' )
        if os.path.exists( self._zip_filename ):
            print( f'{self._zip_filename} already exists.' )
        else:
            subprocess.run( [ 'wget', self._url, '-O', self._zip_filename ] )

        print( f'Extract {self._filename}...' )
        if os.path.exists( self._filename ):
            print( f'{self._filename} already exists.' )
        else:
            folder = '/'.join( self._filename.split( '/' )[:-1] ) # omit the final segment (the filename)
            subprocess.run( [ 'unzip', self._zip_filename, '-d', folder ] )


# RUN: python -m src.models.GloveDownloader
if __name__ == "__main__":

    url = 'https://www.kaggle.com/api/v1/datasets/download/thanakomsn/glove6b300dtxt'
    zip_filename = pretrained_models[ 'glove-zipped' ]
    filename = pretrained_models[ 'glove' ]

    downloader = GloveDownloader( url, zip_filename, filename )
    downloader.download()
