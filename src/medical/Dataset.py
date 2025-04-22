from abc import ABC, abstractmethod

import os
import json

from . import settings as medicalSettings

class Dataset:

    def __init__( self ):

        self._records = []
        # check if the dataset exists
        if not os.path.exists( medicalSettings.dataset_filename ):
            raise Exception( f'{medicalSettings.dataset_filename} not exists.' )

        ids = []
        with open( medicalSettings.ids_filename, 'r', encoding='utf-8' ) as f:
            ids = [ line.strip() for line in f.readlines() ]

        # retrieve the dataset content from disk
        with open( medicalSettings.dataset_filename, 'r', encoding='utf-8' ) as f:
            for line in f:
                record = line.strip().split( '\t' )
                record = {
                    'id': record[ 0 ],
                    'url': record[ 1 ],
                    'title': record[ 2 ],
                    'abstract': record[ 3 ],
                }

                # to include only the documents matching to ids (as proposed in course's examples)
                if record[ 'id' ] in ids:
                    self._records.append( record )

    def toDict( self ) -> dict:
        return { r['id']: r for r in self._records }

    def toList( self ) -> list[dict]:
        return self._records

    def toListTitlesAbstracts( self ) -> list[str]:
        return [ r[ 'title' ] + '-' + r[ 'abstract' ] for r in self._records ]

if __name__ == "__main__":
    ds = Dataset()
    docs = ds.toList()
    print( len( docs ), docs[:3] )