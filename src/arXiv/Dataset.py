import os
import json

from  . import settings
from .Categories import Categories

class Dataset:

    def __init__( self ):

        self._records = {}
        # check if the dataset exists
        if not os.path.exists( settings.dataset_filename ):
            raise Exception( f'{settings.dataset_filename} not exists.' )

        # retrieve the dataset content from disk
        with open( settings.dataset_filename, 'r', encoding='utf-8' ) as f:
            for line in f:
                record = dict( json.loads( line ) )
                self._records[ record[ 'id' ] ] = record

    def toDict( self ):
        return self._records

    def toList( self ):
        return [ x for x in self._records.values() ]

    def analyze( self ):

        # to count the records per category
        catg_ids = { id: 0 for id, _ in Categories( settings.catgs_filter ).toTuples() }

        # to have a look on the lengths of the summaries 
        summary_lengths = { 'le128': 0, 'le256': 0, 'le512': 0, 'gt512': 0 }

        # analyze the records
        records = self.toList()
        for record in records:
            for catg_id in record[ 'catg_ids' ]:
                catg_ids[ catg_id ] += 1

            l = len( record[ 'summary' ] )
            if l <= 128: summary_lengths[ 'le128' ] += 1
            elif l <= 256: summary_lengths[ 'le256' ] += 1
            elif l <= 512: summary_lengths[ 'le512' ] += 1
            else: summary_lengths[ 'gt512' ] += 1

        # print results
        print( "Total records:", len( records ) )
        print( "Records per category:", catg_ids )
        print( "Total categories:", sum( catg_ids.values() ) )
        print( "Summary lengths:", summary_lengths )


        # to check records encountered in many categories
        # for key, record in records():
        #     if len( record[ 'catg_ids' ] ) > 1:
        #         print( key, record[ 'catg_ids' ] )

        # to check utf-8 encoding
        # print( records[ 12181 ] )

if __name__ == "__main__":
    ds = Dataset()
    ds.analyze()

