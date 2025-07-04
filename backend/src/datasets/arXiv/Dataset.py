import sys
import os
import json

from nltk.tokenize import sent_tokenize

from . import settings as arxivSettings
from .Categories import Categories

# +------------------------------+
# | Code to manage arXiv dataset |
# +------------------------------+

class Dataset:

    def __init__( self ):

        self._records = []
        # check if the dataset exists
        if not os.path.exists( arxivSettings.dataset_filename ):
            raise Exception( f'{arxivSettings.dataset_filename} not exists.' )

        # retrieve the dataset content from disk
        with open( arxivSettings.dataset_filename, 'r', encoding='utf-8' ) as f:
            for line in f:
                record = dict( json.loads( line ) )
                self._records.append( record )

    def toDict( self ) -> dict:
        return { r['id']: r for r in self._records }

    def toList( self ) -> list[dict]:
        return self._records
    
    def toListTitlesSummaries( self ) -> list[str]:
        return [ r[ 'title' ] + ' - ' + r[ 'summary' ] for r in self._records ]

    def toSentences( self ) -> tuple[list[str],list[str]]:
        sentences = []
        tags = []
        for i, doc in enumerate( self.toList() ):
            sentences.append( doc[ 'title' ] )
            tags.append( f'{i}.0' )
            more_sentences = sent_tokenize( doc[ 'summary' ] )
            for j, sentence in enumerate( more_sentences ):
                sentences.append( sentence )
                tags.append( f'{i}.{j+1}' )
        return sentences, tags

    def toAuthors( self ) -> tuple[list[str],list[str]]:
        authors = []
        tags = []
        for i, doc in enumerate( self.toList() ):
            some_authors = doc[ 'authors' ]
            for j, author in enumerate( some_authors ):
                authors.append( author )
                tags.append( f'{i}.{j}' )
        return authors, tags

    def toPublished( self ) -> tuple[list[str],list[str]]:
        dates = []
        tags = []
        for i, doc in enumerate( self.toList() ):
            dates.append( doc[ 'published' ] )
            tags.append( f'{i}' )
        return dates, tags

    def info( self ):

        # to count the records per category
        catg_ids = { id: 0 for id, _ in Categories( arxivSettings.catgs_filter ).toTuples() }

        # to have a look on the lengths of the summaries 
        summary_lengths = { 'le128': 0, 'le256': 0, 'le512': 0, 'gt512': 0 }

        # analyze the records
        records = self.toList()
        for record in records:
            for catg_id in record[ 'catg_ids' ]:
                catg_ids[ catg_id ] += 1

            # l = len( record[ 'summary' ] )
            # if l <= 128: summary_lengths[ 'le128' ] += 1
            # elif l <= 256: summary_lengths[ 'le256' ] += 1
            # elif l <= 512: summary_lengths[ 'le512' ] += 1
            # else: summary_lengths[ 'gt512' ] += 1

        # print results
        print( "\nTotal records:", len( records ) )
        # print( "Records per category:", catg_ids )
        print( "---------------------")
        print( "Records per category:" )
        for id, descr in Categories( None ).toTuples():
            print( f'{id} {descr} {catg_ids.get(id,0)}' )
        print( "---------------------")

        print( "Sum of records from categories (possible records in more than 1 category):", sum( catg_ids.values() ) )
        # print( "Summary lengths (in chars):", summary_lengths )


        # to check records encountered in many categories
        # for key, record in records():
        #     if len( record[ 'catg_ids' ] ) > 1:
        #         print( key, record[ 'catg_ids' ] )

        # to check utf-8 encoding
        # print( records[ 12181 ] )

    def info_sentences( self ):

        # analyze the sentences
        sentences, tags = self.toSentences()
        lengths = {}
        for t in tags:
            idoc = t.split('.')[0]
            if not idoc in lengths:
                lengths[ idoc ] = 0
            lengths[ idoc ] += 1
        lengths = [ v for k, v in lengths.items() ]

        print( "\nTotal sentences:", len( sentences ) )

        lengths.sort()
        last_length = lengths[0]
        counter = 0
        for l in lengths:
            if l != last_length:
                print( f'Num of sentences:{last_length}, docs:{counter}' )
                last_length = l
                counter = 0
            counter += 1

    def info_tokens( self ):

        corpus = self.toListTitlesSummaries()
        lengths = { '50':0, '100':0, '150':0, '200':0, '500':0 }
        total_tokens = 0
        for doc in corpus:
            tokens = doc.split( ' ' )
            tokens = [ t for t in tokens if len(t) > 0 ]
            if len( tokens ) <=50:
                lengths[ '50' ] += 1
            if len( tokens ) <=100:
                lengths[ '100' ] += 1
            if len( tokens ) <=150:
                lengths[ '150' ] += 1
            elif len( tokens ) <=200:
                lengths[ '200' ] += 1
            else:
                lengths[ '500' ] += 1
            total_tokens += len( tokens )

        print( "\nTotal tokens:", total_tokens )
        print( lengths )

    def info_authors( self ):

        authors, tags = self.toAuthors()
        unique_authors = set( authors )
        print( "\nTotal author names:", len( authors ) )
        print( "Unique author names:", len( unique_authors ) )


# +----------------------------------------+
# | For development and debugging purposes |
# +----------------------------------------+

# RUN: python -m arXiv.Dataset -m [option]
if __name__ == "__main__":

    option = None
    if len( sys.argv ) > 1:
        option = sys.argv[ 1 ]

    if option == 'get':
        if len( sys.argv ) > 2:
            idoc = int( sys.argv[ 2 ] )
        else:
            option = None

    match option:

        case 'info':
            ds = Dataset()
            ds.info()

        case 'info-sentences':
            ds = Dataset()
            ds.info_sentences()

        case 'info-tokens':
            ds = Dataset()
            ds.info_tokens()

        case 'info-authors':
            ds = Dataset()
            ds.info_authors()

        case 'get':
            ds = Dataset()
            records = ds.toList()
            print( records[ idoc ] )

        case _:
            raise Exception( 'No valid option(s).' )

