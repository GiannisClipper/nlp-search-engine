import sys
from statistics import mean
import os

from nltk.tokenize import sent_tokenize

from . import settings as medicalSettings

# +--------------------------------+
# | Code to manage medical dataset |
# +--------------------------------+

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
        return [ r[ 'title' ] + ' - ' + r[ 'abstract' ] for r in self._records ]
    
    def toSentences( self ) -> tuple[list[str],list[str]]:
        sentences = []
        tags = []
        for i, doc in enumerate( self.toList() ):
            sentences.append( doc[ 'title' ] )
            tags.append( f'{i}.0' )
            more_sentences = sent_tokenize( doc[ 'abstract' ] )
            for j, sentence in enumerate( more_sentences ):
                sentences.append( sentence )
                tags.append( f'{i}.{j+1}' )
        return sentences, tags

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

    def analyze( self ):

        records = self.toList()

        sentences, tags = self.toSentences()
        lengths = {}
        for t in tags:
            idoc = t.split('.')[0]
            if not idoc in lengths:
                lengths[ idoc ] = 0
            lengths[ idoc ] += 1
        lengths = [ v for k, v in lengths.items() ]
        min_sentence_length = min( lengths )
        mean_sentence_length = sum( lengths ) / len( lengths )
        max_sentence_length = max( lengths )

        # print results
        print( "Total records:", len( records ) )
        print( "Total sentences:", len( sentences ) )
        print( "Min-Mean-Max sentence length:", min_sentence_length, mean_sentence_length, max_sentence_length )

class Queries:

    def __init__( self ):

        self._records = []
        # check if the queries' file exists
        if not os.path.exists( medicalSettings.queries_filename ):
            raise Exception( f'{medicalSettings.queries_filename} not exists.' )

        # retrieve the queries from disk
        with open( medicalSettings.queries_filename, 'r', encoding='utf-8' ) as f:
            for line in f:
                record = line.strip().split( '\t' )
                record = {
                    'id': record[ 0 ],
                    'query': record[ 1 ],
                }
                self._records.append( record )

    def toDict( self ) -> dict:
        return { r['id']: r for r in self._records }

    def toList( self ) -> list[dict]:
        return self._records


class QueriesResults:

    def __init__( self ):

        self._records = []
        # check if the results' file exists
        if not os.path.exists( medicalSettings.results_filename ):
            raise Exception( f'{medicalSettings.results_filename} not exists.' )

        # retrieve the queries from disk
        with open( medicalSettings.results_filename, 'r', encoding='utf-8' ) as f:
            for line in f:
                record = line.strip().split( '\t' )
                record = {
                    'query_id': record[ 0 ],
                    'doc_id': record[ 2 ],
                }
                self._records.append( record )

    def toDict( self ) -> dict:
        d = {}
        for r in self._records:
            if r[ 'query_id' ] not in d:
                d[ r[ 'query_id' ] ] = []
            d[ r[ 'query_id' ] ].append( r[ 'doc_id' ] )
        return d

    def toList( self ) -> list[dict]:
        return self._records


class ResultMetrics:

    def __init__( self ):

        self._ids:list[str] = []
        self._tp:list[int] = []
        self._fp:list[int] = []
        self._fn:list[int] = []
        self._precision:list[float] = []
        self._recall:list[float] = []
        self._f1:list[float] = []

        self._docs = Dataset().toList()

    def _computeOne( self, query:dict, result:list[str] ):

        y = set( QueriesResults().toDict()[ query[ 'id' ] ] )
        ids = [ self._docs[ int(idoc) ][ 'id' ] for idoc in result ]
        y_hat = set( ids )

        tp = len( y & y_hat )  # intersection of correct and retrieved
        fp = len( y_hat - y )  # retrieved but not correct
        fn = len( y - y_hat )  # correct but not retrieved

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        return tp, fp, fn, precision, recall, f1

    def compute( self, queries:list[dict], results:list[list[str]] ):

        self._ids = []
        self._tp = []
        self._fp = []
        self._fn = []
        self._precision = []
        self._recall = []
        self._f1 = []
        self._remarks = []

        for q, r in zip( queries, results ):
            tp, fp, fn, precision, recall, f1 = self._computeOne( q, r )
            self._ids.append( q[ 'id' ] )
            self._tp.append( tp )
            self._fp.append( fp )
            self._fn.append( fn )
            self._precision.append( precision )
            self._recall.append( recall )
            self._f1.append( f1 )
            self._remarks.append( q[ 'query' ] )

    def show( self ):

        print( '    ID        TP       FP       FN      PREC    RECALL     F1   ' )
        print( '---------- -------- -------- -------- -------- -------- --------' )

        for id, tp, fp, fn, precision, recall, f1, remarks in zip( self._ids, self._tp, self._fp, self._fn, self._precision, self._recall, self._f1, self._remarks ):
            print( f"{id:<10} {tp:8d} {fp:8d} {fn:8d} {precision:8.4f} {recall:8.4f} {f1:8.4f} {remarks}" )

        if len( self._ids ) > 1:
            print( '---------- -------- -------- -------- -------- -------- --------' )
            print( f'{"MEAN":<10} {mean(self._tp):8.4f} {mean(self._fp):8.4f} {mean(self._fn):8.4f} {mean(self._precision):8.4f} {mean(self._recall):8.4f} {mean(self._f1):8.4f}' )


# +----------------------------------------+
# | For development and debugging purposes |
# +----------------------------------------+

# RUN: python -m medical.Dataset [option]
if __name__ == "__main__":

    option = None
    if len( sys.argv ) > 1:
        option = sys.argv[ 1 ]

    match option:

        case 'dataset':
            ds = Dataset()
            ds.analyze()

        case 'info-sentences':
            ds = Dataset()
            ds.info_sentences()

        case 'queries':
            qr = Queries()
            queries = qr.toList()
            print( len( queries ), queries[:3] )

        case 'results':
            qr = QueriesResults()
            results = qr.toList()
            print( len( results ), results[:10] )

        case 'results-dict':
            qr = QueriesResults()
            results = qr.toDict()
            print( 'query_id: PLAIN-1, doc_ids:', results[ 'PLAIN-1' ] )
            # query_id = qr.toList()[ 0 ][ 'query_id' ]
            # print( results[ query_id ] )

        case _:
            raise Exception( 'No valid option.' )
