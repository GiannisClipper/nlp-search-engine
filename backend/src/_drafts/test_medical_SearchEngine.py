import sys
import time
from ..datasets.medical.Dataset import Queries, ResultMetrics, QueriesResults
from ..SearchEngine import searchEngineFactory

option = None
if len( sys.argv ) >= 2:
    option = sys.argv[ 1 ]

if not option:
    raise Exception( 'No option passed.' )

engine = searchEngineFactory( option )    
queries = Queries().toList()[:100]
results = []
start_time = time.time()
for q in queries:
    result = engine.search( q[ 'query' ] )
    result = [ r[ 0 ] for r in result[:10] ]
    results.append( result )
end_time = time.time()
elapsed = round( end_time-start_time, 2 )
elapsed_per_query = round( elapsed / len( queries ), 4 )
print( '\n' )
print( f'Engine option:{option}, queries:{len(queries)}, {elapsed} secs, {elapsed_per_query} secs per query.' )
print( '\n' )
resultMetrics = ResultMetrics()
resultMetrics.compute( queries, results )
resultMetrics.show()
# qr = QueriesResults().toDict()[ queries[0]['id'] ]
# print( queries[0] )
# print( qr )
# print( results[0] )


