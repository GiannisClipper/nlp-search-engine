import sys
from ..SearchEngine import searchEngineFactory
from ..datasets.arXiv.Dataset import Dataset
from ..models.JudgeModel import JudgeModel
from .queries import queries
from ..helpers.Timer import Timer

# e.g.
# option = 'arxiv-lemm-single-tfidf'
# option = 'arxiv-lemm-2gram-tfidf'
# option = 'arxiv-sentences-glove-retrained-bm25'
# option = 'arxiv-sentences-jina-faiss'

option = None
if len( sys.argv ) >= 2:
    option = sys.argv[ 1 ]

if not option:
    raise Exception( 'No option passed.' )

engine = searchEngineFactory( option )

results = []

for iquery, query in enumerate( queries ):
# for iquery, query in enumerate( queries[5:7] ):

    timer = Timer( start=True )
    print( f'#{iquery} {query}' )
    print( f"Request search engine..." )
    answers = engine.search( query )
    print( f'(passed {timer.stop()} secs)' )
    print()
    results.append( {'answers': len( answers ), 'delay': timer.diff() } )

print( 'Answers Delay  Query' )
print( '------- ------ ----------------------------------------------------------------' )
counter = 0
for query, result in zip( queries, results ):
    counter += 1
    print( f"{result['answers']:7d} {result['delay']:.4f} #{counter} {query}" )

print( '------- ------ ----------------------------------------------------------------' )
print()

delays = [ r['delay'] for r in results ]
min_delay = min( delays )
max_delay = max( delays )
mean_delay = sum( delays ) / len( delays )
print( f"Min time delay: {min_delay:.4f} secs" )
print( f"Max time delay: {max_delay:.4f} secs" )
print( f"Mean: {mean_delay:.4f} secs" )
