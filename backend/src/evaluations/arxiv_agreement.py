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

options = [
    'arxiv-lemm-2gram-tfidf',
    'arxiv-sentences-glove-retrained-bm25',
    'arxiv-sentences-bert-faiss'
]

queries = queries[0:5]

results = []
answers = {}
for option in options:

    print( '\nEngine:', option )
    engine = searchEngineFactory( option )
    answers[ option ] = []

    for iquery, query in enumerate( queries ):
        print( f'\nQuery: #{iquery+1} query' )
        ans = engine.search( query )[:10]
        ans = [ a[0] for a in ans ]
        print( 'Answers:', ans )
        answers[ option ].append( ans )

for iquery, query in enumerate( queries ):
    ans = []
    for option in options:
        # print( type( answers[ option ] ), answers[ option ] )
        ans += answers[ option ][ iquery ]
    # print( type(ans), ans )
    ans = set( ans )

    print( f'\nQuery: #{iquery+1} query' )
    for option in options:
        print( f'{option}: {len( ans &  set( answers[option][ iquery ] ))} / {len(ans)}' )


