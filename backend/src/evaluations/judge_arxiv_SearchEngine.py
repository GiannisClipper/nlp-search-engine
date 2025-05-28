import sys
from ..SearchEngine import searchEngineFactory
from ..datasets.arXiv.Dataset import Dataset
from ..models.JudgeModel import JudgeModel
from .queries import queries

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

corpus = Dataset().toList()

all_results = []

for iquery, query in enumerate( queries ):
# for iquery, query in enumerate( queries[5:6] ):

    results = engine.search( query )[:10] # results is a list of tuples [ ('docid', rate), ... ]
    # print( results )
    idocs = [ int(r[0]) for r in results ]
    ranking = '0.00-0.00'
    if len( results ) > 0:
        ranking = f'{results[0][1]:.2f}-{results[-1][1]:.2f}'

    # titles_summaries = [ r[2]['title'] + '-' + r[2]['summarized'] for r in results ]
    # titles = [ corpus[ idoc ][ 'title' ] for idoc in results ] 
    # print( '\n'.join( titles ) )
    titles_summaries = [ corpus[ idoc ][ 'title' ] + ' - ' + corpus[ idoc ][ 'summary' ] for idoc in idocs ] 

    print( '------------------------------------' )
    print( f'Query #{iquery+1}: {query}' )
    model = JudgeModel()

    counter = 0
    yes = 0
    no = 0
    for answer in titles_summaries:
        result = model.judge( query, answer )
        counter += 1
        yes += 1 if result == 'yes' else 0
        no += 1 if result == 'no' else 0
        print( f"Answer #{counter}: {answer}" )
        print( f"Relevant ?: {result}" )
        print( '-----------' )
    print( f'{counter} answers, {yes} relevant(s), {no} non relevant(s)' )
    all_results.append( { 'answers': counter, 'yes': yes, 'no': no, 'ranking': ranking } )

print( '------------------------------------' )
print()

counter = 0
all_answers = 0
all_yes = 0
all_no = 0

print( 'Answers Yes No Ranking   Query' )
print( '------- --- -- --------- ----------------------------------------------------------------' )
for query, result in zip( queries, all_results ):
    counter += 1
    all_answers += result['answers']
    all_yes += result['yes']
    all_no += result['no']
    print( f"{result['answers']:7d} {result['yes']:3d} {result['no']:2d} {result['ranking']} #{counter} {query}" )

print( '------- --- -- --------- ----------------------------------------------------------------' )
print( f"{all_answers:7d} {all_yes:3d} {all_no:2d} ({round(all_yes/all_answers,2)} total precision)" )
