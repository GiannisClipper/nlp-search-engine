from .Preprocessor import Preprocessor, LemmPreprocessor, StemmPreprocessor
from .DocFinder import DocFinder
from .TextFilter import TextFilter

from .helpers.Pickle import PickleLoader

from .medical.Dataset import Dataset, Queries, ResultMetrics
from .medical.settings import pickle_paths

vectorizer_descr = 'title-summary_lower-punct-specials-stops-lemm_single_tfidf'
index_descr = 'title-summary_lower-punct-specials-stops-lemm_single'

vectorizer_filename = f"{pickle_paths[ 'vectorizers' ]}/{vectorizer_descr}.pkl"
corpus_repr_filename = f"{pickle_paths[ 'corpus_repr' ]}/{vectorizer_descr}.pkl"
index_filename = f"{pickle_paths[ 'indexes' ]}/{index_descr}.pkl"
index = PickleLoader( index_filename ).load()
corpus=Dataset().toList()

docFinder = DocFinder(
    preprocessor=LemmPreprocessor(), 
    vectorizerLoader=PickleLoader( vectorizer_filename ),
    corpusReprLoader=PickleLoader( corpus_repr_filename ),
    corpus=corpus,
    textFilter=TextFilter( corpus, index )
)

q = Queries().toDict()
queries = [ 
    q[ 'PLAIN-1' ],
    q[ 'PLAIN-11' ],
    q[ 'PLAIN-111' ],
    q[ 'PLAIN-122' ]
]
queries = Queries().toList()[:10]

results= []
for q in queries:
    result = docFinder.find( q[ 'query' ] )
    results.append( [ r[0]['id'] for r in result ][:10] )

print( '\n\n' )
resultMetrics = ResultMetrics()
resultMetrics.compute( queries, results )
resultMetrics.show()
