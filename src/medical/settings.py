import os 

file_path = os.path.dirname( os.path.realpath( __file__ ) )

dataset_filename = f'{file_path}/doc_dump.txt'
ids_filename = f'{file_path}/dev.docs.ids'
queries_filename = f'{file_path}/dev.titles.queries'
results_filename = f'{file_path}/dev.3-2-1.qrel'

pickle_paths = {
    'preprocessors': f'{file_path}/pickles/preprocessors',
    'vocabularies': f'{file_path}/pickles/vocabularies',
    'vectorizers': f'{file_path}/pickles/vectorizers',
    'corpus_repr': f'{file_path}/pickles/corpus_repr',
    'indexes': f'{file_path}/pickles/indexes',
}
