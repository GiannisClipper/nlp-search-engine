import os 

dataset_path = os.path.dirname( os.path.realpath( __file__ ) )

dataset_filename = f'{dataset_path}/dataset.jsonl'

catgs_filename = f'{dataset_path}/downloads/categories.txt'
# to select specific categories (not all)
catgs_filter = [ 'AI', 'CL', 'DB', 'DC', 'DS', 'GL', 'IR', 'LG', 'NI', 'PL' , 'SE', 'SI' ]
catgs_filter = [ f'cs.{id}' for id in catgs_filter ]

pickle_paths = {
    'preprocessors': f'{dataset_path}/pickles/preprocessors',
    'vocabularies': f'{dataset_path}/pickles/vocabularies',
    'vectorizers': f'{dataset_path}/pickles/vectorizers',
    'corpus_repr': f'{dataset_path}/pickles/corpus_repr',
    'indexes': f'{dataset_path}/pickles/indexes',
    'clusters': f'{dataset_path}/pickles/clusters',
    'temp': f'{dataset_path}/pickles/temp'
}
