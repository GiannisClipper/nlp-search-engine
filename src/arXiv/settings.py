import os 

file_path = os.path.dirname( os.path.realpath( __file__ ) )

catgs_filename = f'{file_path}/downloads/categories.txt'
dataset_filename = f'{file_path}/datasets/dataset.jsonl'

pickle_paths = {
    'preprocessors': f'{file_path}/pickles/preprocessors',
    'vocabularies': f'{file_path}/pickles/vocabularies',
    'vectorizers': f'{file_path}/pickles/vectorizers',
    'corpus_repr': f'{file_path}/pickles/corpus_repr',
}

# to select specific categories (not all)
catgs_filter = [ 'AI', 'CL', 'DB', 'DC', 'DS', 'GL', 'IR', 'LG', 'NI', 'PL' , 'SE', 'SI' ]
catgs_filter = [ f'cs.{id}' for id in catgs_filter ]
