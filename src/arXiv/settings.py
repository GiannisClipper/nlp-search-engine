import os 

file_path = os.path.dirname( os.path.realpath( __file__ ) )

catgs_filename = f'{file_path}/downloads/categories.txt'
dataset_filename = f'{file_path}/datasets/dataset.jsonl'

pickle_filenames = {
    'preprocessor': f'{file_path}/pickles/preprocessor.pkl',

    'vocabulary': f'{file_path}/pickles/vocabulary.pkl',

    'count': {
        'vectorizer': f'{file_path}/pickles/count_vectorizer.pkl',
        'corpus_repr': f'{file_path}/pickles/count_corpus_repr.pkl'
    },

    'tfidf': {
        'vectorizer': f'{file_path}/pickles/tfidf_vectorizer.pkl',
        'corpus_repr': f'{file_path}/pickles/tfidf_corpus_repr.pkl'
    }
}

# to select specific categories (not all)
catgs_filter = [ 'AI', 'CL', 'DB', 'DC', 'DS', 'GL', 'IR', 'LG', 'NI', 'PL' , 'SE', 'SI' ]
catgs_filter = [ f'cs.{id}' for id in catgs_filter ]
