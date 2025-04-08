import os 

file_path = os.path.dirname( os.path.realpath( __file__ ) )
catgs_filename = f'{file_path}/downloads/categories.txt'
dataset_filename = f'{file_path}/datasets/dataset.jsonl'

# to select specific categories (not all)
catgs_filter = [ 'AI', 'CL', 'DB', 'DC', 'DS', 'GL', 'IR', 'LG', 'NI', 'PL' , 'SE', 'SI' ]
catgs_filter = [ f'cs.{id}' for id in catgs_filter ]
