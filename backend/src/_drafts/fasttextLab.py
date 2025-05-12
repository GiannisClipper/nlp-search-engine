
# load_from_url('https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.bin.gz')
# # Load pretrained FastText model (e.g., from the official FastText website)


from ..settings import pretrained_models

# from gensim.models import FastText
# fasttext_model = FastText.load_fasttext_format( models[ 'fasttext' ] ) # FastText model for English

# (use load_facebook_vectors (to use pretrained embeddings) 
# or load_facebook_model (to continue training with the loaded full model, more RAM) instead)
from gensim.models.fasttext import load_facebook_vectors
fasttext_model = load_facebook_vectors( pretrained_models[ 'fasttext' ] )

# # Example: Get the vector for 'cat'
# vector_fasttext = fasttext_model['cat']
# print(f"Vector for 'cat': {vector_fasttext[:3]}")  # Print the first 3 dimensions of the vector

# Example: Calculate similarity between 'cat' and 'dog'
similarity_fasttext = fasttext_model.wv.similarity( 'cat', 'dog' )
print( f"Similarity between 'cat' and 'dog' (FastText): {similarity_fasttext:.3f}" )
