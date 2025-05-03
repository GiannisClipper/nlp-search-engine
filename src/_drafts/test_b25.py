# base on:
# BM25 for Python: Achieving high performance while simplifying dependencies with BM25S
# https://huggingface.co/blog/xhluca/bm25s

# pip install bm25s
import bm25s

from ..datasets.medical.Dataset import Dataset
from ..Preprocessor import LemmPreprocessor

preprocessor = LemmPreprocessor()
corpus = Dataset().toListTitlesAbstracts()
corpus = preprocessor.transform( corpus )

# Create the BM25 model and index the corpus
print( 'Creating the model...' )
retriever = bm25s.BM25() # class BM25: https://github.com/xhluca/bm25s/blob/main/bm25s/__init__.py
retriever.index( bm25s.tokenize( corpus ) )

# Query the corpus and get top-k results
print( 'Retrieving doc matching to query...' )
query = "I want to learn more about cancer cases..."
results, scores = retriever.retrieve( bm25s.tokenize( query ), k=5 )

print( results )
print( scores )

# # Let's see what we got!
# doc, score = results[0, 0], scores[0, 0]
# print(f"Rank {i+1} (score: {score:.2f}): {doc}")