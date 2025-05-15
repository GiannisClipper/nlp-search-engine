import sys
from flask import Flask, request

# CORS in a Flask API
# https://medium.com/@mterrano1/cors-in-a-flask-api-38051388f8cc
from flask_cors import CORS

from src.SearchEngine import searchEngineFactory, AbstractSearchEngine
from src.datasets.arXiv.Dataset import Dataset

app = Flask( __name__ )
CORS( app )

option:None|str = None
engine:AbstractSearchEngine
corpus:list

def get_doc( idoc ):
    doc = corpus[ int( idoc ) ]
    return {
        'id': doc[ 'id' ],
        'title': doc[ 'title' ],
        'authors': doc[ 'authors' ],
        'catg_ids': doc[ 'catg_ids' ],
        'published': doc[ 'published' ],
        'summary': doc[ 'summary' ],
    }


@app.route( '/', methods=[ 'GET' ] )
def main():
    return f'Search engine ({option}) is up and running...'

@app.route( '/info', methods=[ 'GET' ] )
def info():
    return {
        'option': option
    }

@app.route( '/search', methods=[ 'POST', 'GET' ] )
def search():

    query, authors, authors = None, None, None

    if request.method == 'GET':
        query = request.args.get( 'query' )
        authors = request.args.get( 'authors' )
        published = request.args.get( 'published' )

    if request.method == 'POST':
        params = request.get_json()
        query = params[ 'query' ]
        authors = params[ 'authors' ]
        published = params[ 'published' ]

    print( query, authors, published )
    if not query or query.strip() == '':
        return { 'error': 'No query found.' }

    results = engine.search( query )[:10]
    results = [ get_doc( idoc ) for idoc, rank in results ] 
    return results


if __name__ == '__main__':

    option = None
    if len( sys.argv ) >= 2:
        option = sys.argv[ 1 ]

    match option:
        case 'arxiv-lemm-single-tfidf' |\
             'arxiv-lemm-2gram-tfidf' |\
             'arxiv-sentences-glove-b25' |\
             'arxiv-sentences-bert-faiss':
            engine = searchEngineFactory( option )
            corpus = Dataset().toList()

        case _:
            raise Exception( 'No valid option.' )

    print( main() )
    app.run( debug=True, port=5000 )