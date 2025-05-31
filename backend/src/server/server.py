import sys
from typing import cast
from flask import Flask, request

# CORS in a Flask API
# https://medium.com/@mterrano1/cors-in-a-flask-api-38051388f8cc
from flask_cors import CORS

from ..SearchEngine import searchEngineFactory, AbstractSearchEngine, QueryNamesPeriodSearchEngine
from ..models.JudgeModel import JudgeModel
from ..datasets.arXiv.Dataset import Dataset

app = Flask( __name__ )
CORS( app )

option:None|str = None
engine:AbstractSearchEngine
judge_model:JudgeModel
corpus:list[dict]

@app.route( '/', methods=[ 'GET' ] )
def main():
    return f'Search engine ({option}) is up and running...'

@app.route( '/info', methods=[ 'GET' ] )
def info():
    return {
        'option': option
    }

# @app.route( '/search', methods=[ 'POST', 'GET' ] )
# if request.method == 'GET':
#     query = request.args.get( 'query' )
#     authors = request.args.get( 'authors' )
#     published = request.args.get( 'published' )

@app.route( '/search', methods=[ 'POST' ] )
def search():

    query, authors, authors = None, None, None

    params = request.get_json()
    query = params[ 'query' ]
    authors = params[ 'authors' ]
    published = params[ 'published' ]

    print( '<=', '================================' )
    print( '<=', query )
    print( '<=', authors )
    print( '<=', published )
    if not query or query.strip() == '':
        return { 'error': 'No query found.' }

    if authors != None:
        authors = cast( list[str], authors.split( ',' ) )

    results = engine.search( query=query, names=authors, period=published )  # type: ignore
    idocs = [ (idoc, rank) for idoc, rank, doc in results ]
    results = [ doc for idoc, rank, doc in results ]
    print( '=>', idocs )
    print( '=>', '================================' )

    return results

@app.route( '/judge', methods=[ 'POST' ] )
def judge():

    query, idocs = None, None

    params = request.get_json()
    query = params[ 'query' ]
    idocs = params[ 'idocs' ]

    print( '<=', '================================' )
    print( '<=', query )
    print( '<=', idocs )

    if not query or not idocs or len( idocs ) == 0:
        return { 'error': 'No query / documents found.' }

    results = []
    for idoc in idocs:
        title_summary = corpus[ idoc ][ 'title' ] + ' - ' + corpus[ idoc ][ 'summary' ] 
        result = judge_model.judge( query, title_summary )
        results.append( result == 'yes' )
    results = [ {i:r} for i, r in zip( idocs, results ) ]
    print( '=>', results )
    print( '=>', '================================' )

    return results

if __name__ == '__main__':

    option = None
    if len( sys.argv ) >= 2:
        option = sys.argv[ 1 ]

    match option:
        case 'arxiv-lemm-single-tfidf' |\
             'arxiv-lemm-2gram-tfidf' |\
             'arxiv-sentences-glove-retrained-bm25' |\
             'arxiv-sentences-bert-faiss' |\
             'arxiv-sentences-jina-faiss':
            engine = searchEngineFactory( option )
            judge_model = JudgeModel()
            corpus = Dataset().toList()

        case _:
            raise Exception( 'No valid option.' )

    print( main() )
    app.run( debug=True, port=5000 )