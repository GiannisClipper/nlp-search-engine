import sys

class DocViewer:

    def __init__( self, corpus:list[dict] ):
        self._corpus = corpus

    def view( self, idoc ):
        doc = self._corpus[ idoc ]
        print( f"({idoc})" )
        print( f"Id: {doc[ 'id' ]}" )
        print( f"Title: {doc[ 'title' ]}" )
        print( f"Authors: {doc[ 'authors' ]}" )
        print( f"Summary: {doc[ 'summary' ]}" )
        print( f"Published: {doc[ 'published' ]}" )
        print( f"Categories: {doc[ 'catg_ids' ]}" )
        

# RUN: python -m src.DocViewer
if __name__ == "__main__": 

    docids = [ '1001', '1002' ]
    if len( sys.argv ) > 1:
        docids = sys.argv[ 1 ].split(',')

    from ..datasets.arXiv.Dataset import Dataset
    viewer = DocViewer( corpus=Dataset().toList() )
    for docid in docids:
        print()
        viewer.view( int(docid) )
