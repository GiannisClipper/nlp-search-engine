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

    from .arXiv.Dataset import Dataset

    ds = Dataset()
    viewer = DocViewer( corpus=ds.toList() )
    for i in range( 1001, 1003 ):
        viewer.view( i )
