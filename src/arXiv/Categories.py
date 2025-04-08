# arXiv API User's Manual
# https://info.arxiv.org/help/api/user-manual.html

# Categories within Computer Science
# https://arxiv.org/archive/cs -> downloads/categories.txt

import re
from  . import settings

class Categories:

    def __init__( self, id_filter:list|tuple|None=None ):

        self._categories = []

        with open( settings.catgs_filename, 'r', encoding='utf-8' ) as f:
            for line in f:
                # e.g. match='cs.SI - Social and Information Networks '
                result = re.search( r"^\w{2}\.{1}\w{2}\s\-\s[\w\s]+", line )
                if result:
                    id, name = result.group( 0 ).split( ' - ' )
                    if not id_filter or id in id_filter:
                        self._categories.append( ( id, name ) )
    
    def toTuples( self ):
        return self._categories
