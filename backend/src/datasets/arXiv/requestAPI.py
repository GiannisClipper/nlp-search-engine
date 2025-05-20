import os
import urllib.request as request
import time

from .settings import dataset_path, catgs_filter
from .Categories import Categories

# arXiv API User's Manual
# https://info.arxiv.org/help/api/user-manual.html

has_requested = False
for catg_id, _ in Categories( catgs_filter ).toTuples():

    # compose filename
    filename = f'{dataset_path}/downloads/{catg_id}-1100.xml'

    # no request whenever already has done
    if os.path.exists( filename ):
        print( f'{filename} already exists.' )
        continue

    # time delay due to api limitations
    if has_requested:
        print( f'Waiting 30 secs...' )
        time.sleep( 30 )

    # compose the url to request
    url_string = f'http://export.arxiv.org/api/query?search_query=all:{catg_id}&start=0&max_results=1100'

    print( f'Request {url_string}...' )
    has_requested = True
    with request.urlopen( url_string ) as url:
        response = url.read()

        # convert response type from <class 'bytes'> to text
        text = str( response, 'utf-8' ) 

        # save the api response into a local file
        print( f'Save {filename}...' )
        with open( filename, "w", encoding="utf-8" ) as f:
            f.write( text )
