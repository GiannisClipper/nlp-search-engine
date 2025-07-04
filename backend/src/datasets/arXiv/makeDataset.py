import os
import json
import xml.etree.ElementTree as ET

from .Categories import Categories
from .settings import dataset_path, dataset_filename, catgs_filter

# check if the dataset already exists
if os.path.exists( dataset_filename ):
    print( f'{dataset_filename} already exists.' )
    exit(0)

# to hold the entries that will be retrieved
records = dict()

# iterate all categories - xml files
for catg_id, _ in Categories( catgs_filter ).toTuples():

    # compose the xml filename
    filename = f'{dataset_path}/downloads/{catg_id}-1100.xml'

    # check if the xml file exists
    if not os.path.exists( filename ):
        print( f'{filename} not exists.' )
        continue

    # read the content of the xml file
    print( f'Read {filename}...' )
    text = None
    with open( filename, "r", encoding="utf-8" ) as f:
        text = f.read()

    # parse the xml tags 
    print( f'Parse data...' )
    root = ET.fromstring( text )
    print( root.tag )

    for i, entry in enumerate( root.findall( '{http://www.w3.org/2005/Atom}entry' ) ):
        # print( entry, entry.tag )

        record = dict()

        id = entry.find( '{http://www.w3.org/2005/Atom}id' )
        if id != None:
            id = id.text

        title = entry.find( '{http://www.w3.org/2005/Atom}title' )
        if title !=None and title.text != None:
            lines = title.text.split( "\n" )
            lines = [ l.strip() for l in lines ]
            title = " ".join( lines )

        published = entry.find( '{http://www.w3.org/2005/Atom}published' )
        if published !=None and published.text != None:
            published = published.text[:10]

        updated = entry.find( '{http://www.w3.org/2005/Atom}updated' )
        if updated !=None and updated.text != None:
            updated = updated.text[:10]

        summary = entry.find( '{http://www.w3.org/2005/Atom}summary' )
        if summary != None and summary.text != None:
            lines = summary.text.split( "\n" )
            lines = [ l.strip() for l in lines ]
            summary = " ".join( lines )


        authors = []
        for author in entry.findall( '{http://www.w3.org/2005/Atom}author' ):
            name = author.find( '{http://www.w3.org/2005/Atom}name' )
            if name != None:
                authors.append( name.text )

        # add a new entry in mem storage
        if not id in records.keys():
            records[ id ] = {
                'id': id,
                'catg_ids': [ catg_id ],
                'title': title,
                'summary': summary,
                'authors': authors,
                'published': published,
                'updated': updated,
            }
        # add only category if an entry was already added (from other category)
        else:
            records[ id ][ 'catg_ids' ].append( catg_id )
 
        # for checking/ debugging purposes
        # if i % 100 == 0:
        #     print( f'{catg_id} ({i})')

        # for checking/ debugging purposes
        # print( f'\n{catg_id} ({i+1})')
        # print( 'id:', id )
        # print( 'Title:', title )
        # print( 'Summary:', summary )
        # print( 'Authors:', authors )
        # if i == 9:
        #     break

# save dataset as jsonl file (jsonl: each line represents a json object)
print( 'Total records:', len( records ) )
print( f'Save {dataset_filename}...' )
with open( dataset_filename, 'w', encoding="utf-8" ) as f:
    for key, record in records.items():
        json.dump( record, f )
        f.write( '\n' )
