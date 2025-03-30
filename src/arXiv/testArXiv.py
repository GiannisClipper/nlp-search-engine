import urllib.request as libreq
import xml.etree.ElementTree as ET

# arXiv API User's Manual
# https://info.arxiv.org/help/api/user-manual.html
# (...)
# the titles of an article can be searched, as well as the author list, abstracts, comments and journal reference. 
# To search one of these fields, we simply prepend the field prefix followed by a colon to our search term. 
# For example, suppose we wanted to find all articles by the author Adrian Del Maestro. 
# We could construct the following query http://export.arxiv.org/api/query?search_query=au:del_maestro
# This returns nine results. The following table lists the field prefixes for all the fields that can be searched.
# prefix explanation
# ti	 Title
# au	 Author
# abs	 Abstract
# co	 Comment
# jr	 Journal Reference
# cat	 Subject Category
# rn	 Report Number
# id	 Id (use id_list instead)
# all	 All of the above
# (...)
# The API provides one date filter, submittedDate, that allow you to select data within a given date range of when the data was submitted to arXiv. 
# The expected format is [YYYYMMDDTTTT+TO+YYYYMMDDTTTT] were the TTTT is provided in 24 hour time to the minute, in GMT. 
# We could construct the following query using submittedDate.
# https://export.arxiv.org/api/query?search_query=au:del_maestro+AND+submittedDate:[202301010600+TO+202401010600]
# (...)

url_string = 'http://export.arxiv.org/api/query?search_query=all:"Theoretical%20Economics"&start=0&max_results=10'

with libreq.urlopen( url_string ) as url:
    response = url.read()
    root = ET.fromstring( response )
    print( root.tag )

    # for child in root:
    #   if child.tag == '{http://www.w3.org/2005/Atom}entry':
    #     print( child, child.tag )

    for entry in root.findall( '{http://www.w3.org/2005/Atom}entry' ):
        print( entry, entry.tag )

    print()
    entry = root.find( '{http://www.w3.org/2005/Atom}entry' )
    if entry != None:
      for child in entry:
           print( child, child.tag )

    # entry = root.find( '{http://www.w3.org/2005/Atom}entry' )
    # if entry != None:
    for i, entry in enumerate( root.findall( '{http://www.w3.org/2005/Atom}entry' ) ):
          title = entry.find( '{http://www.w3.org/2005/Atom}title' )
          if title != None:
              title = title.text

          summary = entry.find( '{http://www.w3.org/2005/Atom}summary' )
          if summary != None:
              summary = summary.text

          print( '\n#', i+1 )
          print( 'Title:', title )
          print( 'Summary:', summary )
