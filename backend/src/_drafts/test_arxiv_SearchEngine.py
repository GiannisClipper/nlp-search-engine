from ..SearchEngine import searchEngineFactory

# Reviews on operating systems
# Anything about UNIX and it's history?
# Do you remember MS DOS?

# Info about cloud computing and distributed systems...
# What the meaning of SaaS, PaaS and IaaS...
# Any cost analysis and comparison between cloud systems and local servers?

# About Javascript and frontend frameworks?
# What about the evolution of web browsers... 
# Research regarding the localstorage feature in web development...

# What about mobile applications?
# The role of JAVA in android development...
# Is React Native a good alternative for mobile programming?

# Papers discussing databases (either sql or nosql)...
# A description about DBMS...
# What are the differences between databases, data warehouses and data lakes?

# Are ML algoriths useful in the healthcare sector?
# Machine learning and natural language processing...
# Any general review about AI?

# Technological innovations within smart city context?
# The role of sensors and the IoT in smart cities...
# Are digital technologies involved into sutainability goals?

# Available stuff on neural networks and image processing?

# About information retrieval and natural language processing...

# The TCP/IP protocol in networking.
# Any info about SDN approach in networks?
# Information and communication technologies in shipping industry.

query = "TCP/IP in networking"

options = [
    # 'arxiv-stemm-single-count',
    # 'arxiv-lemm-single-tfidf',
    'arxiv-lemm-2gram-tfidf',
    # 'arxiv-sentences-glove-retrained-bm25',
    # 'arxiv-sentences-jina-bm25',
    # 'arxiv-sentences-jina-faiss',
    'arxiv-sentences-bert-faiss',
]

total_results = []

for option in options:
    engine = searchEngineFactory( option )
    results = engine.search( query ) # results is a list of tuples [ ('docid', rate), ... ]
    total_results.append( [ int(r[0]) for r in results[:100] ] )

for res in total_results:
    print( len(res) )
r1, r2 = total_results
print( r1 )
print( r2 )
print( len(set(r1) & set(r2)) )

# from ..datasets.arXiv.Dataset import Dataset
# corpus = Dataset().toList()

# for i, option in enumerate( options ):
#     print(option)
#     for res in total_results[i]:
#         doc = corpus[ int(res) ]
#         print( f"{doc['id']} {doc['catg_ids']} {doc['title']}" )
