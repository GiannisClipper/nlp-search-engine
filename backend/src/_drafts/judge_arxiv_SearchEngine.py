from ..SearchEngine import searchEngineFactory
from ..datasets.arXiv.Dataset import Dataset
from ..JudgeTester import JudgeTester

# Regarding operating systems...
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

corpus = Dataset().toList()

query = "Do you remember MS DOS?"
query = "Regarding operating systems..."

# option = 'arxiv-lemm-2gram-tfidf'
option = 'arxiv-sentences-bert-faiss'

total_results = []

engine = searchEngineFactory( option )
results = engine.search( query )[:10] # results is a list of tuples [ ('docid', rate), ... ]
print( results )
results = [ int(r[0]) for r in results ]
titles = [ corpus[ idoc ][ 'title' ] for idoc in results ] 
titles_summaries = [ corpus[ idoc ][ 'title' ] + '-' + corpus[ idoc ][ 'summary' ] for idoc in results ] 
print( '\n'.join( titles ) )
print( '------------------------------------' )
print( f'Query: {query}' )
tester = JudgeTester()

counter = 0
yes = 0
no = 0
for answer in titles_summaries:
    result = tester.judge( query, answer )
    counter += 1
    yes += 1 if result == 'yes' else 0
    no += 1 if result == 'no' else 0
    print( f"Answer #{counter}: {answer}" )
    print( f"Relevant ?: {result}" )
    print( '------------------------------------' )
print( f'{counter} answers, {yes} relevant(s), {no} non relevant(s)' )

# from ..datasets.arXiv.Dataset import Dataset
# corpus = Dataset().toList()

# for i, option in enumerate( options ):
#     print(option)
#     for res in total_results[i]:
#         doc = corpus[ int(res) ]
#         print( f"{doc['id']} {doc['catg_ids']} {doc['title']}" )
