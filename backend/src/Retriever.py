import sys
from abc import ABC, abstractmethod
from typing import cast
from .helpers.typing import QueryAnalyzedType

from .TermsFilter import AbstractTermsFilter, OccuredTermsFilter, WeightedTermsFilter
from .TermsFilter import ClusteredTermsFilter, BM25TermsFilter, FaissTermsFilter
from .NameFilter import NamesFilter
from .PeriodFilter import PeriodFilter
from .helpers.Pickle import PickleLoader
from .helpers.DocViewer import DocViewer

# ----------------------------------------------------------- #
# Code to retrieve docs or sentences based on passing filters #
# ----------------------------------------------------------- #

# FROM ChatGPT: Example, Multiple Inheritance with Parameters
# Each class picks its own arguments and passes the rest up the chain.
# **kwargs carries the remaining parameters.
# All classes must use super() and accept **kwargs

# class A:
#     def __init__(self, a_param, **kwargs):
#         print(f"A's constructor: a_param = {a_param}")
#         super().__init__(**kwargs)

# class B(A):
#     def __init__(self, b_param, **kwargs):
#         print(f"B's constructor: b_param = {b_param}")
#         super().__init__(**kwargs)

# class C(A):
#     def __init__(self, c_param, **kwargs):
#         print(f"C's constructor: c_param = {c_param}")
#         super().__init__(**kwargs)

# class D(B, C):
#     def __init__(self, a_param, b_param, c_param):
#         print("D's constructor")
#         super().__init__(a_param=a_param, b_param=b_param, c_param=c_param)

class AbstractRetriever( ABC ):

    @abstractmethod
    def retrieve( self, **kwargs ) -> list[str]:
        pass


class PeriodRetriever( AbstractRetriever ):

    def __init__( self, periodFilter:PeriodFilter, **kwargs ):
        self.periodFilter = periodFilter
        super().__init__( **kwargs )

    def retrieve( self, period:str|None=None ) -> list[str]:

        # Initialize result including all data
        result = self.periodFilter.tags # e.g. tags -> '234', '235', ...

        if period:
            result = self.periodFilter( period )
        return result


class NamesRetriever( AbstractRetriever ):

    def __init__( self, namesFilter:NamesFilter, **kwargs ):
        self.namesFilter = namesFilter
        super().__init__( **kwargs )

    def retrieve( self, names:list[str]|None=None ) -> list[str]:

        # Initialize result including all data
        result = list( set( [ t.split('.')[0] for t in self.namesFilter.tags ] ) ) # e.g. tags -> '234.0', '234.1', '235.0', ...

        if names:
            result = list( self.namesFilter( names ) )
        return result


class TermsRetriever( AbstractRetriever ):

    def __init__( self, termsFilters:list[AbstractTermsFilter], **kwargs ):
        self.termsFilters = termsFilters
        super().__init__( **kwargs )

    def retrieve( self, query_analyzed:QueryAnalyzedType|None=None ) -> list[str]:

        # Initialize result with no data
        result = []

        if query_analyzed:
            temp_result = []
            for termsFilter in self.termsFilters:
                temp_result = termsFilter.filter( query_analyzed )
                temp_result = [ str(i) for i in temp_result ]
                result = list( set( result + temp_result ) )

        return result


class PeriodNamesTermsRetriever( PeriodRetriever, NamesRetriever, TermsRetriever ):

    def __init__( 
        self, 
        periodFilter:PeriodFilter, 
        namesFilter:NamesFilter, 
        termsFilters:list[AbstractTermsFilter],
        sentences_tags:list[str]|None=None
    ):
        super().__init__( periodFilter=periodFilter, namesFilter=namesFilter, termsFilters=termsFilters )
        self._sentences_tags = sentences_tags

    def retrieve( self, query_analyzed:QueryAnalyzedType|None=None, names:list[str]|None=None, period:str|None=None ) -> list[str]:

        # Filter by period
        period_result = PeriodRetriever.retrieve( self, period )
        if len( period_result ) == 0:
            return period_result

        # Filter by name
        names_result = NamesRetriever.retrieve( self, names )
        if len( names_result ) == 0:
            return names_result
        
        # Intersect period, names filter results
        period_names_result = list( set( period_result ) & set( names_result ) )
        if len( period_names_result ) == 0:
            return period_names_result

        # Filter by terms
        terms_result = TermsRetriever.retrieve( self, query_analyzed )
        # print( 'DEBUG-Retriever-terms_result', terms_result )
        if not terms_result:
            return period_names_result

        # Intersect period, names, terms filters
        if self._sentences_tags == None: # Documents matching
            result = list( set( period_names_result ) & set( terms_result ) )

        else: # Sentences matching
            # get sentence tags for isents
            sent_tags = [ self._sentences_tags[ int(r) ] for r in terms_result ]
            # mark via tags which match with period-names result
            sent_ok = [ True if t.split('.')[0] in period_names_result else False for t in sent_tags ] # both for idocs or isents 
            # filter isents based on previous mark
            result = [ res for res, ok in zip(terms_result, sent_ok) if ok ]

        return result


def retrieverFactory( option:str ) -> AbstractRetriever:

    match option:

        case 'arxiv-stemm-single':
            from .datasets.arXiv.Dataset import Dataset
            from .datasets.arXiv.settings import pickle_paths
            ds = Dataset()

            dates, tags = ds.toPublished()
            periodFilter = PeriodFilter( dates=dates, tags=tags )

            names, tags = ds.toAuthors()
            namesFilter = NamesFilter( names=names, tags=tags )

            index_descr = 'title-summary_lower-punct-specials-stops-stemm_single'
            index_filename = f"{pickle_paths[ 'indexes' ]}/{index_descr}.pkl"
            index = PickleLoader( index_filename ).load()
            corpus = ds.toList()
            termsFilters = [
                OccuredTermsFilter( index=index, threshold=0.5 ),
                WeightedTermsFilter( index=index, corpus=corpus, limit=200 )
            ]

            return PeriodNamesTermsRetriever( 
                periodFilter=periodFilter, 
                namesFilter=namesFilter, 
                termsFilters=termsFilters
            )

        case 'arxiv-lemm-single':
            from .datasets.arXiv.Dataset import Dataset
            from .datasets.arXiv.settings import pickle_paths
            ds = Dataset()

            dates, tags = ds.toPublished()
            periodFilter = PeriodFilter( dates=dates, tags=tags )

            names, tags = ds.toAuthors()
            namesFilter = NamesFilter( names=names, tags=tags )

            index_descr = 'title-summary_lower-punct-specials-stops-lemm_single'
            index_filename = f"{pickle_paths[ 'indexes' ]}/{index_descr}.pkl"
            index = PickleLoader( index_filename ).load()
            corpus = ds.toList()
            termsFilters = [
                OccuredTermsFilter( index=index, threshold=0.5 ),
                WeightedTermsFilter( index=index, corpus=corpus, limit=200 )
            ]

            return PeriodNamesTermsRetriever( 
                periodFilter=periodFilter, 
                namesFilter=namesFilter, 
                termsFilters=termsFilters
            )

        case 'arxiv-lemm-2gram':
            from .datasets.arXiv.Dataset import Dataset
            from .datasets.arXiv.settings import pickle_paths
            ds = Dataset()

            dates, tags = ds.toPublished()
            periodFilter = PeriodFilter( dates=dates, tags=tags )

            names, tags = ds.toAuthors()
            namesFilter = NamesFilter( names=names, tags=tags )

            index_descr = 'title-summary_lower-punct-specials-stops-lemm_2gram'
            index_filename = f"{pickle_paths[ 'indexes' ]}/{index_descr}.pkl"
            index = PickleLoader( index_filename ).load()
            corpus = ds.toList()
            termsFilters = [
                OccuredTermsFilter( index=index, threshold=0.5 ),
                WeightedTermsFilter( index=index, corpus=corpus, limit=200 )
            ]

            return PeriodNamesTermsRetriever( 
                periodFilter=periodFilter, 
                namesFilter=namesFilter, 
                termsFilters=termsFilters
            )

        case 'arxiv-sentences-jina-kmeans':
            from .datasets.arXiv.Dataset import Dataset
            from .datasets.arXiv.settings import pickle_paths
            ds = Dataset()

            dates, tags = ds.toPublished()
            periodFilter = PeriodFilter( dates=dates, tags=tags )

            names, tags = ds.toAuthors()
            namesFilter = NamesFilter( names=names, tags=tags )

            clusters_descr = 'sentences-jina-kmeans'
            clusters_filename = f"{pickle_paths[ 'clusters' ]}/{clusters_descr}.pkl"
            clustering_model = PickleLoader( clusters_filename ).load()
            termsFilters = [ ClusteredTermsFilter( model=clustering_model ) ]
            _, sentences_tags = ds.toSentences()

            return PeriodNamesTermsRetriever( 
                periodFilter=periodFilter, 
                namesFilter=namesFilter, 
                termsFilters=termsFilters,
                sentences_tags=sentences_tags
            )

        case 'arxiv-sentences-bm25':
            from .datasets.arXiv.Dataset import Dataset
            from .datasets.arXiv.settings import pickle_paths
            ds = Dataset()
            dates, tags = ds.toPublished()
            periodFilter = PeriodFilter( dates=dates, tags=tags )
            names, tags = ds.toAuthors()
            namesFilter = NamesFilter( names=names, tags=tags )
            sentences, tags = ds.toSentences()
            termsFilters = [ BM25TermsFilter( corpus=sentences ) ]
            _, sentences_tags = ds.toSentences()

            return PeriodNamesTermsRetriever( 
                periodFilter=periodFilter, 
                namesFilter=namesFilter, 
                termsFilters=termsFilters,
                sentences_tags=sentences_tags
            )

        case 'arxiv-sentences-jina-faiss':
            from .datasets.arXiv.Dataset import Dataset
            from .datasets.arXiv.settings import pickle_paths
            ds = Dataset()
            dates, tags = ds.toPublished()
            periodFilter = PeriodFilter( dates=dates, tags=tags )
            names, tags = ds.toAuthors()
            namesFilter = NamesFilter( names=names, tags=tags )
            descr = 'sentences-jina'
            filename = f"{pickle_paths[ 'corpus_repr' ]}/{descr}.pkl"
            embeddings = PickleLoader( filename ).load()
            termsFilters = [ FaissTermsFilter( sentences_embeddings=embeddings ) ]
            _, sentences_tags = ds.toSentences()

            return PeriodNamesTermsRetriever( 
                periodFilter=periodFilter, 
                namesFilter=namesFilter, 
                termsFilters=termsFilters,
                sentences_tags=sentences_tags
            )

        case 'arxiv-sentences-bert-faiss':
            from .datasets.arXiv.Dataset import Dataset
            from .datasets.arXiv.settings import pickle_paths
            ds = Dataset()
            dates, tags = ds.toPublished()
            periodFilter = PeriodFilter( dates=dates, tags=tags )
            names, tags = ds.toAuthors()
            namesFilter = NamesFilter( names=names, tags=tags )
            descr = 'sentences-bert'
            filename = f"{pickle_paths[ 'corpus_repr' ]}/{descr}.pkl"
            embeddings = PickleLoader( filename ).load()
            _, sentences_tags = ds.toSentences()
            termsFilters = [ FaissTermsFilter( sentences_embeddings=embeddings ) ]

            return PeriodNamesTermsRetriever( 
                periodFilter=periodFilter, 
                namesFilter=namesFilter, 
                termsFilters=termsFilters, 
                sentences_tags=sentences_tags
            )

        ###########
        # medical #
        ###########

        case 'medical-stemm-single':
            from .datasets.medical.Dataset import Dataset
            from .datasets.medical.settings import pickle_paths
            ds = Dataset()
            index_descr = 'title-summary_lower-punct-specials-stops-stemm_single'
            index_filename = f"{pickle_paths[ 'indexes' ]}/{index_descr}.pkl"
            index = PickleLoader( index_filename ).load()
            corpus = ds.toList()
            termsFilters:list[AbstractTermsFilter] = [
                OccuredTermsFilter( index=index, threshold=0.5 ),
                WeightedTermsFilter( index=index, corpus=corpus, limit=200 )
            ]
            return TermsRetriever( termsFilters=termsFilters )

        case 'medical-lemm-single':
            from .datasets.medical.Dataset import Dataset
            from .datasets.medical.settings import pickle_paths
            ds = Dataset()
            index_descr = 'title-summary_lower-punct-specials-stops-lemm_single'
            index_filename = f"{pickle_paths[ 'indexes' ]}/{index_descr}.pkl"
            index = PickleLoader( index_filename ).load()
            corpus = ds.toList()
            termsFilters:list[AbstractTermsFilter] = [
                OccuredTermsFilter( index=index, threshold=0.5 ),
                WeightedTermsFilter( index=index, corpus=corpus, limit=200 )
            ]
            return TermsRetriever( termsFilters=termsFilters )

        case 'medical-lemm-2gram':
            from .datasets.medical.Dataset import Dataset
            from .datasets.medical.settings import pickle_paths
            ds = Dataset()
            index_descr = 'title-summary_lower-punct-specials-stops-lemm_2gram'
            index_filename = f"{pickle_paths[ 'indexes' ]}/{index_descr}.pkl"
            index = PickleLoader( index_filename ).load()
            corpus = ds.toList()
            termsFilters:list[AbstractTermsFilter] = [
                OccuredTermsFilter( index=index, threshold=0.5 ),
                WeightedTermsFilter( index=index, corpus=corpus, limit=200 )
            ]
            return TermsRetriever( termsFilters=termsFilters )

        case 'medical-sentences-bm25':
            from .datasets.medical.Dataset import Dataset
            sentences, tags = Dataset().toSentences()
            termsFilters = [ BM25TermsFilter( corpus=sentences ) ]
            return TermsRetriever( termsFilters=termsFilters )

        case 'medical-sentences-bert-kmeans':
            from .datasets.medical.settings import pickle_paths
            clusters_descr = 'sentences-bert-kmeans'
            clusters_filename = f"{pickle_paths[ 'clusters' ]}/{clusters_descr}.pkl"
            clustering_model = PickleLoader( clusters_filename ).load()
            # from .datasets.medical.Dataset import Dataset
            # sentences, tags = Dataset().toSentences()
            termsFilters = [
                ClusteredTermsFilter( model=clustering_model ),
                # BM25TermsFilter( sentences )
            ]
            return TermsRetriever( termsFilters=termsFilters )

        case 'medical-sentences-bert-faiss':
            from .datasets.medical.settings import pickle_paths
            descr = 'sentences-bert'
            filename = f"{pickle_paths[ 'corpus_repr' ]}/{descr}.pkl"
            embeddings = PickleLoader( filename ).load()
            termsFilters = [ FaissTermsFilter( sentences_embeddings=embeddings ) ]
            return TermsRetriever( termsFilters=termsFilters )

        case 'medical-sentences-bert-retrained-faiss':
            from .datasets.medical.settings import pickle_paths
            descr = 'sentences-bert-retrained'
            filename = f"{pickle_paths[ 'corpus_repr' ]}/{descr}.pkl"
            embeddings = PickleLoader( filename ).load()
            termsFilters = [ FaissTermsFilter( sentences_embeddings=embeddings ) ]
            return TermsRetriever( termsFilters=termsFilters )

        case 'medical-sentences-jina-faiss':
            from .datasets.medical.settings import pickle_paths
            descr = 'sentences-jina'
            filename = f"{pickle_paths[ 'corpus_repr' ]}/{descr}.pkl"
            embeddings = PickleLoader( filename ).load()
            termsFilters = [ FaissTermsFilter( sentences_embeddings=embeddings ) ]
            return TermsRetriever( termsFilters=termsFilters )

        case _:
            raise Exception( 'retrieverFactory(): No valid option.' )


# +----------------------------------------+
# | For development and debugging purposes |
# +----------------------------------------+

# RUN: python -m src.Retriever
if __name__ == "__main__": 

    # Initialize involved instances

    retriever = retrieverFactory( 'arxiv-lemm-single' )

    from .datasets.arXiv.Dataset import Dataset
    ds = Dataset()
    corpus = ds.toList()
    docViewer = DocViewer( corpus=corpus )

    # Set testing parameters

    params:dict[str,str|None] = {
        'terms': None,
        'names': None,
        'period': None
    }
    for i in range( 1, len( sys.argv ) ):
        key, value = sys.argv[ i ].split( '=' )
        params[ key ] = value

    if len( sys.argv ) == 1:
        params[ 'terms' ] = 'probability,geometric,poisson,distribution'
        params[ 'names' ] = 'michael jordan'
        params[ 'period' ] = '2012-01-01,2012-12-31'

    # Perform filterings

    terms = None
    if params[ 'terms' ]:
        terms = params[ 'terms' ].split(',')
        query_analyzed = cast( QueryAnalyzedType, { 'terms': terms } )
        result = retriever.retrieve( query_analyzed=query_analyzed )
        print( '-------------------------------------------------------------' )
        print( terms, len( result ), result[:5] )
        for res in result[:5]:
            docViewer.view( int( res ) )

    names = None
    if params[ 'names' ]:
        names = params[ 'names' ].split(',')
        result = retriever.retrieve( names=names )
        print( '-------------------------------------------------------------' )
        print( names, len( result ), result[:5] )
        for res in result[:5]:
            docViewer.view( int( res ) )

    period = None
    if params[ 'period' ]:
        period = params[ 'period' ]
        result = retriever.retrieve( period=period )
        print( '-------------------------------------------------------------' )
        print( period, len( result ), result[:5] )
        for res in result[:5]:
            docViewer.view( int( res ) )

    query_analyzed = cast( QueryAnalyzedType, { 'terms': terms } )
    result = retriever.retrieve( query_analyzed=query_analyzed, names=names, period=period )
    print( '-------------------------------------------------------------' )
    print( terms, names, period, len( result ), result[:5] )
    for res in result[:5]:
        docViewer.view( int( res ) )
