from abc import ABC, abstractmethod
import math

class AbstractTermsFilter( ABC ):

    def __init__( 
        self, 
        corpus:list[dict],
        index:dict,
    ):
        self._corpus = corpus
        self._index = index

    @abstractmethod
    def filter( self, terms:list[str]|tuple[str,...] ) -> list[str]:
        pass

    def __str__( self ):
        return self.__class__


class OccuredTermsFilter( AbstractTermsFilter ):

    def filter( self, terms:list[str]|tuple[str,...], threshold:float=0.5 )-> list[str]:

        doc_stats = {}
        for term in terms:

            # if term is included in index
            if term in self._index:

                # iterate all docs the term is ocuured within
                for key in self._index[ term ].keys():
                    if key not in doc_stats:
                        doc_stats[ key ] = 0
                    doc_stats[ key ] += 1

        # select docs with including the half terms at least
        minTerms = len( terms ) * threshold if threshold > 0.0 else 1
        result = [ ( key, stat ) for key, stat in doc_stats.items() if stat >= minTerms ]
        result = [ k for k, _ in result ]
        return result


class WeightedTermsFilter( AbstractTermsFilter ):

    def filter( self, terms:list[str]|tuple[str,...], limit:int=100 ) -> list[str]:

        term_weights = {}
        doc_stats = {}
        for term in terms:

            # if term is included in index
            if term in self._index:

                # compute a term weight like idf (+1 to avoid zero division)
                term_weights[ term ] = math.log10( len( self._corpus ) / ( 1 + len( self._index[ term ].keys() ) ) )

                # iterate all docs the term is ocurred within
                for key in self._index[ term ].keys():
                    if key not in doc_stats:
                        doc_stats[ key ] = 0
                    doc_stats[ key ] += term_weights[ term ]

        # sort the documents' weights and get the top to check similarities
        result = [ ( key, weight ) for key, weight in doc_stats.items() ]
        result.sort( key=lambda x: x[1], reverse=True )
        result = result[:limit] if limit > 0 and len( result ) > limit else result
        result = [ k for k, _ in result ]
        return result


class TermsFilter( AbstractTermsFilter ):

    def filter( self, terms:list[str]|tuple[str,...], threshold:float=0.5, limit:int=100 ) -> list[str]:

        result1 = OccuredTermsFilter( self._corpus, self._index ).filter( terms, threshold=threshold )
        result2 = WeightedTermsFilter( self._corpus, self._index ).filter( terms, limit=limit )
        # print( 'result1:', result1, 'result2:', result2 )
        return list( set( result1 + result2 ) )

