from typing import TypedDict, NewType, List, Dict
from scipy.sparse import spmatrix

QueryAnalyzedType = TypedDict( 'QueryAnalyzedType', { 'terms': list[str], 'repr': spmatrix } )

# CustomType = NewType( 'CustomType', List[Dict[str,int]] )
# test:CustomType = CustomType( [{ 'key': 56 }] )
