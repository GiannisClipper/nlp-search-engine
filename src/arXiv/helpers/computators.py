import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def compute_similarities0( single_repr, corpus_repr ):

    similarities = []
    for i in range( corpus_repr.shape[ 0 ] ):
        sim = cosine_similarity( single_repr.reshape( 1,-1 ), corpus_repr[i].reshape( 1,-1 ) )
        similarities.append( sim[0][0] )
    return similarities


def compute_similarities1( single_repr, corpus_repr ): 
    # REMARK: 3 to 5 times slower than compute_similarities0

    similarities = []

    # keep in dict only pos with non zero values
    single_repr = single_repr.toarray() # to convert Compressed Sparse Row matrix 
    print(single_repr.shape, single_repr)
    single_repr_d = dict( [ ( i, r ) for i, r in enumerate( single_repr[ 0 ] ) if r != 0 ] )
    for i in range( corpus_repr.shape[ 0 ] ):

        # keep in dict only pos with non zero values
        corpus_repr[i] = corpus_repr[i].reshape( 1,-1 )
        text_repr_d = dict( [ ( i, r ) for i, r in enumerate( corpus_repr[i] ) if r != 0 ] )

        # unify the positions from both representations
        unified = set( list( single_repr_d.keys() ) + list( text_repr_d.keys() ) )

        # get the values from the unified positions only
        x = np.array( [ single_repr[ 0 ][ j ] for j in unified ] )
        y = np.array([ corpus_repr[ i ][ j ] for j in unified ] )

        sim = cosine_similarity( x.reshape( 1,-1 ), y.reshape( 1,-1 ) )
        similarities.append( sim[0][0] )

    return similarities

# with compute_similarities0 (4.7 secs) 
# ('http://arxiv.org/abs/2003.00054v1', ['cs.DB'], np.float64(0.36291305263530205))
# ('http://arxiv.org/abs/1703.06348v1', ['cs.DB'], np.float64(0.2900236464228073))
# ('http://arxiv.org/abs/2412.18143v1', ['cs.DB'], np.float64(0.28432769951674614))
# ('http://arxiv.org/abs/1401.2101v1', ['cs.DB'], np.float64(0.2825920489632685))
# ('http://arxiv.org/abs/1908.08113v1', ['cs.CL'], np.float64(0.24562069355189417))
# ('http://arxiv.org/abs/1909.03291v1', ['cs.DB'], np.float64(0.24185067075739353))
# ('http://arxiv.org/abs/1606.02669v1', ['cs.SE'], np.float64(0.21697436744340542))
# ('http://arxiv.org/abs/1506.07950v1', ['cs.DB'], np.float64(0.19916753670822376))
# ('http://arxiv.org/abs/cs/0110052v1', ['cs.DB'], np.float64(0.19837667961082875))
# ('http://arxiv.org/abs/2306.13486v1', ['cs.DB'], np.float64(0.18940084638548435))
# ('http://arxiv.org/abs/0906.0328v1', ['cs.DS'], np.float64(0.1867123546264957))
# ('http://arxiv.org/abs/1907.05618v1', ['cs.DB'], np.float64(0.18248054711409337))
# ('http://arxiv.org/abs/1411.2160v1', ['cs.DC'], np.float64(0.18008588213272111))
# ('http://arxiv.org/abs/1001.1276v1', ['cs.DB'], np.float64(0.17992179527765123))
# ('http://arxiv.org/abs/1610.06084v1', ['cs.DB'], np.float64(0.1787396980310451))
# ('http://arxiv.org/abs/cs/0701173v1', ['cs.DB'], np.float64(0.17688847972901092))
# ('http://arxiv.org/abs/2303.16577v1', ['cs.DB'], np.float64(0.17568719163068322))
# ('http://arxiv.org/abs/2411.16742v1', ['cs.DB'], np.float64(0.17408197811906675))
# ('http://arxiv.org/abs/2006.08842v1', ['cs.DB'], np.float64(0.17312599515471003))
# ('http://arxiv.org/abs/2301.10673v2', ['cs.DB'], np.float64(0.16891804070796199))

# with compute_similarities1 (25.2 secs)
# ('http://arxiv.org/abs/2003.00054v1', ['cs.DB'], np.float64(0.36291305263530205))
# ('http://arxiv.org/abs/1703.06348v1', ['cs.DB'], np.float64(0.2900236464228073))
# ('http://arxiv.org/abs/2412.18143v1', ['cs.DB'], np.float64(0.2843276995167461))
# ('http://arxiv.org/abs/1401.2101v1', ['cs.DB'], np.float64(0.2825920489632685))
# ('http://arxiv.org/abs/1908.08113v1', ['cs.CL'], np.float64(0.24562069355189417))
# ('http://arxiv.org/abs/1909.03291v1', ['cs.DB'], np.float64(0.24185067075739353))
# ('http://arxiv.org/abs/1606.02669v1', ['cs.SE'], np.float64(0.21697436744340542))
# ('http://arxiv.org/abs/1506.07950v1', ['cs.DB'], np.float64(0.19916753670822376))
# ('http://arxiv.org/abs/cs/0110052v1', ['cs.DB'], np.float64(0.19837667961082872))
# ('http://arxiv.org/abs/2306.13486v1', ['cs.DB'], np.float64(0.18940084638548435))
# ('http://arxiv.org/abs/0906.0328v1', ['cs.DS'], np.float64(0.18671235462649569))
# ('http://arxiv.org/abs/1907.05618v1', ['cs.DB'], np.float64(0.18248054711409337))
# ('http://arxiv.org/abs/1411.2160v1', ['cs.DC'], np.float64(0.18008588213272114))
# ('http://arxiv.org/abs/1001.1276v1', ['cs.DB'], np.float64(0.17992179527765126))
# ('http://arxiv.org/abs/1610.06084v1', ['cs.DB'], np.float64(0.1787396980310451))
# ('http://arxiv.org/abs/cs/0701173v1', ['cs.DB'], np.float64(0.17688847972901092))
# ('http://arxiv.org/abs/2303.16577v1', ['cs.DB'], np.float64(0.17568719163068322))
# ('http://arxiv.org/abs/2411.16742v1', ['cs.DB'], np.float64(0.17408197811906678))
# ('http://arxiv.org/abs/2006.08842v1', ['cs.DB'], np.float64(0.17312599515471003))
# ('http://arxiv.org/abs/2301.10673v2', ['cs.DB'], np.float64(0.16891804070796199))
