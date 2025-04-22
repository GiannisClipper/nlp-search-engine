
from .Dataset import Dataset

print( f'Load corpus...' )
ds = Dataset()
records = ds.toList()
authors = []
for i in range( len( records ) ):
    authors += records[ i ][ 'authors' ]

print( 'All instances:', len( authors ) )
print( 'Unique names:', len( set( authors ) ) )
print( authors[:10] )

print( [ a for a in authors if 'Taylor' in a ] )