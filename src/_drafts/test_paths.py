# to run: $ python3 -m tests.test_paths

import os
print( 'current working directory:', os.getcwd() )
print( 'file path:', os.path.dirname( os.path.realpath( __file__ ) ) )
