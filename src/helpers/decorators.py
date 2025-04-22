import time

def with_time_counter( func ):

    def wrapper( message=None, *args, **kwargs ):
        start_time = time.time()
        message = message if message else f'Running {func.__name__} (with_time_counter)...'
        print( message, end=' ' )
        result = func( *args, **kwargs )
        print( f'({round(time.time()-start_time, 1)} secs)', end=' ' )
        return result

    return wrapper
