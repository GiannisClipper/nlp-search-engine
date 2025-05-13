import time

class Timer:

    def __init__( self, start:bool=False ):
        self._start_time:None|float
        self._stop_time:None|float
        if start:
            self.start()

    def start( self ) -> None:
        self._start_time = time.time()
        self._stop_time = None

    def stop( self ) -> float:
        if self._start_time == None:
            raise Exception( 'Timer was not started.' )
        self._stop_time = time.time()
        return self.diff()

    def diff( self ) -> float:
        if self._start_time == None:
            raise Exception( 'Timer was not started.' )
        if self._stop_time == None:
            raise Exception( 'Timer was not stopped.' )
        return round( self._stop_time - self._start_time, 4 )
