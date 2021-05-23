import time

# =========================================================================== #

def recursionBuffer (buff) :
    def decorator (func) :
        def wrapper (n) :
            if n in buff : return buff[n]
            
            result = func(n)
            buff[n] = result
            return result
        return wrapper
    return decorator
# ........................................................................... #
def fib (n) :
    if type(n) != int :
        raise TypeError("Fibonacci sequence only defined on positive integers")
    
    if n < 0 :
        raise ValueError("Fibonacci sequence only defined on positive integers")
    
    if   n == 0 : return 0
    elif n == 1 : return 1
    else : return fib(n - 1) + fib(n - 2)
# ........................................................................... #
def takeTime (func) :
    def wrapper (*args, **kwargs) :
        tic = time.perf_counter()
        result = func(*args, **kwargs)
        toc = time.perf_counter()
        print(f"Time elapsed: {(toc - tic) * 1000:8.3f} ms")
        return result
    return wrapper
# --------------------------------------------------------------------------- #

buff = dict()
bufferedFib = recursionBuffer(buff)(fib)
recBufferedFib = takeTime( bufferedFib )


print( recBufferedFib(30) )

