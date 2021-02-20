# =========================================================================== #
# problem 1

import time

# --------------------------------------------------------------------------- #

def takeTime (func) :
    def wrapper (*args, **kwargs) :
        tic = time.perf_counter()
        result = func(*args, **kwargs)
        toc = time.perf_counter()
        print(f"Time elapsed: {(toc - tic) * 1000:8.3f} ms")
        return result
    return wrapper

# ........................................................................... #
    
@takeTime
def arraySum(array, start = 0, stop=None) :
    return sum(array[start:stop])

# --------------------------------------------------------------------------- #

a = [i for i in range(20000)]

print( "Array sum:", arraySum(a) )
print()

# =========================================================================== #
# problem 2

import matplotlib.pyplot as plt

# --------------------------------------------------------------------------- #

def fib (n) :
    if type(n) != int :
        raise TypeError("Fibonacci sequence only defined on positive integers")
    
    if n < 0 :
        raise ValueError("Fibonacci sequence only defined on positive integers")
        
    # NOTE:
    # You might want to combine both the above tests into a single if block
    # and make use of Python's short circuiting. In terms of runtime, this is
    # in fact better. However note that convention is that you return 
    # *different* error type: TypeError or ValueError.
    
    if   n == 0 : return 0
    elif n == 1 : return 1
    else : return fib(n - 1) + fib(n - 2)

# There are two calls to fib in each layer of recursion, so the number of open
# instances grows exponentially with recursion depth. We reduce N only by 1
# (or by two, but this doesn't really matter) per recursion step, so we have
# a recursion tree of depth N:

#                             N               \
#                          /     \            |
#                        N-1    N-2           |
#                      /    \  /   \           > N levels ==> 2^N leafs
#                          .....              |
#                  0 1 0 0 0 1 .... 0 1 1    /

# At the bottom of this recursion tree we have (approximately) 2^N leafs, so
# the overall time complexity is in exp(N)

# ........................................................................... #

def fibInternalBuffer (n) :
    if type(n) != int :
        raise TypeError("Fibonacci sequence only defined on positive integers")
    
    if n < 0 :
        raise ValueError("Fibonacci sequence only defined on positive integers")
    
    results = dict()
    
    def worker (n) :
        nonlocal results
        
        if   n == 0 : return 0
        elif n == 1 : return 1
        else :
            if n in results :
                return results[n]
            else :
                result     = worker(n - 2) + worker(n - 1)
                results[n] = result
                return result
    
    return worker(n)

# For this approach, each F_k needs to be computed exactly once. After that, it
# is in the buffer and essentially comes for free. For F_N we need F_{N - 1},
# for which we need F_{N - 2}, for which we need F_{N - 3}, for which ...
# So, for each n in range(N) we compute exactly one expression: This algorithm
# is in O(N)

# ........................................................................... #

def resultBuffer (func) :
    buffer = dict()
    
    def wrapper (n) :
        if n in buffer :
            return buffer[n]
        else :
            result = func(n)
            buffer[n] = result
            return result
    
    return wrapper

# Let's consider what happens if we set
#   fib = resultBuffer(fib)
# The original (undecorated) function fib is somewhere in memory. The local
# symbol func holds a reference to exactly this memory location where the
# undecorated code resides.
# Only the symbol fib is overwritten, and now uses the buffer *in the very 
# first step only*
# If the result was in the buffer -- return the stored result (massive speedup)
# If not -- use func, i.e. the old code with no speedup whatsoever.

# The internal buffer first constructs an empty dict. This isn't instantaneous,
# and this overhead costs some 10 ns

# ........................................................................... #

def takeAndReturnTime (func) :
    def wrapper (*args, **kwargs) :
        tic = time.perf_counter()
        result = func(*args, **kwargs)
        toc = time.perf_counter()
        return toc - tic, result
    return wrapper

# --------------------------------------------------------------------------- #

timedFib                = takeTime(             fib )
timedFibInternalBuffer  = takeTime(fibInternalBuffer)
timedFibResultBuffer    = takeTime(resultBuffer(fib))

N = 30

print("FIBONACCI")

print("naive approach")
print( timedFib(N) )
print()

print("internal buffer, first call")
print( timedFibInternalBuffer(N) )
print()

print("internal buffer, second call")
print( timedFibInternalBuffer(N) )
print()

print("result buffer, first call")
print( timedFibResultBuffer(N) )
print()

print("result buffer, second call")
print( timedFibResultBuffer(N) )
print()

print("Measuring Runtime Behaviour. This may take some time...")


N = 15

timedFib                = takeAndReturnTime(             fib )
timedFibInternalBuffer  = takeAndReturnTime(fibInternalBuffer)
timedFibResultBuffer    = takeAndReturnTime(resultBuffer(fib))

timesPure           = [timedFib(n)[0] for n in range(N)]
timesInternalBuffer = [timedFibInternalBuffer(n)[0] for n in range(N)]
timesResultBuffer1  = [timedFibResultBuffer(n)[0] for n in range(N)]
timesResultBuffer2  = [timedFibResultBuffer(n)[0] for n in range(N)]

plt.plot(range(N), timesPure, label="naive approach")
plt.plot(range(N), timesInternalBuffer, label="Internal buffer")
plt.plot(range(N), timesResultBuffer1, label="Result buffer, first invocation")
plt.plot(range(N), timesResultBuffer2, label="Result buffer, second invocation")

plt.yscale('log')
plt.legend()

plt.title('Comparison of Time Requirement for Different Evaluation Methods\n(log scale)')
plt.xlabel('$N$')
plt.ylabel('time in s')

plt.show()

# this first plot shows almost identical results for the naive approach and the
# result buffer solution. This is not surprising as they are essentially the 
# same code. The buffer code causes some negligible overhead, but nothing 
# on the same time scale as computing the final result.
# The plot is log scale, so the straight lines in this projection confirm that
# we really are in O(exp(N)) for this.
# The other two lines are WAY better, by orders of magnitude.
# The result buffer code after second invocation performs even better than the
# internal buffer solution, but this comes at the price of having to compute
# the result the hard way a first time.
# Of course, we could stack the two approaches together to get the best of both
# worlds.

plt.plot(range(N), timesResultBuffer2, label="Internal buffer")

plt.title('Time for Result Buffer, second invocation\n(linear scale)')
plt.xlabel('$N$')
plt.ylabel('time in s')

plt.show()

# Notably, the result buffer code for second invocation is near constant in 
# time, which is not  surprising: we only do lookup old results.
# For really BIG dict's, there is some overhead involved for finding the 
# correct result in the dict, but thanks to the smart way they implemented
# dicts, we hardly notice this at all.
# We also see how noisy the graph is. The Fluctuations can more than  to ten 
# times larger than the mean dict lookup time itself

N = 100

timedFibInternalBuffer  = takeAndReturnTime(fibInternalBuffer)                 # reset the internal buffer
timesInternalBuffer = [timedFibInternalBuffer(n)[0] for n in range(N)]

plt.plot(range(N), timesInternalBuffer, label="Internal buffer")

plt.title('Time for Internal Buffer\n(linear scale)')
plt.xlabel('$N$')
plt.ylabel('time in s')

plt.show()

# Finally, this confirms that our internal buffer code works in linear time.
# Again: heavy fluctuations

print("... done")

# =========================================================================== #
# Bonus content

# in functools, the result buffer stuff is prebuilt and ready to use:
# functools.lru_cache does essentially what we've seen above.

import functools

@takeAndReturnTime
@functools.lru_cache
def fib2 (n) :
    if type(n) != int :
        raise TypeError("Fibonacci sequence only defined on positive integers")
    
    if n < 0 :
        raise ValueError("Fibonacci sequence only defined on positive integers")
    
    if   n == 0 : return 0
    elif n == 1 : return 1
    else : return fib(n - 1) + fib(n - 2)

print( fib2(35) )
print( fib2(35) )