# =========================================================================== #
# problem 1

# There is one loop of length N = len(data).
# In this loop, we call min on an array of size N - k.
# To evaluate min, we need to "touch" each item within the array at least once,
#   so min(a) is in O(len(a)).
# k goes from 0 to N-1, so on average, we call min on an array of length N/2.
# Thus, the algorithm is in O(N²/2) ~ O(N²)

# =========================================================================== #
# problem 2

# To compute a matrix with N rows and N colums, we need to compute N² numbers.
# Thus, we can never get below O(N²).
# 
# Computing the matrix product is usually done as in the definition:
#   (A * B)_ij = sum_k A_ik * B_kj
# So we have two loops over rows i and colums j, and one additional loop over
# summands k.
# Optimization of matrix multiplication usually is done close to the hardware,
# using knowledge about processor architecture or chances for parallelization.
# 
# In particular parallelization is important here. GPUs are essentially 
# glorified parallel processors with the sole intent of doing tons of matrix
# multiplications.

# =========================================================================== #
# problem 3

# The recursive approach does not reduce N by a factor b, but only subtracts a
# constant value 1. Thus, the algorithm cannot be matched to the Master Theorem
# time formula:
#     T(N) = a * T(N/b) + f(N)
# 
# There are no branches that lead to different tracks in the recursion. In other
# words, this recursive algorithm behaves like a loop:
#     for i in range(N) : ...
# Thus, runtime is in O(N)
# 
# Runtime overhead for function calls is immense and can slow down your code by
# an order of magnitude. Since the non-recursive code is mechanically identical,
# it is to be preferred.

# =========================================================================== #
# problem 4

# First, let's understand how this algorithm tries to sum up the values:
# It selects disjoint sublists of at most 2 elements which it then sums.
# The sums of these sublists are then combined into a new sublist which is 
# summed up following the same idea.
# Picture:
# 
#   1    2    3    4    5    6
#   \    /    \    /    \    /
#    \ 3/      \ 7/      \11/
#      \         /         |
#       \   10  /         11
#            \             /
#             \     21    /
# 
# Let's for a momemt assume that there are N = 2^k summands in the list. Then 
# each recursion step reduces the number of summands by a factor of 2. We carry
# on until there are only two summands left. Tham means, we go through 
#     k = log_2(N)
# stages of this summation game. In the k-th stage, there are N/(2^k) summands,
# so:
#     N_sums = sum_k=1^log_2(N) N/(2^k) = N sum_k=1^log_2(N) 1/(2^k) = N
# 
# When N is not a power of 2, we "hand through" a number one layer. This is 
# equivalent to adding a 0 to our list; in other words, the cases with fewer
# than 2^k terms are equivalent to such where there ARE 2^k and the "missing 
# ones" are just zeros.
# 
# Now let's apply the master theorem to find the same result:
# 
# Regard line 15: there are two calls to listRecSum, so a = 2.
# Each time, we send only half the data (due to splitting the list at its 
#   midpoint), so b = 2.
# From this, we get c_crit = log_2(2) = 1.
# All operations up to the recursion calls can be done in constant time: there
#   are no loops involved, neither hidden nor openly visible.
#   (We assume here, than len(A) can be evaluated in constant time, too. This 
#    holds for lists, but needs not to be true in general. As we have seen in 
#    the winter term, len calls the dunder __len__ which can be of arbitrary
#    complexity. For lists, it simply returns the value of an instance attribute
#    of the list, so it _is_ constant in time.)
# This means that f is in O(1) = O(N^0) or c = 0.
# Since 0 < 1  <==> c < c_crit, we are in Case 1 of the Master Theorem, and get
# that our recursive algorithm is in Theta(N).
# 
# We often don' distinguish Theta, Omega and O, as they rarely really matter in
# practice.
# 
# The take away of this consideration is that this algorithm performs no better
# than good old for loops over each element. In fact, it is MUCH worse, for the
# same reasons as stated in problem 3.

# =========================================================================== #
# problem 5

# The trick here is to identify such criteria that kick out most of the 
# employees and evaluate them first.
# Extra consideration should be put into the data types; depending on them it
# can be too expensive to evaluate certain expressions first. As a rule of
# thumb, string operations should be considered last for that reason.
# 
# Filtering for women is cheap (comparing booleans), but eliminates only half
# of the employees. If the IT department is not extremely large, filtering for
# this criterion first will eliminate more employess from the search at the same
# cost as filtering for women first. The join date can be evaluated at the same
# cost and might also be a good choice for the first criterion to check.
# Checking the first letter of the last name on the other hand should be 
# evaluated last, since it involves string parsing which takes a lot of time.
# 
# One could argue that, since there are 26 letters, first filtering for that 
# will bring down the data set to 1/26 which might offset the high cost for 
# string parsing. In principle a fair point, but here we were filtering for the
# rather common first letter A. Initials aren't distributed uniformly, so the
# net effect will be much less than reducing the data by 1/26.
# On the other hand, filtering by initial Q might actually be worthwhile.
# 
# So, the pseudocode might look something like:
# 
# limitJoinDate = getDateFromString("2005-01-01")
# results = []
# for emp in employees :
#     if emp['department'] == DEPT_ID_IT    and
#        emp['join date']   > limitJoinDate and
#        emp['sex']                         and    # we said 'sex' is a boolean. No need to explicitly evaluate equality. If women are represented by False, check for not emp['sex'].
#        emp['last name'].starts('A') :
#        results.append(emp)
#
# Note that due to short circuiting, emp['join date'] is only ever evaluated if
# emp['department'] == DEPT_ID_IT has already been True. Otherwise, Python won't
# even bother to read the join date from memory.
# 
# The same is true for or:
#   if expression1 or expression2
# will NOT evaluate expression2 if expression1 was already True.

# =========================================================================== #
# problem 6

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import time
import numpy as np

# --------------------------------------------------------------------------- #
# this is just for the plot

alpha = .1
omega_x = 5
omega_y = 0.5

xMin = -16
xMax = +16
xRes = 1601
yMin = -16
yMax = +16
yRes = 1601

x = np.linspace(xMin, xMax, xRes)
y = np.linspace(yMin, yMax, xRes)


X, Y = np.meshgrid(x, y)

Z = np.exp(-alpha * (X**2 + Y**2)) * np.abs(np.sin(omega_x * X) + np.sin(omega_y * Y))

fig = plt.figure( figsize=(15,15))
drw = fig.add_subplot(projection='3d')

drw.plot_surface(X, Y, Z)

plt.show()

# --------------------------------------------------------------------------- #
# Parameters to the integration methods

R = 15
N = 100000
evalRadius = (1 / alpha)

# ........................................................................... #
# brute force approach: sum up everything

tic = time.time()
for r in range(R) :
    # to be able to compare both methods, we have to include constructing the
    # grid into account. In general, we cannot assume this to be pre-made
    x = np.linspace(xMin, xMax, xRes)
    y = np.linspace(yMin, yMax, yRes)
    X, Y = np.meshgrid(x, y)
    
    factor = (xMax - xMin) * (yMax - yMin) / X.size   # ... / (xRes * yRes)
    
    # this is the actual integral, if you wish so
    Z = np.exp(-alpha * (X**2 + Y**2)) * np.abs(np.sin(omega_x * X) + np.sin(omega_y * Y))
    integralDirect = np.sum(Z) * factor
    
toc = time.time()
integralDirectTime = (toc - tic) / R

print(f"direct integration: result = {integralDirect:5.2f}, time requirement = {integralDirectTime * 1000:6.2f} ms")

# ........................................................................... #

tic = time.time()
for r in range(R) :
    randX = np.random.uniform( low=xMin, high = xMax, size=N)
    randY = np.random.uniform( low=yMin, high = yMax, size=N)
    toEvaluate = (randX**2 + randY**2) < evalRadius**2
    
    selectX = randX[toEvaluate]
    selectY = randY[toEvaluate]
    
    factor = (xMax - xMin) * (yMax - yMin) / N
    # NOT ... / area(circle), unless you make it so that randX, randY are purely within that circle
    
    Z = np.exp(-alpha * (selectX**2 + selectY**2)) * np.abs(np.sin(omega_x * selectX) + np.sin(omega_y * selectY))
    integralMC = np.sum(Z) * factor
    
toc = time.time()
integralMCTime = (toc - tic) / R

print(f"MC integration    : result = {integralMC:5.2f}, time requirement = {integralMCTime * 1000:6.2f} ms")
print()

print(f"Speedup by choosing MC over direct: {integralDirectTime / integralMCTime:3.2f}")
print(f"relative error MC                 : {100 * np.abs(integralMC - integralDirect) / integralDirect:3.2f}%")
