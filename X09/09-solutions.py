import numpy as np
import unittest
import itertools

# ============================================================================ #

def allPermutations (N : int) -> list :
    """
    Computes the symmetric group of size N together with the signa of the
    permutations within.

    PARAMETERS
        N : int, >= 0
            Number of elements per permutation

    RETURNS
        list of lists: all permutations of [0, 1, ..., N-1]
        format is [[indices1], [indices2], ...]
        where indices_ are all the indices that together make up a permutation
        indices_ are between 0 and N-1

    RAISES
        * TypeError if N is not an int
        * ValueError if N < 0

    Examples:
    >>> allPermutations(2)
    [[0, 1], [1, 0]]

    >>> allPermutations(0)
    []

    """

    if type(N) != int : raise TypeError("Parameter N has to be of type int!")
    if N  < 0         : raise ValueError("Parameter N must be nonnegative!")
    if N == 0         : return []


    def subPerm(options) :
        # trivial case: list of length 1 has only one permutation
        if len(options) == 1 : return [options]

        # nontrivial cases...
        results = []            # prepare results one by one in here...

        for startElementIndex in range(len(options)) :                          # index in options: which element goes first?
            subresults = subPerm(options[:startElementIndex] + options[startElementIndex + 1:])
            for perm in subresults :
                results.append( [options[startElementIndex]] + perm )

        return results

    result = subPerm(list(range(N)))
    return result

# ---------------------------------------------------------------------------- #

def permanent (M : np.ndarray) :
    """
    computes the permanent of a matrix M

    PARAMETERS
        M : numpy.ndarray
            square matrix

    RETURNS
        permanent of the matrix

    RAISES
        * TypeError if M is not an np.ndarray
        * ValueError if dim(M) != 2
        * ValueError if M is not a square matrix

    Examples:
    >>> permanent([[1]])
    1

    >>> permanent([[1,1], [1,1]])
    2
    """

    if type(M) != np.ndarray    : raise TypeError("only compatible with numpy ndarrays")
    if len(M.shape) != 2        : raise ValueError("not a matrix")
    if M.shape[0] != M.shape[1] : raise ValueError("not a square matrix")

    N = M.shape[0]
    result = 0

    for permutation in allPermutations(N) :
        factor = 1
        for rowID in range(N) :
            factor *= M[rowID, permutation[rowID]]
        result += factor

    return result

# ---------------------------------------------------------------------------- #

def signumOfPerm(P : list) -> int :
    """
    Computes the signum of a permutation

    PARAMETERS
        P: list of integers [0, 1, ..., N-1]

    RETURNS
        signum of P, as integer

    RAISES
        * TypeError if P is not a list
        * ValueError if P is empty
        * ValueError if P is not a permutation of [0, 1, ..., N-1]

    Examples:
    >>> signumOfPerm([0])
    1

    >>> signumOfPerm([0, 2, 1])
    -1
    """

    # to Python from this source:
    # https://math.stackexchange.com/questions/65923/how-does-one-compute-the-sign-of-a-permutation

    if type(P) != list : raise TypeError("not a list")
    N = len(P)
    if N == 0          : raise ValueError("empty permutation")

    foundAll = True
    for i in range(N) : foundAll &= i in P
    if not foundAll    : raise ValueError("not a permutation of [0, 1, ..., N-1]!")

    visited = [False for i in range(N)]
    sgn = 1
    for k in range(N) :
        L = 1                                                                   # this was not in the code on stackexchange... I guess the forgot about it...
        if not visited[k] :
            nxt = k
            L = 0
            while not visited[nxt] :
                L += 1
                visited[nxt] = True
                nxt = P[nxt]

        if L % 2 == 0 : sgn = -sgn

    return sgn

# ............................................................................ #
def determinant (M) :
    """
    computes the determinant of a matrix M

    PARAMETERS
        M : numpy.ndarray
            square matrix

    RETURNS
        determinant of the matrix

    RAISES
        * TypeError if M is not an np.ndarray
        * ValueError if dim(M) != 2
        * ValueError if M is not a square matrix

    Examples:
    >>> determinant([[1]])
    1

    >>> determinant([[1,1], [1,1]])
    0
    """
    if type(M) != np.ndarray    : raise TypeError("only compatible with numpy ndarrays")
    if len(M.shape) != 2        : raise ValueError("not a matrix")
    if M.shape[0] != M.shape[1] : raise ValueError("not a square matrix")

    N = M.shape[0]
    result = 0

    for permutation in allPermutations(N) :
        factor = 1
        for rowID in range(N) :
            factor *= M[rowID, permutation[rowID]]
        result += signumOfPerm(permutation) * factor

    return result

# ============================================================================ #

class TestSuite_det_perm (unittest.TestCase) :
    def setUp(self) :
        """
        Provides standard matrices against which det and perm are tested
        """

        # These matrices are computed anew for each test.
        # Strictly speaking, this is a bit wasteful. You might want to put this
        # in __init__ as the matrices have to be computed only once.
        # I only wanted to demonstrate that this feature exists in unittest
        # and how to use it.

        self.detsAndPerms = [                                                   # format: [(matrix, det, perm), ...]
            (np.array([[1]]), 1, 1),
            (np.array([[1, 2], [2,4]]),  0,  8),
            (np.array([[1, 2], [3,4]]), -2, 10),
            (1 + np.array( range(9) ).reshape((3,3)), 0, 450)                   # the telephone matrix
                                                                                # values certified by Wolfram Alpha:
                                                                                # https://reference.wolfram.com/language/ref/Permanent.html
            # We could include matrices of type float or complex, but since there is nothing that uses int-specifics, we may spare ourselves the hassle.
            # (there is only standard arithmetics -- addition and multiplication -- in the determinant and permanent code)
            # When in a pinch, when you aren't sure whether or not a type can be excluded from a test: just put it in there.
        ]

        self.invalidInputs = [                                                  # format [(input, resultingError), ...]
            ("any string", TypeError),
            (0, TypeError),
            (np.array([]), ValueError),
            (np.array([1]), ValueError),
            (np.array([[1, 2]]), ValueError),
            (np.array([[1, 2, 3],
                       [4, 5, 6]]), ValueError),
        ]

    # ........................................................................ #

    def test_allPermutations (self) :
        """
        Tests the function allPermutations.
        Tested are type errors, value errors, and the cases 0, 1, 2, 3 and 4
        """

        # Albeit unittest provides some nice formating, it helps with
        # development to have some separator lines on screen -- hence the prints
        # in this function.
        # Also, personally I find it nice to see on screen what my code does
        # without having to read the entire code.

        print("#" * 80)
        print(" entering testSuite allPermutations ".center(80, "#"))
        print()

        print(" Typechecks ... ".center(80, "~"))
        with self.assertRaises(TypeError) :
            allPermutations("any string")
            allPermutations(1.0)
        print("okay")

        with self.assertRaises(ValueError) :
            allPermutations(-1)


        print(" Valid inputs ... ".center(80, "~"))

        print("0", end=" ... ", flush=True)
        self.assertEqual(allPermutations(0), [])
        print("check")


        print("1", end=" ... ", flush=True)
        self.assertEqual(allPermutations(1),
                         [[0]]
                        )
        print("check")

        print("2", end=" ... ", flush=True)
        self.assertEqual(allPermutations(2),
                         [[0, 1], [1, 0]]
                        )
        print("check")

        print("0", end=" ... ", flush=True)
        self.assertEqual(
            allPermutations(3),
            [[0, 1, 2],
             [0, 2, 1],
             [1, 0, 2],
             [1, 2, 0],
             [2, 0, 1],
             [2, 1, 0]]
        )
        print("check")

        print("0", end=" ... ", flush=True)
        iterPerms = list(itertools.permutations( range(4) ))                    # normally we don't have a function to check against, but for the sake of it...
        for i, perm in enumerate(iterPerms) :                                   # however, itertools returls a list of tuples.
            iterPerms[i] = list(perm)                                           # we want a list of lists.
        self.assertEqual(allPermutations(4), iterPerms)
        print("check")
        print("okay")

        # in fact, it's not *that* uncommon to have a second function to test
        # your code against.
        # often, what we want is either a more efficient version of an existing
        # solution, or we want the generic solution to a problem for which we
        # have at least a specific solution.

        print()
        print(" passed testSuite allPermutations ".center(80, "#"))
        print("#" * 80)

    # ........................................................................ #

    def test_permanent(self) :
        """
        Tests the function permanent.
        uses the detsAndPerms and invalidInputs attributes for test matrices
        """

        print()
        print("#" * 80)
        print(" entering testSuite permanent ".center(80, "#"))
        print()

        print(" Typechecks ... ".center(80, "~"))
        for inp, err in self.invalidInputs :                                    # isn't it nice how we can define a re-usable set of test matrices ...
            print(inp, end = " ... ", flush=True)
            with self.assertRaises(err) :                                       # ... and how the code reads almost like normal English?
                permanent(inp)
            print("check")
        print("okay")

        print(" Integer matrices ".center(80, "~"))
        for mat, det, perm in self.detsAndPerms :
            print(mat, end = " ... ", flush=True)
            self.assertEqual(permanent(mat), perm)                              # in particular here!
            print("check")
        print("okay")

        print()
        print(" passed testSuite permanent ".center(80, "#"))
        print("#" * 80)

    # ........................................................................ #

    def test_signumOfPerm(self) :
        """
        Tests the function signumOfPerm.
        uses the S_3 and some invalid inputs
        """

        print()
        print("#" * 80)
        print(" entering testSuite signumOfPerm ".center(80, "#"))
        print()

        errTests = [("anything not a list", TypeError),
                    ([0,1,3], ValueError),                                      # not a permutation of [0, 1, .. N-1]
                    ([], ValueError)
        ]
        validTests = [([0, 1, 2],  1),
                      ([0, 2, 1], -1),
                      ([1, 0, 2], -1),
                      ([1, 2, 0],  1),
                      ([2, 0, 1],  1),
                      ([2, 1, 0], -1)]

        print(" Typechecks ... ".center(80, "~"))
        for inp, err in errTests :
            print(inp, end = " ... ", flush=True)
            with self.assertRaises(err) :
                signumOfPerm(inp)
            print("check")
        print("okay")

        print(" Valid inputs ... ".center(80, "~"))
        for perm, sig in validTests :
            print(perm, end = " ... ", flush=True)
            self.assertEqual(signumOfPerm(perm), sig)
            print("check")
        print("okay")

        print()
        print(" passed testSuite signumOfPerm ".center(80, "#"))
        print("#" * 80)

    # ........................................................................ #

    def test_determinant(self) :
        """
        Tests the function determinant.
        uses the detsAndPerms and invalidInputs attributes for test matrices
        """

        print()
        print("#" * 80)
        print(" entering testSuite determinant ".center(80, "#"))
        print()

        print(" Typechecks ... ".center(80, "~"))
        for inp, err in self.invalidInputs :
            print(inp, end = " ... ", flush=True)
            with self.assertRaises(err) :
                determinant(inp)
            print("check")
        print("okay")

        print(" Integer matrices ".center(80, "~"))
        for mat, det, perm in self.detsAndPerms :
            print(mat, end = " ... ", flush=True)
            self.assertEqual(determinant(mat), det)
            print("check")
        print("okay")

        print()
        print(" passed testSuite determinant ".center(80, "#"))
        print("#" * 80)

# ============================================================================ #

if __name__ == '__main__' :
    unittest.main()
