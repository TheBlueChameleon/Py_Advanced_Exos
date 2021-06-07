# https://math.stackexchange.com/questions/65923/how-does-one-compute-the-sign-of-a-permutation

import numpy as np
import unittest

# ============================================================================ #

def allPermutations (N : int) -> list :
    """
    Computes the symmetric group of size N together with the signa of the
    permutations within.

    PARAMETERS
        N : int, >= 0
            Number of elements per permutation

    RETURNS
        list of tuples: permutations and their signum
        format is [([indices1], signum1), ([indices2], signum2), ...]
        where indices_ are all the indices that together make up a permutation
        indices_ are between 0 and N-1

    RAISES
        * TypeError if N is not an int
        * ValueError if N < 0

    Example:
    >>> allPermutations(2)
    [([0, 1], 1), ([1, 0], -1)]

    >>> allPermutations(0)
    []

    """

    if type(N) != int : raise TypeError("Parameter N has to be of type int!")
    if N  < 0         : raise ValueError("Parameter N must be nonnegative!")
    if N == 0         : return []


    def subPerm(options) :
        # trivial case: list of length 1 has only one permutation
        if len(options) == 1 : return [(options, 1)]

        # nontrivial cases...
        results = []            # prepare results one by one in here...

        # treat first element specially: copy signum
        subresults = subPerm(options[1:])
        for perm, sgn in subresults :
            results.append( ([options[0]] + perm, sgn) )

        # all subsequent elements: flipped signum
        optionsIter = iter(range(len(options)))         # index in options: which element goes first?
        next(optionsIter)                               # we just did startElementIndex = 0

        for startElementIndex in optionsIter :
            subresults = subPerm(options[:startElementIndex] + options[startElementIndex + 1:])
            for perm, sgn in subresults :
                results.append( ([options[startElementIndex]] + perm, -sgn) )


        return results

    result = subPerm(list(range(N)))
    return result

# ---------------------------------------------------------------------------- #

def permanent (M) :
    if type(M) != np.ndarray    : raise TypeError("only compatible with numpy ndarrays")
    if len(M.shape) != 2        : raise ValueError("not a matrix")
    if M.shape[0] != M.shape[1] : raise ValueError("not a square matrix")

    N = M.shape[0]
    result = 0

    for permutation, signum in allPermutations(N) :
        factor = 1
        for rowID in range(N) :
            factor *= M[rowID, permutation[rowID]]
        result += factor

    return result

# ---------------------------------------------------------------------------- #

def determinant (M) :
    if type(M) != np.ndarray    : raise TypeError("only compatible with numpy ndarrays")
    if len(M.shape) != 2        : raise ValueError("not a matrix")
    if M.shape[0] != M.shape[1] : raise ValueError("not a square matrix")

    N = M.shape[0]
    result = 0

    for permutation, signum in allPermutations(N) :
        factor = 1
        for rowID in range(N) :
            factor *= M[rowID, permutation[rowID]]
        result += signum * factor

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
        Tested are type errors, value errors, and the cases 0, 1, 2, 3
        """

        # Albeit unittest provides some nice formating, it helps with
        # development to have some separator lines on screen -- hence the prints
        # in this function.
        # Also, personally I find it nice to see on screen what my code does
        # without having to read the entire code.

        print("#" * 40)
        print(" entering testSuite allPermutations ".center(40, "#"))
        print()

        print(" Typechecks ... ".center(40, "~"))
        with self.assertRaises(TypeError) :
            allPermutations("any string")
            allPermutations(1.0)
        print("okay")

        with self.assertRaises(ValueError) :
            allPermutations(-1)

        print(" Input value 0 ... ".center(40, "~"))
        self.assertEqual(allPermutations(0), [])
        print("okay")


        print(" Input value 1 ... ".center(40, "~"))
        self.assertEqual(allPermutations(1),
                         [([0], 1)]
                        )
        print("okay")

        print(" Input value 2 ... ".center(40, "~"))
        self.assertEqual(  allPermutations(2), [([0, 1], 1), ([1, 0], -1)]  )
        print("okay")

        print(" Input value 3 ... ".center(40, "~"))
        self.assertEqual(
            allPermutations(3),
            [([0, 1, 2],  1),
             ([0, 2, 1], -1),
             ([1, 0, 2], -1),
             ([1, 2, 0],  1),
             ([2, 0, 1],  1),
             ([2, 1, 0], -1)]
        )
        print("okay")

        print()
        print(" passed testSuite allPermutations ".center(40, "#"))
        print("#" * 40)

    # ........................................................................ #

    def test_permanent(self) :
        print()
        print("#" * 40)
        print(" entering testSuite permanent ".center(40, "#"))
        print()

        print(" Typechecks ... ".center(40, "~"))
        for inp, err in self.invalidInputs :
            print(inp, end = " ... ", flush=True)
            with self.assertRaises(err) :
                permanent(inp)
            print("check")
        print("okay")

        for mat, det, perm in self.detsAndPerms :
            print(" Integer matrix ".center(40, "~"))
            print(mat, end = " ... ", flush=True)
            self.assertEqual(permanent(mat), perm)
            print("check")
        print("okay")

        print()
        print(" passed testSuite permanent ".center(40, "#"))
        print("#" * 40)

    # ........................................................................ #

    def test_determinant(self) :
        print()
        print("#" * 40)
        print(" entering testSuite determinant ".center(40, "#"))
        print()

        print(" Typechecks ... ".center(40, "~"))
        for inp, err in self.invalidInputs :
            print(inp, end = " ... ", flush=True)
            with self.assertRaises(err) :
                determinant(inp)
            print("check")
        print("okay")

        for mat, det, perm in self.detsAndPerms :
            print(" Integer matrix ".center(40, "~"))
            print(mat, end = " ... ", flush=True)
            self.assertEqual(determinant(mat), det)
            print("check")
        print("okay")

        print()
        print(" passed testSuite determinant ".center(40, "#"))
        print("#" * 40)

# ============================================================================ #

if __name__ == '__main__' :
    unittest.main()
