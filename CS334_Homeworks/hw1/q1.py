"""
THIS CODE IS MY OWN WORK, IT WAS WRITTEN WITHOUT CONSULTING CODE
WRITTEN BY OTHER STUDENTS OR LARGE LANGUAGE MODELS SUCH AS CHATGPT.
Tommy Skodje

I collaborated with the following classmates for this homework:
None
"""


import numpy as np
import numpy.testing as npt
import timeit


def gen_random_samples(n):
    """
    Generate n random samples using the
    numpy random.randn module.

    Returns
    ----------

    sample : 1d array of size n
        An array of n random samples
    """

    sample = np.random.randn(n)

    return sample


def sum_squares_for(samples):
    """
    Compute the sum of squares using a forloop

    Parameters
    ----------
    samples : 1d-array with shape n
        An array of numbers.

    Returns
    -------
    ss : float
        The sum of squares of the samples
    """
    ss = 0
    # Compute the sum of squares
    for j in range(samples.shape[0]):
        ss = ss + (samples[j] * samples[j])


    return ss


def sum_squares_np(samples):
    """
    Compute the sum of squares using Numpy's dot module

    Parameters
    ----------
    samples : 1d-array with shape n
        An array of numbers.

    Returns
    -------
    ss : float
        The sum of squares of the samples
    """
    ss = 0
    # Compute the sum of squares using numpy
    ss = np.dot(samples, samples)
    return ss


def main():
    # generate 5 million random samples
    samples = gen_random_samples(5000000)

    # call the for version
    start = timeit.default_timer()
    ss_for = sum_squares_for(samples)
    time_for = timeit.default_timer() - start
    # call the numpy version
    start = timeit.default_timer()
    ss_np = sum_squares_np(samples)
    time_np = timeit.default_timer() - start

    # make sure they're the same value
    npt.assert_almost_equal(ss_for, ss_np, decimal=5)
    # print out the values
    print("Time [sec] (for loop):", time_for)
    print("Time [sec] (np loop):", time_np)


if __name__ == "__main__":
    main()
