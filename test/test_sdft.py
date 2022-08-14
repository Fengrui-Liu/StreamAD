import numpy as np
from streamad.util.math_toolkit import SDFT


def test_sdft():
    X = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    window_size = 5
    sdft = SDFT(window_size)
    for i, x in enumerate(X):
        sdft = sdft.update(x)
        if i + 1 >= window_size:
            print("co:", sdft.coefficients)
            print("----")
            print("np:", np.fft.fft(X[i + 1 - window_size : i + 1]))
            print("----------------------")
            # assert np.allclose(
            #     sdft.coefficients, np.fft.fft(X[i + 1 - window_size : i + 1])
            # )


def test_dft_time():
    import time

    X = np.random.randn(1000000)

    sdft = SDFT(10)
    start_time = time.time()
    for x in X:
        sdft = sdft.update(x)
    print("sdft", time.time() - start_time)

    start_time = time.time()
    for i in range(len(X) - 10):
        np.fft.fft(X[i : i + 10])

    print("np", time.time() - start_time)
