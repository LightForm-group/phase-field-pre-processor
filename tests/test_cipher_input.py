from textwrap import dedent
import pytest

import numpy as np
from cipher_input import compress_1D_array_string, decompress_1D_array_string


def test_expected_compresss_1D_array():
    """Test compression as expected."""
    arr = np.array([1, 1, 1, 1, 2, 2, 1, 2, 3, 1, 3, 3, 2, 2, 2, 1, 1, 4])
    arr_str = compress_1D_array_string(arr)
    expected = dedent(
        """
        4 of 1
        2 of 2
        1
        2
        3
        1
        2 of 3
        3 of 2
        2 of 1
        4
        """
    ).strip()
    assert arr_str == expected


def test_round_trip_1D_array():
    """Test round-trip compress/decompress of 1D array."""
    arr = np.random.choice(3, size=100)  # likely to get consecutive repeats
    arr_str = compress_1D_array_string(arr)
    arr_reload = decompress_1D_array_string(arr_str)
    assert np.all(arr_reload == arr)
