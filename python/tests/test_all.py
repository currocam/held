import pytest
import held


def test_sum_as_string():
    assert held.sum_as_string(1, 1) == "2"
