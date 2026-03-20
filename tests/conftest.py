"""
conftest.py is a configuration file, which pytest automatically incorporates.

It is used to set up fixtures, paths to test data, etc that can then be used by all tests.
"""

import pytest
from pathlib import Path

@pytest.fixture(scope="session")
def test_data():
    yield Path(__file__).resolve().parent / 'test_data'
