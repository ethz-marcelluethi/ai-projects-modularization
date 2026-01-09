import pytest
from src import download_penguins_data


@pytest.fixture(scope="session", autouse=True)
def setup_test_data():
    """
    Download penguins data before running tests.
    This 'fixture' is run automatically before the test session starts
    """
    download_penguins_data()
    yield
