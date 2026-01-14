import os
import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "openai: tests requiring OPENAI_API_KEY")
    config.addinivalue_line("markers", "anthropic: tests requiring ANTHROPIC_API_KEY")


@pytest.fixture
def openai_api_key():
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        pytest.skip("OPENAI_API_KEY not set")
    return key


@pytest.fixture
def anthropic_api_key():
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        pytest.skip("ANTHROPIC_API_KEY not set")
    return key
