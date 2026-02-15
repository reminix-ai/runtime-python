"""Placeholder test to verify test setup works."""


def test_setup_works():
    """Verify pytest is configured correctly."""
    assert True


def test_can_import_package():
    """Verify we can import the package."""
    from reminix_runtime import agent, serve, tool

    assert serve is not None
    assert agent is not None
    assert tool is not None
