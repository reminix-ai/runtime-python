"""Placeholder test to verify test setup works."""


def test_setup_works():
    """Verify pytest is configured correctly."""
    assert True


def test_can_import_package():
    """Verify we can import the package."""
    from reminix_runtime import AgentAdapter, serve

    assert serve is not None
    assert AgentAdapter is not None
