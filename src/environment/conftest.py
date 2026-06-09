def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "slow: integration tests that run real FDFD (deselect with -m 'not slow')",
    )
