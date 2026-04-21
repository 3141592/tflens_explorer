from tflens_explorer.cli.command_registry import build_registry


def test_registry_loads_default_commands():
    registry = build_registry()
    names = [command.name for command in registry.unique_commands()]
    assert "help" in names
    assert "commands" in names
    assert "quit" in names
