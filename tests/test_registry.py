from tflens_explorer.cli.command_registry import build_registry


def test_registry_loads_default_commands():
    registry = build_registry()
    names = [command.name for command in registry.unique_commands()]
    assert "help" in names
    assert "commands" in names
    assert "quit" in names

def test_command_aliases_are_registered():
    registry = build_registry()

    for command in registry.unique_commands():
        for alias in command.aliases:
            resolved = registry.get(alias)
            assert resolved.name == command.name