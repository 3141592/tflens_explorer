"""Eval command handlers."""

import statistics
from pathlib import Path
from tflens_explorer.core.types import CommandContext
from tflens_explorer.services.compare_service import snapshot_create
from tflens_explorer.services.compare_service import Snapshot
from tflens_explorer.cli.utilities import parse_kv_args

def handle_snapshot_create(context: CommandContext) -> None:
    model = context.session.model
    if model is None:
        print("No model loaded.")
        return

    prompt = context.session.current_prompt
    if not prompt:
        print("No prompt set. Use: prompt-set <text>")
        return

    kwargs = parse_kv_args(context.args)
    snapshot_name = kwargs.get("name")

    if snapshot_name == None:
        print("A snapshot name is required. Use: run-create name=<name>")
        return

    if not linux_filename_validation(snapshot_name):
        print("Error: The snapshot name must be a valid Linux filename.")
        return

    snapshot_create(context, snapshot_name)

    return

def linux_filename_validation(filename):
    forbidden_chars = ['/', '\0']
    if any(char in filename for char in forbidden_chars):
        return False
    if len(filename) > 255:
        return False
    return True
