"""Eval command handlers."""

import statistics
from pathlib import Path
from tflens_explorer.core.types import CommandContext
from tflens_explorer.services.compare_service import snapshot_create
from tflens_explorer.services.compare_service import snapshots_list
from tflens_explorer.services.compare_service import compare_runs
from tflens_explorer.services.compare_service import compare_mlp
from tflens_explorer.services.compare_service import compare_attention
from tflens_explorer.services.compare_service import compare_generated
from tflens_explorer.services.compare_service import compare_evals
from tflens_explorer.services.compare_service import compare_tokens
from tflens_explorer.services.compare_service import compare_logits
from tflens_explorer.services.compare_service import compare_cache
from tflens_explorer.services.compare_service import compare_models
from tflens_explorer.services.compare_service import compare_snapshots
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
    hook = kwargs.get("hook")

    if snapshot_name == None:
        print("A snapshot name is required. Use: snapshot-create name=<name> hook=<hook name>")
        return

    if not linux_filename_validation(snapshot_name):
        print("Error: The snapshot name must be a valid Linux filename.")
        return

    if hook == None:
        print("A cache layer is required. Use: snapshot-create name=<name> hook=<hook name>")
        print("Find a hook name using the cache-layer command.")
        return

    snapshot_create(context, snapshot_name, hook)

    return

def linux_filename_validation(filename):
    forbidden_chars = ['/', '\0']
    if any(char in filename for char in forbidden_chars):
        return False
    if len(filename) > 255:
        return False
    return True

def handle_snapshots_list(context: CommandContext) -> None:
    snapshots_list()
    return

def handle_compare_models(context: CommandContext) -> None:
    compare_models()
    return

def handle_compare_attention(context: CommandContext) -> None:
    compare_attention()
    return

def handle_compare_mlp(context: CommandContext) -> None:
    compare_mlp()
    return

def handle_compare_generated(context: CommandContext) -> None:
    compare_generated()
    return

def handle_compare_evals(context: CommandContext) -> None:
    compare_evals()
    return

def handle_compare_tokens(context: CommandContext) -> None:
    compare_tokens()
    return

def handle_compare_logits(context: CommandContext) -> None:
    if context.args == None or len(context.args) != 2:
        print("Two snapshot names are required. Use: compare-models <snapshot1> <snapshot2>")
        return  
    
    snapshot1 = context.args[0]
    snapshot2 = context.args[1]

    compare_logits(snapshot1, snapshot2)
    return

def handle_compare_cache(context: CommandContext) -> None:
    if context.args == None or len(context.args) != 2:
        print("Two snapshot names are required. Use: compare-models <snapshot>")
        return  
    
    snapshot1 = context.args[0]
    snapshot2 = context.args[1]
    compare_cache(snapshot1, snapshot2)
    return

def handle_compare_snapshots(context: CommandContext) -> None:
    if context.args == None or len(context.args) != 2:
        print("Two snapshot names are required. Use: compare-models <snapshot>")
        return  
    
    snapshot1 = context.args[0]
    snapshot2 = context.args[1]
    compare_snapshots(snapshot1, snapshot2)
    return
