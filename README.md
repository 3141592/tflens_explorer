# tflens-explorer

A command-driven CLI skeleton for exploring TransformerLens models.

## Development

```bash
pip install -e .[dev]
tflens-explorer
```

## Command Reference

All commands are entered at the `>` prompt in the interactive shell.

### Core Commands

| Command | Aliases | Description | Usage |
|---------|---------|-------------|-------|
| `help` | `?` | Show available commands. | `help` |
| `commands` | `ls`, `ll` | List registered commands. Optional filter. | `commands [filter]` |
| `quit` | `exit`, `q` | Exit the CLI. | `quit` |
| `clear` | `c` | Clear the terminal. | `clear` |
| `session-clear` | — | Clear the current session state (model, cache, prompt, etc.). | `session-clear` |
| `context-show` | — | Show the contents of the current session context (prompt, cache present, etc.). | `context-show` |

### Model Commands

| Command | Aliases | Description | Usage |
|---------|---------|-------------|-------|
| `model-list` | — | List all models available via TransformerLens. Optional filter substring. | `model-list [filter]` |
| `model-aliases` | `model-alias` | List configured local model aliases (defined in `config/models.yaml`). | `model-aliases` |
| `model-cache` | — | List models currently stored in the local Hugging Face cache. | `model-cache` |
| `model-load` | — | Load a model into the current session. Accepts an alias or a full model name. | `model-load <model_name>` |
| `model-load-quantized` | — | Load a quantized model into the current session. | `model-load-quantized <model_name>` |
| `model-info` | — | Show information about the currently loaded model (parameters, layers, etc.). | `model-info` |
| `model-show` | — | Show the architecture of the currently loaded model. | `model-show` |

### Prompt Commands

| Command | Aliases | Description | Usage |
|---------|---------|-------------|-------|
| `prompt-set` | — | Store the prompt text for the current session. | `prompt-set <text>` |
| `prompt-show` | — | Display the currently stored prompt text. | `prompt-show` |
| `prompt-run` | — | Run inference on the current prompt. Accepts `new_tokens=<N>` to control generation length (default: 10). | `prompt-run [new_tokens=<N>]` |

### Token Commands

| Command | Aliases | Description | Usage |
|---------|---------|-------------|-------|
| `tokens` | — | Show the tokenized representation of the current prompt (token IDs and decoded strings). | `tokens` |
| `token-decode` | — | Decode a single token ID to its string representation. | `token-decode <token_id>` |
| `token-encode` | — | Encode a text string to its token ID(s). | `token-encode <text>` |
| `token-next` | — | Add the next predicted token to the current prompt. | `token-next` |
| `embedding-similarity` | `embedding-cos`, `embedding-sim`, `cos-sim` | Calculate the cosine similarity between the embeddings of two words. | `embedding-similarity <word1> <word2>` |
| `logits` | — | Show the final position logits (top predictions) for the current prompt. | `logits` |
| `logits-for` | — | Show the final position logits for a specific text string or token ID. | `logits-for "<text>"` or `logits-for <integer>` |

### Cache Commands

| Command | Aliases | Description | Usage |
|---------|---------|-------------|-------|
| `cache-run` | — | Run the current prompt through the model with caching enabled and store the cache in the session. | `cache-run` |
| `cache-show` | — | Show summary information about the cached activations (prompt, BOS setting, number of keys). | `cache-show` |
| `cache-keys` | — | List all cached activation keys. Optional filter substring. | `cache-keys [filter]` |
| `cache-layer` | `cache-block` | Show available tensor names for a specific layer. | `cache-layer <layer_number>` |
| `cache-tensor` | — | Show detailed information (shape, dtype, etc.) about a specific cached tensor by its full layer name. | `cache-tensor <layer_name>` |

### Eval Commands

| Command | Aliases | Description | Usage |
|---------|---------|-------------|-------|
| `eval-run` | — | Run all model evaluations defined in `config/evals.yaml` and print per-eval results plus a summary. | `eval-run` |
| `eval-summary` | — | Run all model evaluations but print only the summary statistics (top-1 accuracy, top-5 accuracy, rank stats). | `eval-summary` |

### Snapshot Commands

| Command | Aliases | Description | Usage |
|---------|---------|-------------|-------|
| `snapshot-create` | `sc` | Create a snapshot YAML file capturing activations and logits for comparison. Requires `name` and `hook` parameters. Use `hook=all` to capture all layers. | `snapshot-create name=<name> hook=<hookname>\|all` |
| `snapshots-list` | — | List all available snapshot YAML files in the `snapshots/` directory. | `snapshots-list` |

### Compare Commands

| Command | Aliases | Description | Usage |
|---------|---------|-------------|-------|
| `compare-mlp` | — | Compare MLP activations between two models. | `compare-mlp` |
| `compare-attention` | — | Compare attention values between two models. | `compare-attention` |
| `compare-generated` | — | Compare generated text between two models. | `compare-generated` |
| `compare-evals` | — | Compare evaluation results between two models. | `compare-evals` |
| `compare-tokens` | — | Compare token outputs between two models. | `compare-tokens` |
| `compare-logits` | — | Compare logits of two snapshots. Requires two snapshot names. | `compare-logits <snapshot1> <snapshot2>` |
| `compare-cache` | — | Compare cached activations of two snapshots. Requires two snapshot names. | `compare-cache <snapshot1> <snapshot2>` |
| `compare-models` | — | Compare two models. | `compare-models` |
| `compare-snapshots` | `cs` | Compare two snapshots with optional filtering. `diff=<N>` shows only differences beyond a threshold. `percent=<N>` filters by top-N%. | `compare-snapshots <snapshot1> <snapshot2> [diff=<N>] [percent=<N>]` |

## Example Session

```bash
model-load gpt2
prompt-set The cat sat on the
tokens
logits
prompt-run new_tokens=25
cache-run
cache-show
cache-keys
cache-layer 0
snapshot-create name=cat_sat_gpt2 hook=blocks.0.mlp.hook_post
snapshot-create name=dog_sat_gpt2 hook=blocks.0.mlp.hook_post
compare-logits cat_sat_gpt2 dog_sat_gpt2
compare-snapshots cat_sat_gpt2 dog_sat_gpt2 diff=1
```

## Configuration

- **`config/commands.yaml`** — Defines the registered commands, aliases, descriptions, and handler mappings.
- **`config/models.yaml`** — Configures local model name aliases for shorthand usage.
- **`config/evals.yaml`** — Defines evaluation prompts and expected tokens for model evaluation.
- **`config/internals.yaml`** — Internal configuration options.

## Project Conventions

- YAML stores metadata and summaries only; raw tensors belong in separate `.pt` files.
- Snapshot creation and comparison logic are separated.
- Avoid dumping large tensor lists into YAML.

## Quick Commands
snapshot-create name=cat_sat_gpt2 hook=blocks.0.mlp.hook_post
snapshot-create name=dog_sat_gpt2 hook=blocks.0.mlp.hook_post
snapshot-create name=test hook=all
snapshot-create name=cat_sat_gpt2_all hook=all
snapshot-create name=dog_sat_gpt2_all hook=all

compare-logits cat_sat_gpt2 dog_sat_gpt2
compare-logits cat_sat_gpt2 dog_sat_gpt2
compare-logits shepherd_chat2 shepherd_deepseek
compare-logits shepherd_chat2 shepherd_gpt2
compare-logits shepherd_gpt2 shepherd_deepseek
compare-logits cat_sat_gpt2_all dog_sat_gpt2_all
compare-logits cat_sat_gpt2_all cat_sat_deepseek_all

compare-cache cat_sat_gpt2 dog_sat_gpt2
compare-cache cat_sat_gpt2 dog_sat_gpt2
compare-cache shepherd_chat2 shepherd_deepseek
compare-cache shepherd_chat2 shepherd_gpt2
compare-cache shepherd_gpt2 shepherd_deepseek
compare-cache cat_sat_gpt2_all dog_sat_gpt2_all
compare-cache cat_sat_gpt2_all cat_sat_deepseek_all
compare-cache cat_sat_mistral1_all dog_sat_mistral1_all

compare-snapshots cat_sat_gpt2 dog_sat_gpt2
compare-snapshots shepherd_chat2 shepherd_deepseek
compare-snapshots shepherd_chat2_all shepherd_deepseek_all
compare-snapshots shepherd_chat2 shepherd_gpt2
compare-snapshots shepherd_gpt2 shepherd_deepseek
compare-snapshots cat_sat_gpt2_all dog_sat_gpt2_all
compare-snapshots cat_sat_gpt2_all pig_sat_gpt2_all
compare-snapshots cat_sat_gpt2_all cat_sat_deepseek_all
compare-snapshots cat_sat_mistral1_all dog_sat_mistral1_all
compare-snapshots cat_sat_gpt2_all dog_sat_gpt2_all diff=1
compare-snapshots cat_sat_gpt2_all dog_sat_gpt2_all percent=50

prompt-run new_tokens=50