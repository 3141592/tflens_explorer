# tflens-explorer

A command-driven CLI skeleton for exploring TransformerLens models.

## Development

```bash
pip install -e .[dev]
tflens-explorer
```

## Current built-in commands

- `help`
- `commands`
- `quit`

## Next commands to add

- `model load <name>`
- `model info`
- `prompt set <text>`
- `prompt run`
- `cache list`
- `head inspect <layer> <head>`

## Shortcut compares

compare-logits cat_sat_gpt2 dog_sat_gpt2
compare-logits cat_sat_gpt2 dog_sat_gpt2
compare-logits shepherd_chat2 shepherd_deepseek
compare-logits shepherd_chat2 shepherd_gpt2
compare-logits shepherd_gpt2 shepherd_deepseek

compare-snapshots cat_sat_gpt2 dog_sat_gpt2
compare-snapshots cat_sat_gpt2 dog_sat_gpt2
compare-snapshots shepherd_chat2 shepherd_deepseek
compare-snapshots shepherd_chat2 shepherd_gpt2
compare-snapshots shepherd_gpt2 shepherd_deepseek