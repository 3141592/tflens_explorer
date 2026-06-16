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
- `head inspect <hook> <head>`

## Shortcut commands

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