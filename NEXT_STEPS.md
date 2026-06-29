# Next Steps

## Reporting changes

Refactor compare_service calls to return data instead of printing data.


## Comparison Changes
You are comparing the entire selected tensor as one long vector. For a head slice like:

[seq_len, d_head]

you are measuring the direction change of that head across all token positions and all head dimensions together.

So your angle answers:

"How much did this whole activation block rotate in vector space?"

It does not answer:

"Which token position changed most?"
"Which feature dimension changed most?"
"Was the change concentrated or spread out?"

## Comparing Models with Different Architectures

For your current table, I would suppress these fields when dimensions differ:

mean_abs_diff: n/a
cos_sim: n/a

And replace them with something like:

norm_A
norm_B
std_A
std_B
max_abs_A
max_abs_B

The one useful next refactor: add a compare_mode distinction:

same_architecture_compare
cross_model_behavior_compare
cross_model_summary_compare