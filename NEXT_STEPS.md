You are comparing the entire selected tensor as one long vector. For a head slice like:

[seq_len, d_head]

you are measuring the direction change of that head across all token positions and all head dimensions together.

So your angle answers:

"How much did this whole activation block rotate in vector space?"

It does not answer:

"Which token position changed most?"
"Which feature dimension changed most?"
"Was the change concentrated or spread out?"
