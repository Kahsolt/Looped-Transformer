# Looped Transformers

    Unofficial implementation of the "Looped Transformers as Programmable Computers" 

----

Unofficial implementation of the **Looped Transformers as Programmable Computers**: [https://arxiv.org/abs/2301.13196](https://arxiv.org/abs/2301.13196)


Possibly cannot implement (at essay page 11):

- the `FFW` weights for `read` operation is not specified, and I failed to infer & construct by myself
- the `Attn(x)` output contains posenc row `P0,P1` value in set `{+1,-1}`, which will be ruined after `ReLU` activation

----
by Armit
2023/11/13
