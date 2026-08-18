##Generation, conceptually:

- [ ] Build a prompt (user question + enough catalog so the model can pick a function).
- [ ] Encode it.
- [ ] Ask for logits.
- [ ] Mask token IDs that would be illegal right now (JSON broken, or type/schema broken).
- [ ] Pick among the rest (usually the highest remaining logit).
- [ ] Append that ID; repeat until the JSON object is finished.