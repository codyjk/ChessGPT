# `trained_models/len_25_embd_128_layers_2_heads_2`

This model was trained with the following hyperparameters:
* Context length: 25
* Embeddings: 128
* Layers: 2
* Heads: 2

The model was trained on 5,000,000 "master" level games using `poetry run train-model` on an RTX 3090 across 5 epochs, taking about 7 hours total.

To play with it, use the command below from the root of the project:

```sh
poetry run play --input-model-file trained_models/len_25_embd_128_layers_2_heads_2/model.pth --input-tokenizer-file trained_models/len_25_embd_128_layers_2_heads_2/tokenizer.json --max-context-length 25 --num-embeddings 128 --num-layers 2 --num-heads 2
```
