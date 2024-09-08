# `trained_models/len_25_embd_128_layers_2_heads_2`

## Hyperparameters

Training details:
* Context length: 25
* Embeddings: 128
* Layers: 2
* Heads: 2
* Training data: 1,000,000 "master" level games
* Epochs: 5
* Final validation loss: 1.71

```sh
poetry run play --input-model-file trained_models/len_25_embd_128_layers_2_heads_2/model.pth --input-tokenizer-file trained_models/len_25_embd_128_layers_2_heads_2/tokenizer.json --max-context-length 25 --num-embeddings 128 --num-layers 2 --num-heads 2
```

## Results

This was the first reasonable model that was able to produce realistic games beyond 5 moves. (Prior to this, I was training on context size of 5 with small hyperparameters just to get the model working. I did not add that model to `training_models` since it is quick to reproduce.)

This model was trained before I implemented the technique masking out the losing player's moves in the dataset, however, so some of the move choices don't make sense - though up to around 25-30 moves, it is able to make valid predictions before it starts to lose the thread.
