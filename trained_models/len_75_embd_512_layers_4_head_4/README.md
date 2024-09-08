# `trained_models/len_75_embd_512_layers_4_head_4`

## Hyperparameters

Training details:
* Context length: 75
* Embeddings: 512
* Layers: 4
* Heads: 4
* Training data: 1,000,000 "master" level games
* Epochs: 7
* Final validation loss: 1.70

```sh
poetry run train-model --input-training-data-file /workspace/out/train.csv --input-validation-data-file /workspace/out/val.csv --input-tokenizer-file /workspace/out/tokenizer.json --max-context-length 75 --batch-size 512 --num-embeddings 512 --num-layers 4 --num-heads 4 --num-epochs 7 --output-model-file /workspace/out/len_75_embd_512_layers_4_heads_4.pth
```

## Results

This is the same dataset as the `trained_models/len_75_embd_512_layers_4_head_4` model. The context window has extended to 75. It makes stronger moves after move 50, and was able to make valid moves to 80-90 in some cases, though it struggles when there are few pieces on the board lategame. I think that is partially due to the small sample size (1m games), and high degree of variability in endgame positions.
