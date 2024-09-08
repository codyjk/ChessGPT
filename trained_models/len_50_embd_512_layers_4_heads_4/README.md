# `trained_models/len_50_embd_512_layers_4_heads_4`

## Hyperparameters

Training details:
* Context length: 50
* Embeddings: 512
* Layers: 4
* Heads: 4
* Epochs: 3
* Training data: 1m master-level games
* Final validation loss: 1.65

Took 4.5 hours to train on a single RTX 3090.

```console
$ poetry run train-model --input-training-data-file out/training-data.csv --input-validation-data-file out/validation-data.csv --input-tokenizer-file out/chess_tokenizer.json --max-context-length 50 --batch-size 1024 --num-embeddings 512 --num-layers 4 --num-heads 4 --num-epochs 3
###################################################################################################
## Training model with args:
Training data:          out/training-data.csv
Validation data:        out/validation-data.csv
Tokenizer file:         out/chess_tokenizer.json
State dict file:        None
Model output file:      out/chess_transformer_model.pth
Max length:             50
Batch size:             1024
Num embeddings:         512
Num layers:             4
Num heads:              4
Num training epochs:    3
Initial learning rate:  0.001
###################################################################################################
Loading tokenizer...
Tokenizer initialized with vocab_size=8708
Loading training/validation data...
Indexing CSV file: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████| 2.08G/2.08G [00:04<00:00, 452MB/s]
Indexing CSV file: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████| 231M/231M [00:00<00:00, 444MB/s]
Using device: cuda
Training Progress:  33%|██████████████████████████▎                                                    | 10265/30795 [1:23:00<3:42:53,  1.54it/s, epoch=1, loss=1.7415]
Epoch 1/3, Train Loss: 2.0229, Val Loss: 1.6629, Learning Rate: 0.001000
Training Progress:  67%|████████████████████████████████████████████████████▋                          | 20530/30795 [2:50:10<1:52:50,  1.52it/s, epoch=2, loss=1.6584]
Epoch 2/3, Train Loss: 1.7065, Val Loss: 1.5965, Learning Rate: 0.001000
Training Progress: 100%|█████████████████████████████████████████████████████████████████████████████████| 30795/30795 [4:16:55<00:00,  1.56it/s, epoch=3, loss=1.6551]
Epoch 3/3, Train Loss: 1.6575, Val Loss: 1.5679, Learning Rate: 0.001000
Training Progress: 100%|█████████████████████████████████████████████████████████████████████████████████| 30795/30795 [4:20:56<00:00,  1.97it/s, epoch=3, loss=1.6551]
Model saved to: out/chess_transformer_model.pth
```

Training data:

```console
$ wc -l out/master.txt out/training-data.csv out/validation-data.csv
  11678553 out/master.txt
  10511160 out/training-data.csv
   1167395 out/validation-data.csv
  23357108 total
```

## Results

A larger version of `trained_models/len_25_embd_128_layers_2_heads_2`, same training data. In this approach and in each example, the winning player's moves are trained against, whereas the losing player's moves are masked out, so much fewer "bad" moves are made, but the model still doesn't actively seek out checkmates. The model is able to make valid (and sometimes competitive) moves way beyond 50, up to 75 sometimes, surprisingly. But it commonly loses the thread in lategame.
