# `trained_models/len_75_embd_1024_layers_8_heads_8`

## Hyperparameters

Training details:
* Context length: 75
* Embeddings: 512
* Layers: 4
* Heads: 4
* Epochs: 3
* Training data: 1.3m "master" and "grandmaster" level games
* Final validation loss: 3.3481

```console
root@ebfe8693b340:~/chess-llm# poetry run train-model --input-training-data-file /workspace/out/train.csv --input-validation-data-file /workspace/out/val.csv --input-tokenizer-file /workspace/out/tokenizer.json --output-model-file /workspace/out/model.pth --max-context-length 75 --batch-size 256 --num-embeddings 1024 --num-layers 8 --num-heads 8 --num-epochs 3
###################################################################################################
## Training model with args:
Training data:          /workspace/out/train.csv
Validation data:        /workspace/out/val.csv
Tokenizer file:         /workspace/out/tokenizer.json
State dict file:        None
Model output file:      /workspace/out/model.pth
Max length:             75
Batch size:             256
Num embeddings:         1024
Num layers:             8
Num heads:              8
Num training epochs:    3
Initial learning rate:  0.001
###################################################################################################
Loading tokenizer...
Tokenizer initialized with vocab_size=12360
Loading training/validation data...
Indexing CSV file: 100%|███████████████████████████████████████████████████████████████████████████████████████████| 3.06G/3.06G [00:03<00:00, 847MB/s]
Indexing CSV file: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 341M/341M [00:06<00:00, 56.6MB/s]
Using device: cuda
Training Progress:  33%|████████████████████▎                                        | 45910/137730 [4:22:58<11:45:24,  2.17it/s, epoch=1, loss=3.6458]
Epoch 1/3, Train Loss: 4.0206, Val Loss: 3.6083, Learning Rate: 0.001000
Training Progress:  67%|█████████████████████████████████████████▎                    | 91820/137730 [8:56:20<6:08:06,  2.08it/s, epoch=2, loss=3.3546]
Epoch 2/3, Train Loss: 3.6444, Val Loss: 3.3835, Learning Rate: 0.001000
Training Progress: 100%|██████████████████████████████████████████████████████████████| 137730/137730 [13:29:40<00:00,  2.19it/s, epoch=3, loss=3.3481]
Epoch 3/3, Train Loss: 3.5500, Val Loss: 3.3263, Learning Rate: 0.001000
Training Progress: 100%|██████████████████████████████████████████████████████████████| 137730/137730 [13:39:55<00:00,  2.80it/s, epoch=3, loss=3.3481]
Model saved to: /workspace/out/model.pth
```

## Results

Surprisingly, this model did not converge nearly as well as the smaller models. This manifests in the gameplay that terminates early because the model fails to find a valid move. More investigation is needed to understand this.
