# `chess-llm`
An LLM that knows how to play chess.

## Setup

This project uses [Poetry](https://python-poetry.org/) for dependency management. Once you have poetry, set up the project and its dependencies with:

```sh
poetry install
```

## Usage

### Preparing training data

#### Reducing PGN files

The model is trained against games that are represented as sequences of moves in algebraic notation, terminated by the outcome of the game (`1-0`, `0-1`, or `1/2-1/2`).

```
c4 Nf6 Nf3 c5 e3 g6 Nc3 Bg7 Bd3 d5 cxd5 Nxd5 Nxd5 Qxd5 Qb3 Qxb3 axb3 O-O Ra5 Nd7 Bb5 Rd8 Bxd7 Bxd7 Rxc5 b6 Rc7 e5 O-O e4 Ng5 Bb5 Rd1 Bd3 Nxf7 Rdc8 Re7 Rc2 Nd6 Bf8 Rxe4 Bxd6 Rd4 Be2 Re1 Rd8 f4 Bb5 f5 gxf5 g3 Kf7 Rd1 Be7 Rxd8 Bxd8 Kg2 Bc6+ Kf2 Be7 Ke2 Bb5+ Kf3 Bc6+ Ke2 Ke8 Rg1 Bb5+ Kf3 Bc6+ Kf4 a5 Kxf5 Bd7+ Ke4 Bc6+ Kd3 Rc5 b4 Rd5+ Kc2 Ba4+ b3 Bc6 bxa5 Rxa5 Re1 Be4+ Kc3 Bf6+ Kb4 Be7+ Kc3 Rc5+ Kd4 Bb1 e4 Kd7 e5 Rc6 Ba3 Bc5+ Bxc5 bxc5+ Kc4 Bg6 Rf1 Re6 Kd5 Ke7 Ra1 Bf5 Ra7+ Ke8 Ra8+ Ke7 Ra7+ Ke8 Ra4 h5 Rf4 Bg4 Rxg4 Rxe5+ Kxe5 hxg4 Kd5 Kd7 Kxc5 Kc8 Kc6 Kb8 b4 Ka7 b5 Kb8 b6 Kc8 b7+ Kb8 Kb6 1/2-1/2
e4 c6 d4 d5 Nc3 dxe4 Nxe4 Bf5 f3 e6 c3 Nf6 Bd3 Nbd7 Ne2 Be7 O-O O-O N2g3 Bg6 Qc2 Nd5 f4 f5 Nd2 N7f6 Nf3 Bd6 Ng5 Re8 a3 Ng4 Re1 Bxf4 Nxe6 Rxe6 Rxe6 Bxg3 hxg3 Nde3 Bxe3 Qd7 Bc4 Bf7 Qxf5 Nxe3 0-1
...
```

[Lichess](https://database.lichess.org) provides an open database of nearly 6B games in PGN format. Use the `reduce-pgn` script to transform one (or all) of these PGN files into the format described above. You can optionally segment the games by changing the `DIRECTORY_TO_ELO_RATING_RANGE` map in `scripts/reduce_pgn_to_moves.py`.

```console
$ poetry run reduce-pgn --help
usage: reduce-pgn [-h] --input-pgn INPUT_PGN --output-dir OUTPUT_DIR

Reduce a chess games PGN file to a list of moves.

options:
  -h, --help            show this help message and exit
  --input-pgn INPUT_PGN
                        The input PGN file.
  --output-dir OUTPUT_DIR
                        The output directory.
```

The default configuration for ELO ranges yields the following:

```console
$ poetry run reduce-pgn --input-pgn data/lichess_db_standard_rated_2024-06.pgn --output-dir out
Processing file...
Processed 1116435 games in beginner.
Processed 3120115 games in intermediate.
Processed 3285341 games in master.
Processed 111121 games in grandmaster.
Processed 0 games in unknown.

$ head -n 5 out/grandmaster.txt
d4 a5 Bg5 Ra6 Nf3 Rg6 h4 d6 Nbd2 b6 e3 Bb7 Bd3 Na6 Bxg6 hxg6 c3 Qa8 Qc2 e6 O-O-O Nh6 e4 c5 dxc5 Nxc5 Rhe1 Ng4 Nc4 Qb8 b4 Na6 Qa4+ b5 Qxb5+ Bc6 Qxc6# 1-0
d4 e6 Nf3 b6 c4 Bb7 Nc3 Bb4 g3 f5 Qc2 Nf6 Bg2 O-O O-O Bxc3 bxc3 d6 Bg5 Nbd7 a4 a5 Rfb1 Qe8 Bxf6 Nxf6 c5 dxc5 dxc5 Be4 Qb3 Bxb1 Rxb1 bxc5 Ng5 Ra6 Qb7 Rb6 Rxb6 cxb6 Qxb6 Qxa4 Qxe6+ Kh8 Nf7+ Rxf7 Qxf7 Qe8 Qc7 h6 Qxa5 Qxe2 Qxc5 Ng4 h3 Ne5 Qc8+ Kh7 Bd5 Kg6 Qe6+ Kh7 Qg8+ Kg6 Qe6+ Kh7 Qxf5+ Kh8 Be4 Kg8 Qxe5 Qe1+ Kg2 Qd2 Qd5+ 1-0
g3 e6 Bg2 d5 d3 f5 Nd2 Nf6 e4 c6 e5 Nfd7 d4 Be7 c3 O-O f4 c5 Ne2 cxd4 cxd4 Nc6 Nf3 Bb4+ Kf2 Qb6 Be3 h6 h4 Re8 h5 Nf8 Bh3 Qd8 g4 fxg4 Bxg4 Qe7 Qd3 Qf7 Rag1 Bd7 Nh4 a6 Ng3 Ne7 Kg2 Bb5 Qb1 Rac8 f5 Nxf5 Ngxf5 exf5 Nxf5 Ne6 Kh2 Kh8 Rf1 Qc7 Rf2 Be7 Rg1 Bg5 Qe1 Bxe3 Qxe3 Ng5 Nh4 Qe7 Ng6+ Kg8 Nxe7+ Rxe7 Bxc8 1-0
d4 Nf6 Nf3 e6 e3 c5 Be2 cxd4 exd4 d6 c4 Be7 Nc3 O-O O-O a6 d5 Nbd7 dxe6 fxe6 Nd4 Ne5 f4 Ng6 f5 e5 fxg6 exd4 gxh7+ Nxh7 Qxd4 Rxf1+ Bxf1 Bf6 Qd5+ Kh8 Bf4 Qb6+ Kh1 Qxb2 Re1 Bd7 Ne4 Bc6 Qxd6 Bxe4 Rxe4 Qb1 Qd3 Qxa2 Be5 Rd8 Qe2 Qxe2 Bxe2 Re8 Rh4 Rxe5 Bd3 Re1+ 0-1
g3 Nc6 Bg2 b6 Nf3 Bb7 O-O g6 d4 Bg7 c4 d6 Nc3 e6 e4 Nge7 Re1 O-O Be3 d5 cxd5 exd5 e5 f6 exf6 Bxf6 Qd2 Nf5 Ne5 Nxe3 fxe3 Bxe5 dxe5 d4 exd4 Qxd4+ Qxd4 Nxd4 Bxb7 Rae8 Bd5+ Kh8 Rf1 b5 Rxf8+ Rxf8 e6 b4 Ne4 Kg7 Rf1 Re8 Ng5 c6 Rf7+ Kh6 h4 Kh5 Kg2 cxd5 Kh3 h6 Rh7 Nxe6 Nf7 Nf4+ gxf4 Re3+ Kg2 Re7 Rxh6+ Kg4 Rxg6+ Kxf4 Nd6 Re2+ Kh3 Rxb2 h5 Ke5 Nf7+ Kf5 Rg5+ Kf6 Rxd5 Kxf7 Rb5 1-0
```

#### Generating training and validation data sets

Once the games have been reduced to the format described above, you can use the `prepare-training-data` script to generate training and validation data sets.

```console
$ poetry run prepare-training-data --help
usage: prepare-training-data [-h] --input-reduced-pgn-file INPUT_REDUCED_PGN_FILE
                             [--output-training-data-file OUTPUT_TRAINING_DATA_FILE]
                             [--output-validation-data-file OUTPUT_VALIDATION_DATA_FILE]
                             [--max-context-length MAX_CONTEXT_LENGTH]
                             [--validation-split VALIDATION_SPLIT]

Prepares training and validation data sets for the model training step.

options:
  -h, --help            show this help message and exit
  --input-reduced-pgn-file INPUT_REDUCED_PGN_FILE
                        The input file, as returned by `poetry run reduce-pgn`.
  --output-training-data-file OUTPUT_TRAINING_DATA_FILE
                        Where to save the training data. Default: out/training-data.csv
  --output-validation-data-file OUTPUT_VALIDATION_DATA_FILE
                        Where to save the validation data. Default: out/validation-data.csv
  --max-context-length MAX_CONTEXT_LENGTH
                        The maximum number of moves to include in the context for the
                        examples written to the training and validation data files. Default:
                        10
  --validation-split VALIDATION_SPLIT
                        The proportion of the data to use for validation. Default: 0.1
```

Depending on the file size, this may take a while, but you will see the progress as the script runs:

```console
$ poetry run prepare-training-data --input-reduced-pgn-file out/all-trunc.txt --max-context-length 5
Processing games: 100%|███████████████████████████| 500000/500000 [00:02<00:00, 176300.05it/s]
Training data written to: out/training-data.csv
Validation data written to: out/validation-data.csv
```

#### Preparing the tokenizer

After preparing the training data, use the `fit-and-save-tokenizer` script to fit the tokenizer to the training data. Pre-generating the tokenizer guarantees that the same tokenization is used when training and running the model.

```console
$ poetry run fit-and-save-tokenizer --help
usage: fit-and-save-tokenizer [-h] --input-training-data-file INPUT_TRAINING_DATA_FILE
                              [--output-tokenizer-file OUTPUT_TOKENIZER_FILE]

Fit and save the tokenizer.

options:
  -h, --help            show this help message and exit
  --input-training-data-file INPUT_TRAINING_DATA_FILE
                        The input training data file, as returned by `poetry run prepare-
                        training-data`
  --output-tokenizer-file OUTPUT_TOKENIZER_FILE
                        Where to save tokenizer state. Default: out/chess_tokenizer.json
```

For example:

```console
$ poetry run fit-and-save-tokenizer --input-training-data-file out/training-data.csv
Fitting tokenizer...
Counting lines in out/training-data.csv...
Found 2245190 lines.
Processing moves: 100%|████████████████████████| 2245190/2245190 [00:01<00:00, 1757451.32it/s]
Tokenizer initialized with vocab_size=388
Tokenizer saved to: out/chess_tokenizer.json
```

#### Training the model

After preparing the training data and validation data sets using `prepare-training-data`, the `train-model` command will train the model with the given hyperparameters, and save the resulting model to a `.pth` file.


```console
$ poetry run train-model --help
usage: train-model [-h] --input-training-data-file INPUT_TRAINING_DATA_FILE
                   --input-validation-data-file INPUT_VALIDATION_DATA_FILE
                   [--input-tokenizer-file INPUT_TOKENIZER_FILE]
                   [--input-state-dict-file INPUT_STATE_DICT_FILE]
                   [--output-model-file OUTPUT_MODEL_FILE]
                   [--max-context-length MAX_CONTEXT_LENGTH] [--batch-size BATCH_SIZE]
                   [--num-embeddings NUM_EMBEDDINGS] [--num-layers NUM_LAYERS]
                   [--num-heads NUM_HEADS] [--num-epochs NUM_EPOCHS]
                   [--initial-learning-rate INITIAL_LEARNING_RATE]
                   [--show-random-baseline SHOW_RANDOM_BASELINE]

Train the LLM.

options:
  -h, --help            show this help message and exit
  --input-training-data-file INPUT_TRAINING_DATA_FILE
                        The input training data file, as returned by `poetry run prepare-
                        training-data`
  --input-validation-data-file INPUT_VALIDATION_DATA_FILE
                        The input validation data file, as returned by `poetry run prepare-
                        training-data`
  --input-tokenizer-file INPUT_TOKENIZER_FILE
                        The tokenizer file, as generated by `poetry run fit-and-save-tokenizer`.
                        Defaults to out/chess_tokenizer.json
  --input-state-dict-file INPUT_STATE_DICT_FILE
                        The state dict file to load the initial model from. If not provided,
                        the model will be randomly initialized.
  --output-model-file OUTPUT_MODEL_FILE
                        Where to save the `.pth` file for the trained model. Default:
                        out/chess_transformer_model.pth
  --max-context-length MAX_CONTEXT_LENGTH
                        The maximum context length (number of moves) to train against.
                        Default: 50
  --batch-size BATCH_SIZE
                        The batch size to use. Default: 128
  --num-embeddings NUM_EMBEDDINGS
                        The number of embeddings to use in the model. Default: 256
  --num-layers NUM_LAYERS
                        The number of layers to use in the model. Default: 4
  --num-heads NUM_HEADS
                        The number of heads to use in the model. Default: 4
  --num-epochs NUM_EPOCHS
                        The number of epochs to train the model for. Default: 10
  --initial-learning-rate INITIAL_LEARNING_RATE
                        The initial learning rate to use. Default: 0.001
  --show-random-baseline SHOW_RANDOM_BASELINE
                        Whether to show the random baseline loss. Default: True
```

Here's an example with a small model and dataset:

```console
$ poetry run train-model \
    --input-training-data-file out/training-data.csv \
    --input-validation-data-file out/validation-data.csv \
    --input-tokenizer-file out/chess_tokenizer.json \
    --max-context-length 5 \
    --num-embeddings 64 \
    --num-epochs 3 \
    --batch-size 32 \
    --num-layers 1 \
    --num-heads 1

###################################################################################################
## Training model with args:
Training data:          out/training-data.csv
Validation data:        out/validation-data.csv
Tokenizer file:         out/chess_tokenizer.json
State dict file:        None
Model output file:      out/chess_transformer_model.pth
Max length:             5
Batch size:             32
Num embeddings:         64
Num layers:             1
Num heads:              1
Num training epochs:    3
Initial learning rate:  0.001
###################################################################################################
Loading tokenizer...
Tokenizer initialized with vocab_size=305
Loading training/validation data...
Indexing CSV file: 100%|█████████████████████████████████| 6.32M/6.32M [00:00<00:00, 36.8MB/s]
Indexing CSV file: 100%|███████████████████████████████████| 704k/704k [00:00<00:00, 34.6MB/s]
Using device: mps
Calculating random baseline: 100%|█████████████████████| 14028/14028 [00:20<00:00, 677.71it/s]
Random baseline loss: 5.7613
Epoch 1/3, Train Loss: 1.9003, Val Loss: 1.7609, Learning Rate: 0.001000
Training Progress:  67%|██████▋   | 28056/42084 [05:35<02:49, 82.63it/s, epoch=2, loss=1.7381]
Epoch 2/3, Train Loss: 1.8021, Val Loss: 1.7376, Learning Rate: 0.001000
Training Progress: 100%|██████████| 42084/42084 [08:22<00:00, 87.52it/s, epoch=3, loss=1.9243]
Epoch 3/3, Train Loss: 1.7790, Val Loss: 1.7136, Learning Rate: 0.001000
Training Progress: 100%|██████████| 42084/42084 [08:27<00:00, 82.98it/s, epoch=3, loss=1.9243]
Model saved to: out/chess_transformer_model.pth
```

### Playing against the model

Once the model is trained, you can play a chess game against it using the `play` script. You will need to use the same hyperparameters you used to train the model.

```console
$ poetry run play --help
usage: play [-h] --input-model-file INPUT_MODEL_FILE --input-tokenizer-file
            INPUT_TOKENIZER_FILE --max-context-length MAX_CONTEXT_LENGTH --num-embeddings
            NUM_EMBEDDINGS --num-layers NUM_LAYERS --num-heads NUM_HEADS
            [--color {white,black}] [--top-k TOP_K] [--debug]

Chess CLI for playing against a trained model

options:
  -h, --help            show this help message and exit
  --input-model-file INPUT_MODEL_FILE
                        Path to the trained model file
  --input-tokenizer-file INPUT_TOKENIZER_FILE
                        Path to the tokenizer file
  --max-context-length MAX_CONTEXT_LENGTH
                        The maximum context length (number of moves) that the model was
                        trained against.
  --num-embeddings NUM_EMBEDDINGS
                        The number of embeddings that the model was trained with.
  --num-layers NUM_LAYERS
                        The number of layers that the model was trained with.
  --num-heads NUM_HEADS
                        The number of heads that the model was trained with.
  --color {white,black}
                        Player's color
  --top-k TOP_K         Top-k sampling parameter
  --debug               Enable debug output
```

When it boots up, you will be dropped into a chess game in the terminal! You will need to enter moves in algebraic notation.

```console
    h   g   f   e   d   c   b   a
  ┌───┬───┬───┬───┬───┬───┬───┬───┐
1 │ ♜ │   │ · │ ♚ │ ♛ │   │ · │ ♜ │ 1
  ├───┼───┼───┼───┼───┼───┼───┼───┤
2 │ ♟ │ ♟ │ ♟ │ ♝ │   │ · │   │ ♟ │ 2
  ├───┼───┼───┼───┼───┼───┼───┼───┤
3 │ · │   │ ♞ │ ♟ │ · │ ♞ │ · │   │ 3
  ├───┼───┼───┼───┼───┼───┼───┼───┤
4 │   │ · │ ♝ │ · │ ♟ │ ♟ │   │ · │ 4
  ├───┼───┼───┼───┼───┼───┼───┼───┤
5 │ · │   │ · │   │ · │   │ · │   │ 5
  ├───┼───┼───┼───┼───┼───┼───┼───┤
6 │   │ ♙ │ ♘ │ · │   │ ♘ │   │ ♙ │ 6
  ├───┼───┼───┼───┼───┼───┼───┼───┤
7 │ ♙ │ ♗ │ ♙ │ ♙ │ ♙ │ ♙ │ · │   │ 7
  ├───┼───┼───┼───┼───┼───┼───┼───┤
8 │   │ ♔ │ ♖ │ · │ ♕ │ ♗ │   │ ♖ │ 8
  └───┴───┴───┴───┴───┴───┴───┴───┘
    h   g   f   e   d   c   b   a

Moves played: d4 Nc6 Nf3 g6 Bf4 Bg7 e3 Nf6 c4 O-O Be2 a6 Nc3 b5 b3 bxc4 bxc4
Enter your move:
```

## Quick way of validating that the model actually works

A quick way to validate that the model works is to use a small dataset of ~100,000 games with small hyperperameters, like the example shown above.

For convenience, here are the steps:

```sh
# Reduce the PGN
poetry run reduce-pgn --input-pgn data/lichess_db_standard_rated_2024-06.pgn --output-dir out

# Take a subset of the games
shuf -n 100000 out/master.txt > out/master-trunc.txt

# Prepare the training data
poetry run fit-and-save-tokenizer --input-training-data-file out/training-data.csv

# Train the model. Make sure to use the same max-context-length as above
poetry run train-model --input-training-data-file out/training-data.csv --input-validation-data-file out/validation-data.csv --input-tokenizer-file out/chess_tokenizer.json --max-context-length 5 --num-embeddings 64 --num-epochs 3 --batch-size 32 --num-layers 1 --num-heads 1
```

This takes about 5-10 minutes end-to-end on an M1 Macbook Pro, and produces a model that can play 5-10 "good" moves before it starts to lose the thread.

Then, either play the model using the `play` script to see what kinds of moves it makes, open the `run_trained_model.ipynb` notebook to explore the model and plug in any list of moves that you want. In either case, make sure the hyperparameters at the top match the ones used above.
