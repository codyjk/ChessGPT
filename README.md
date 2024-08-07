# `chess-llm`
An LLM that knows how to play chess.

## Setup

This project uses [Poetry](https://python-poetry.org/) for dependency management. Once you have poetry, set up the project and its dependencies with:

```sh
poetry install
```

## Usage

### Preparing training data

The model is trained against games that are represented as sequences of moves in algebraic notation, terminated by the outcome of the game (`1-0`, `0-1`, or `1/2-1/2`).

```
c4 Nf6 Nf3 c5 e3 g6 Nc3 Bg7 Bd3 d5 cxd5 Nxd5 Nxd5 Qxd5 Qb3 Qxb3 axb3 O-O Ra5 Nd7 Bb5 Rd8 Bxd7 Bxd7 Rxc5 b6 Rc7 e5 O-O e4 Ng5 Bb5 Rd1 Bd3 Nxf7 Rdc8 Re7 Rc2 Nd6 Bf8 Rxe4 Bxd6 Rd4 Be2 Re1 Rd8 f4 Bb5 f5 gxf5 g3 Kf7 Rd1 Be7 Rxd8 Bxd8 Kg2 Bc6+ Kf2 Be7 Ke2 Bb5+ Kf3 Bc6+ Ke2 Ke8 Rg1 Bb5+ Kf3 Bc6+ Kf4 a5 Kxf5 Bd7+ Ke4 Bc6+ Kd3 Rc5 b4 Rd5+ Kc2 Ba4+ b3 Bc6 bxa5 Rxa5 Re1 Be4+ Kc3 Bf6+ Kb4 Be7+ Kc3 Rc5+ Kd4 Bb1 e4 Kd7 e5 Rc6 Ba3 Bc5+ Bxc5 bxc5+ Kc4 Bg6 Rf1 Re6 Kd5 Ke7 Ra1 Bf5 Ra7+ Ke8 Ra8+ Ke7 Ra7+ Ke8 Ra4 h5 Rf4 Bg4 Rxg4 Rxe5+ Kxe5 hxg4 Kd5 Kd7 Kxc5 Kc8 Kc6 Kb8 b4 Ka7 b5 Kb8 b6 Kc8 b7+ Kb8 Kb6 1/2-1/2
e4 c6 d4 d5 Nc3 dxe4 Nxe4 Bf5 f3 e6 c3 Nf6 Bd3 Nbd7 Ne2 Be7 O-O O-O N2g3 Bg6 Qc2 Nd5 f4 f5 Nd2 N7f6 Nf3 Bd6 Ng5 Re8 a3 Ng4 Re1 Bxf4 Nxe6 Rxe6 Rxe6 Bxg3 hxg3 Nde3 Bxe3 Qd7 Bc4 Bf7 Qxf5 Nxe3 0-1
...
```

[Lichess](https://database.lichess.org) provides an open database of nearly 6B games in PGN format. Use the `reduce-pgn` script to transform one (or all) of these PGN files into the format described above. You can optionally segment the games by changing the `DIRECTORY_TO_ELO_RATING_RANGE` map in `scripts/reduce_pgn_to_moves.py`. The default configuration yields the following:

```sh
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
