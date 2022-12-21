import multiprocessing as mp
from pathlib import Path
import random
import re
import sys
from typing import Optional

import chess
import chess.pgn
import numpy as np
import pandas as pd
from pyunpack import Archive
import requests
from tqdm.notebook import tqdm_notebook as tqdm


sys.setrecursionlimit(10_000)


DATA_DIR = Path("./data").resolve()
DATASET_URL = "http://ccrl.chessdom.com/ccrl/4040/CCRL-4040.[1505357].pgn.7z"
DATASET_FILE_COMPRESSED = DATA_DIR / DATASET_URL.rsplit("/")[-1]
DATASET_FILE = DATA_DIR / DATASET_FILE_COMPRESSED.stem

WIN_DATASET = DATA_DIR / "win.pkl"
WIN_VAL_DATASET = DATA_DIR / "win_val.pkl"
LOSS_DATASET = DATA_DIR / "loss.pkl"
LOSS_VAL_DATASET = DATA_DIR / "loss_val.pkl"


def create():
    DATA_DIR.mkdir(exist_ok=True)
    download_dataset()
    unpack_dataset()
    preprocess_dataset()
    split_dataset()
    print("Done")


def download_dataset():
    if DATASET_FILE_COMPRESSED.exists():
        print(f"Dataset archive file found at {DATASET_FILE_COMPRESSED}")
        return

    with open(DATASET_FILE_COMPRESSED, "wb") as f:
        r = requests.get(DATASET_URL, stream=True)

        if (content_length := int(r.headers.get("content-length", 0))) == 0:
            print(f"No dataset file found at {DATASET_URL}")
            return

        for chunk in tqdm(
            r.iter_content(chunk_size=(CHUNK_SIZE := 4096)),
            total=content_length // CHUNK_SIZE + 1,
            desc="Downloading",
        ):
            f.write(chunk)


def unpack_dataset():
    if DATASET_FILE.exists():
        print(f"Dataset file found at {DATASET_FILE}")
        return
    with tqdm(total=1, desc="Unpacking") as pbar:
        Archive(DATASET_FILE_COMPRESSED).extractall(DATA_DIR)
        pbar.update(1)


def to_positions(game: Optional[chess.pgn.Game] = None):
    if game is None:
        return None, None
    board, positions = game.board(), []
    for i, move in enumerate(game.mainline_moves()):
        # not within first 5 moves or capture move
        if i >= 10 and not board.is_capture(move):
            positions.append(board.copy())
        board.push(move)

    positions = positions if len(positions) < 10 else random.sample(positions, 10)
    positions = [{"bitboard": to_bitboard(board)} for board in positions]
    return positions, game.headers["Result"] == "1-0"


def to_bitboard(board: chess.Board):
    bitboard = []

    for color in (chess.WHITE, chess.BLACK):
        for piece in (chess.PAWN, chess.ROOK, chess.KNIGHT, chess.BISHOP, chess.QUEEN, chess.KING):
            bitboard.extend(board.pieces(piece, color).tolist())
    for color in (chess.WHITE, chess.BLACK):
        bitboard.extend((board.has_kingside_castling_rights(color), board.has_queenside_castling_rights(color)))
    bitboard.append(board.turn)

    return np.asarray(bitboard)


def dataset_games():
    with open(DATASET_FILE) as f:
        while game := chess.pgn.read_game(f):
            if game.headers["Result"] in ("1-0", "0-1"):
                yield game
            else: # yield draws to keep the game count accurate
                yield None


def preprocess_dataset():
    if WIN_DATASET.exists() and LOSS_DATASET.exists():
        print(f"Dataset parquet files found at {DATA_DIR}")
        return

    count = int(re.search("\[([0-9]*)]", DATASET_FILE.name).group(1))
    win, loss = pd.DataFrame(columns=["bitboard"]), pd.DataFrame(columns=["bitboard"])
    win_buffer, loss_buffer = [], []

    with mp.Pool(32) as p:
        for positions, is_win in tqdm(p.imap_unordered(to_positions, dataset_games()), total=count, desc="Processing"):
            if not positions:
                continue
            if is_win:
                win_buffer.extend(positions)
                if len(win_buffer) > 10_000:
                    win = pd.concat((win, pd.DataFrame(win_buffer)))
                    win_buffer = []
            else:
                loss_buffer.extend(positions)
                if len(loss_buffer) > 10_000:
                    loss = pd.concat((loss, pd.DataFrame(loss_buffer)))
                    loss_buffer = []

    win = pd.concat((win, pd.DataFrame(win_buffer)))
    loss = pd.concat((loss, pd.DataFrame(loss_buffer)))

    win = win.reset_index(drop=True)
    loss = loss.reset_index(drop=True)

    win.to_pickle(WIN_DATASET)
    loss.to_pickle(LOSS_DATASET)


def split_dataset():
    if WIN_VAL_DATASET.exists() and LOSS_VAL_DATASET.exists():
        print(f"Dataset val parquet files found at {DATA_DIR}")
        return

    for DATA, VAL in tqdm(((WIN_DATASET, WIN_VAL_DATASET), (LOSS_DATASET, LOSS_VAL_DATASET)), desc="Sampling"):
        data = pd.read_pickle(DATA)
        val = data.sample(100_000)
        data = data.drop(index=val.index)
        data = data.reset_index(drop=True)
        val = val.reset_index(drop=True)

        data.to_pickle(DATA)
        val.to_pickle(VAL)
