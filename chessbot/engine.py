import math
from pathlib import Path

import chess
import numpy as np

from .environment import to_bitboard
from .model import get_deepchess


DEPTH = 4
BOOK_DIR = Path("./books").resolve()
HUMAN_OPENING = BOOK_DIR / "human.bin"
TITAN_OPENING = BOOK_DIR / "titan.bin"


class Engine:
    def __init__(self, play_white: bool):
        self.play_white = play_white
        self.board = chess.Board()
        self.deepchess = get_deepchess()

    def calculate_move(self):
        move = self.alphabeta(
            self.board, DEPTH, -math.inf, math.inf, self.play_white
        ).move_stack[len(self.board.move_stack)]
        return move

    def alphabeta(self, node: chess.Board, depth: int, α_pos, β_pos, maximizing: bool):
        if depth == 0 or node.is_game_over():
            return node
        if maximizing:
            value = -math.inf
            for move in node.legal_moves:
                child = node.copy()
                child.push(move)

                candidate = self.alphabeta(child, depth - 1, α_pos, β_pos, False)
                value = candidate if value == -math.inf else self.compare(value, candidate)[0]
                α_pos = value if α_pos == -math.inf else self.compare(α_pos, value)[0]

                if β_pos != math.inf:
                    if self.compare(value, β_pos)[0] == candidate:
                        break
            return value
        else:
            value = math.inf
            for move in node.legal_moves:
                child = node.copy()
                child.push(move)
                
                candidate = self.alphabeta(child, depth - 1, α_pos, β_pos, True)
                value = candidate if value == math.inf else self.compare(value, candidate)[1]
                β_pos = value if β_pos == math.inf else self.compare(β_pos, value)[1]

                if α_pos != -math.inf:
                    if self.compare(α_pos, value)[0] == α_pos:
                        break

            return value

    def compare(self, l_board: chess.Board, r_board: chess.Board):
        left_bitboard, right_bitboard = np.reshape(to_bitboard(l_board), (1, 773)), np.reshape(to_bitboard(r_board), (1, 773))
        return (
            (l_board, r_board)
            if np.argmax(self.deepchess([left_bitboard, right_bitboard])) == 0
            else (r_board, l_board)
        )
