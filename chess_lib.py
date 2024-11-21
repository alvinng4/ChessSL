import ctypes
import sys
import timeit

import numpy as np
import torch

MCTS_MAX_CHILDREN = 218

# fmt: off
piece_to_value_map = {
    "None": 0,
    "P": 1,  # White pawn
    "N": 2,  # White knight
    "B": 3,  # White bishop
    "R": 4,  # White rook
    "Q": 5,  # White queen
    "K": 6,  # White king
    "p": -1,  # Black pawn
    "n": -2,  # Black knight
    "b": -3,  # Black bishop
    "r": -4,  # Black rook
    "q": -5,  # Black queen
    "k": -6,  # Black king
}
value_to_piece_map = {v: k for k, v in piece_to_value_map.items()}

en_passant_map = {
    "a6": 0, "b6": 1, "c6": 2, "d6": 3, "e6": 4, "f6": 5, "g6": 6, "h6": 7,
    "a3": 0, "b3": 1, "c3": 2, "d3": 3, "e3": 4, "f3": 5, "g3": 6, "h3": 7,
}

white_board_pos_to_value = {
    "a8": 0, "b8": 1, "c8": 2, "d8": 3, "e8": 4, "f8": 5, "g8": 6, "h8": 7,
    "a7": 8, "b7": 9, "c7": 10, "d7": 11, "e7": 12, "f7": 13, "g7": 14, "h7": 15,
    "a6": 16, "b6": 17, "c6": 18, "d6": 19, "e6": 20, "f6": 21, "g6": 22, "h6": 23,
    "a5": 24, "b5": 25, "c5": 26, "d5": 27, "e5": 28, "f5": 29, "g5": 30, "h5": 31,
    "a4": 32, "b4": 33, "c4": 34, "d4": 35, "e4": 36, "f4": 37, "g4": 38, "h4": 39,
    "a3": 40, "b3": 41, "c3": 42, "d3": 43, "e3": 44, "f3": 45, "g3": 46, "h3": 47,
    "a2": 48, "b2": 49, "c2": 50, "d2": 51, "e2": 52, "f2": 53, "g2": 54, "h2": 55,
    "a1": 56, "b1": 57, "c1": 58, "d1": 59, "e1": 60, "f1": 61, "g1": 62, "h1": 63,
}
white_board_value_to_pos = {v: k for k, v in white_board_pos_to_value.items()}

promotion_to_value = {"": 0, "q": 1, "r": 2, "b": 3, "n": 4}
value_to_promotion = {v: k for k, v in promotion_to_value.items()}
# fmt: on


class Chess:
    """
    Chess class for ChessSL

    parameters
    ----------
    c_lib : ctypes.CDLL
        C library for chess
    model : torch.nn.Module
        Neural network model

    Note
    ----
    This is a minimal implementation for uci protocol.
    We assume that the user will start from a fen string
    and then play moves one by one consecutively.
    """
    class Node(ctypes.Structure):
        pass

    Node._fields_ = [
        ("parent", ctypes.POINTER(Node)),
        ("wdl", ctypes.c_float * 3),
        ("white_board", ctypes.c_int8 * 64),
        ("board_metadata", ctypes.c_bool * 6),
        ("en_passant", ctypes.c_int8),
        ("num_half_moves", ctypes.c_int8),
        ("previous_white_board_one_ply", ctypes.c_int8 * 64),
        ("previous_white_board_two_ply", ctypes.c_int8 * 64),
        ("previous_white_board_three_ply", ctypes.c_int8 * 64),
        ("previous_white_board_four_ply", ctypes.c_int8 * 64),
        ("encoded_white_board", ctypes.c_float * (64 * 112)),
        ("encoded_black_board", ctypes.c_float * (64 * 112)),
        ("legal_from", ctypes.c_int8 * MCTS_MAX_CHILDREN),
        ("legal_to", ctypes.c_int8 * MCTS_MAX_CHILDREN),
        ("legal_promotion", ctypes.c_int8 * MCTS_MAX_CHILDREN),
        ("legal_actions_prior", ctypes.c_float * MCTS_MAX_CHILDREN),
        ("num_legal_actions", ctypes.c_int16),
        ("children", ctypes.POINTER(Node) * MCTS_MAX_CHILDREN),
        ("children_idx", ctypes.c_int16 * MCTS_MAX_CHILDREN),
        ("total_policy_explored_children", ctypes.c_float),
        ("num_children", ctypes.c_int16),
        ("num_descents", ctypes.c_int32),
        ("num_descents_virtual_loss", ctypes.c_float),
        ("value_virtual_loss", ctypes.c_float)
    ]

    def __init__(
            self, 
            c_lib: ctypes.CDLL, 
            model: torch.nn.Module, 
            device: str, 
            batch_size: int,
            c_puct: float,
            c_fpu: float,
            virtual_loss: float,
        ):
        self.c_lib = c_lib
        self.model = model
        self.device = device
        self.batch_size = batch_size
        self.c_puct = c_puct
        self.c_fpu = c_fpu
        self.virtual_loss = virtual_loss
        
        self.batched_encoded_boards = np.zeros((batch_size * 112 * 64), dtype=np.float32)

        self.new_game()

    @torch.no_grad()
    def new_game(self, moves: list=None, fen_string: list = None):
        self.white_board = np.zeros(64, dtype=np.int8)
        self.en_passant_ctypes = ctypes.c_int8(0)
        self.num_half_moves_ctypes = ctypes.c_int8(0)
        self.full_moves = 1 # Start from 1
        # Board metadata
        # 0: white_king_castle
        # 1: white_queen_castle
        # 2: black_king_castle
        # 3: black_queen_castle
        # 4: is_repetition
        # 5: is_white_move
        self.board_metadata = np.array([True, True, True, True, False, True], dtype=bool)
        self.previous_white_board_one_ply = np.zeros(64, dtype=np.int8)
        self.previous_white_board_two_ply = np.zeros(64, dtype=np.int8)
        self.previous_white_board_three_ply = np.zeros(64, dtype=np.int8)
        self.previous_white_board_four_ply = np.zeros(64, dtype=np.int8)
        if fen_string is None:
            self.c_lib.initialize_new_boards(
                self.white_board.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
            )
            self.en_passant_ctypes.value = 0
            self.num_half_moves_ctypes.value = 0
            self.full_moves = 1
        else:
            (
                self.en_passant_ctypes.value,
                self.num_half_moves_ctypes.value,
                self.full_moves,
            ) = self.set_game_to_fen(
                self.white_board,
                self.board_metadata,
                fen_string,
            )

        self.encoded_white_board = np.zeros(int(112 * 64), dtype=np.float32)
        self.encoded_black_board = np.zeros(int(112 * 64), dtype=np.float32)

        self.c_lib.encode_boards(
            self.encoded_white_board.copy().ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            self.encoded_black_board.copy().ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            self.encoded_white_board.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            self.encoded_black_board.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            self.white_board.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
            self.board_metadata.ctypes.data_as(ctypes.POINTER(ctypes.c_bool)),
            self.num_half_moves_ctypes,
        )

        if moves is not None:
            self.moves = moves
        else:
            self.moves = []

        for move in self.moves:
            from_pos = ctypes.c_int8(white_board_pos_to_value[move[:2]])
            to_pos = ctypes.c_int8(white_board_pos_to_value[move[2:4]])

            if len(move) == 5:
                match move[4]:
                    case "q":
                        promotion = ctypes.c_int8(1)
                    case "r":
                        promotion = ctypes.c_int8(2)
                    case "b":
                        promotion = ctypes.c_int8(3)
                    case "n":
                        promotion = ctypes.c_int8(4)
            else:
                promotion = ctypes.c_int8(0)

            self.previous_white_board_four_ply = self.previous_white_board_three_ply.copy()
            self.previous_white_board_three_ply = self.previous_white_board_two_ply.copy()
            self.previous_white_board_two_ply = self.previous_white_board_one_ply.copy()
            self.previous_white_board_one_ply = self.white_board.copy()
            self.c_lib.make_move(
                self.white_board.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
                self.board_metadata.ctypes.data_as(ctypes.POINTER(ctypes.c_bool)),
                ctypes.byref(self.en_passant_ctypes),
                ctypes.byref(self.num_half_moves_ctypes),
                from_pos,
                to_pos,
                promotion,
            )
            if (self.previous_white_board_four_ply == self.white_board).all():
                self.board_metadata[4] = True
            self.c_lib.encode_boards(
                self.encoded_white_board.copy().ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                self.encoded_black_board.copy().ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                self.encoded_white_board.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                self.encoded_black_board.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                self.white_board.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
                self.board_metadata.ctypes.data_as(ctypes.POINTER(ctypes.c_bool)),
                self.num_half_moves_ctypes,
            )

    @staticmethod
    def set_game_to_fen(
        white_board: np.ndarray,
        board_metadata: np.ndarray,
        fen_string: list,
    ):
        white_board[:] = 0

        processed_pos = 0
        for char in fen_string[0]:
            if char == " ":
                break
            if char == "/":
                continue
            elif char.isdigit():
                processed_pos += int(char)
            else:
                white_board[processed_pos] = piece_to_value_map[char]
                processed_pos += 1
        
        if fen_string[1] == "w":
            board_metadata[5] = True
        else:
            board_metadata[5] = False

        if "K" in fen_string[2]:
            board_metadata[0] = True
        else:
            board_metadata[0] = False

        if "Q" in fen_string[2]:
            board_metadata[1] = True
        else:
            board_metadata[1] = False

        if "k" in fen_string[2]:
            board_metadata[2] = True
        else:
            board_metadata[2] = False

        if "q" in fen_string[2]:
            board_metadata[3] = True
        else:
            board_metadata[3] = False

        if fen_string[3] != "-":
            en_passant = en_passant_map[fen_string[3]]
        else:
            en_passant = 0

        half_moves = int(fen_string[4])
        full_moves = int(fen_string[5])

        board_metadata[4] = False

        return en_passant, half_moves, full_moves

    @staticmethod
    def make_moves(
        c_lib: ctypes.CDLL,
        white_board: np.ndarray,
        encoded_white_board: np.ndarray,
        encoded_black_board: np.ndarray,
        board_metadata: np.ndarray,
        en_passant_ctypes: ctypes.c_int8,
        num_half_moves_ctypes: ctypes.c_int8,
        past_moves: str,
        moves: str,
    ):
        new_move_flag = False
        for i, move in enumerate(moves):
            if not new_move_flag:
                if i >= len(past_moves):
                    new_move_flag = True
                elif move != past_moves[i]:
                    new_move_flag = True
            
            if not new_move_flag:
                continue

            from_pos = ctypes.c_int8(white_board_pos_to_value[move[:2]])
            to_pos = ctypes.c_int8(white_board_pos_to_value[move[2:4]])

            if len(move) == 5:
                match move[4]:
                    case "q":
                        promotion = ctypes.c_int8(1)
                    case "r":
                        promotion = ctypes.c_int8(2)
                    case "b":
                        promotion = ctypes.c_int8(3)
                    case "n":
                        promotion = ctypes.c_int8(4)
            else:
                promotion = ctypes.c_int8(0)

            c_lib.make_move(
                white_board.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
                board_metadata.ctypes.data_as(ctypes.POINTER(ctypes.c_bool)),
                ctypes.byref(en_passant_ctypes),
                ctypes.byref(num_half_moves_ctypes),
                from_pos,
                to_pos,
                promotion,
            )
            c_lib.encode_boards(
                encoded_white_board.copy().ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                encoded_black_board.copy().ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                encoded_white_board.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                encoded_black_board.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                white_board.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
                board_metadata.ctypes.data_as(ctypes.POINTER(ctypes.c_bool)),
                num_half_moves_ctypes,
            )

            past_moves.append(move)

    @staticmethod
    @torch.no_grad()
    def initialize_node_pool_and_root_node(
        c_lib: ctypes.CDLL,
        model: torch.nn.Module,
        device: str,
        white_board: np.ndarray, 
        encoded_white_board: np.ndarray,
        encoded_black_board: np.ndarray,
        board_metadata: np.ndarray, 
        en_passant_ctypes: ctypes.c_int8, 
        num_half_moves_ctypes: ctypes.c_int8,
        num_total_nodes: int,
    ) -> ctypes.POINTER:
        if (board_metadata[5]):
            policy, wdl, _ = model(torch.tensor(encoded_white_board).to(device).reshape(1, 112, 8, 8))
        else:
            policy, wdl, _ = model(torch.tensor(encoded_black_board).to(device).reshape(1, 112, 8, 8))

        policy = policy.squeeze(0).cpu().numpy()
        wdl = torch.nn.functional.softmax(wdl, dim=1).squeeze(0).cpu().numpy()

        c_lib.mask_illegal_and_softmax_policy(
            policy.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            white_board.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
            board_metadata.ctypes.data_as(ctypes.POINTER(ctypes.c_bool)),
            en_passant_ctypes,
        )

        node_pool = ctypes.POINTER(Chess.Node)()
        return_value = c_lib.initialize_node_pool(ctypes.byref(node_pool), ctypes.c_int32(num_total_nodes))
        if return_value == 1:
            sys.exit("Error detected in c_lib.initialize_node_pool. Exiting the program...")

        c_lib.initialize_root(
            node_pool,
            encoded_white_board.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            encoded_black_board.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            white_board.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
            board_metadata.ctypes.data_as(ctypes.POINTER(ctypes.c_bool)),
            en_passant_ctypes,
            num_half_moves_ctypes,
            policy.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_float(wdl[0]),
            ctypes.c_float(wdl[1]),
            ctypes.c_float(wdl[2]),
        )

        return node_pool
    
    @staticmethod
    @torch.no_grad()
    def search(
        c_lib: ctypes.CDLL,
        model: torch.nn.Module,
        device: str,
        white_board: np.ndarray,
        encoded_white_board: np.ndarray,
        encoded_black_board: np.ndarray,
        board_metadata: np.ndarray,
        en_passant_ctypes: ctypes.c_int8,
        num_half_moves_ctypes: ctypes.c_int8,
        num_total_nodes: int,
        batched_encoded_boards: np.ndarray,
        batch_size: int,
        batch_early_stop: int,
        c_puct: float,
        c_fpu: float,
        virtual_loss: float,
        is_print_tree_info: bool,
    ) -> str:
        softmax = torch.nn.Softmax(dim=1)
        current_nodes = ctypes.c_int32(1)
        node_pool = Chess.initialize_node_pool_and_root_node(
            c_lib,
            model,
            device,
            white_board,
            encoded_white_board,
            encoded_black_board,
            board_metadata,
            en_passant_ctypes,
            num_half_moves_ctypes,
            num_total_nodes
        )
        num_batched_nodes = ctypes.c_int32(0)
        batch_nodes = (ctypes.POINTER(Chess.Node) * batch_size)()
        
        max_depth = 0   # To print information
        num_empty_batches = 0   # To prevent infinite loop
        start = timeit.default_timer()
        while (current_nodes.value < num_total_nodes):
            start = timeit.default_timer()
            num_batched_nodes.value = 0
            c_lib.get_batch(
                node_pool,
                ctypes.c_int32(batch_size),
                ctypes.c_int32(batch_early_stop),
                batched_encoded_boards.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                batch_nodes,
                ctypes.byref(num_batched_nodes),
                ctypes.byref(current_nodes),
                ctypes.c_int32(num_total_nodes),
                ctypes.c_float(c_puct),
                ctypes.c_float(c_fpu),
                ctypes.c_float(virtual_loss),
            )
            
            policy, wdl, moves_left = model(torch.tensor(batched_encoded_boards[: (num_batched_nodes.value * 112 * 64)]).reshape(num_batched_nodes.value, 112, 8, 8).to(device))
            policy = policy.cpu().numpy()
            wdl = softmax(wdl).cpu().numpy()
            
            c_lib.mcts_batches_mask_illegal_and_softmax_policy(
                policy.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                batch_nodes,
                num_batched_nodes
            )

            c_lib.put_batch(
                batch_nodes,
                num_batched_nodes,
                policy.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                wdl.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            )

            c_lib.reset_virtual_loss(node_pool, current_nodes)

            if num_batched_nodes.value == 0:
                num_empty_batches += 1
            else:
                num_empty_batches = 0

            if num_empty_batches > 10:
                break

            depth = c_lib.get_tree_depth(node_pool)
            if depth > max_depth:
                max_depth = depth
                if is_print_tree_info:
                    c_lib.print_tree_info(
                        node_pool,
                        ctypes.c_int16(depth),
                        num_batched_nodes,
                        current_nodes,
                        ctypes.c_float(timeit.default_timer() - start),
                    )
        if is_print_tree_info:
            c_lib.print_tree_info(
                node_pool,
                ctypes.c_int16(max_depth),
                num_batched_nodes,
                current_nodes,
                ctypes.c_float(timeit.default_timer() - start),
            )
        bestmove_from = ctypes.c_int8(0)
        bestmove_to = ctypes.c_int8(0)
        bestmove_promotion = ctypes.c_int8(0)
        w = ctypes.c_float(0.0)
        d = ctypes.c_float(0.0)
        l = ctypes.c_float(0.0)
        c_lib.get_best_move_and_wdl(
            node_pool,
            ctypes.byref(bestmove_from),
            ctypes.byref(bestmove_to),
            ctypes.byref(bestmove_promotion),
            ctypes.byref(w),
            ctypes.byref(d),
            ctypes.byref(l),
        )

        bestmove_str = white_board_value_to_pos[bestmove_from.value] + white_board_value_to_pos[bestmove_to.value] + value_to_promotion[bestmove_promotion.value]

        c_lib.free_node_pool(node_pool)

        return bestmove_str, [w.value, d.value, l.value]

    def debug_print_board(self):
        print("White board:")
        print(self.white_board.reshape(8, 8))
        print("Black board:")
        black_board = np.zeros(64, dtype=np.int8)
        self.c_lib.flip_board(
            self.white_board.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
            black_board.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
        )
        print(black_board.reshape(8, 8))
