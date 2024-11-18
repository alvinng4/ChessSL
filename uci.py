import yaml
from pathlib import Path

import numpy as np
import torch

import utils
from chess_lib import Chess

RANDOM_STATE = 1
utils.seed_everything(RANDOM_STATE)

def main():
    print("ChessSL version 0.0.1")
    c_lib = utils.load_c_lib()
    
    if not (Path(__file__).parent / "config.yaml").exists():
        raise FileNotFoundError("config.yaml not found.")

    with open(Path(__file__).parent / "config.yaml") as file:
        engine_config = yaml.safe_load(file)["engine"]

    model = torch.load(engine_config["model_path"], map_location=torch.device(engine_config["device"]), weights_only=False)
    model.eval()

    chess = Chess(
        c_lib, 
        model, 
        engine_config["device"], 
        engine_config["mcts_batch_size"],
        engine_config["c_puct"],
        engine_config["c_fpu"],
        engine_config["virtual_loss"],
    )
    
    show_wdl = False
    wdl = None
    while True:
        user_input = input().strip().split()
        user_input_len = len(user_input)
        if user_input_len > 0:
            match user_input[0]:
                case "uci":
                    print("id name ChessSL")
                    print("id author Ching-Yin Ng")
                    print()
                    print("option name UCI_ShowWDL type check default false")
                    print("uciok")
                case "isready":
                    print("readyok")
                case "ucinewgame":
                    chess.new_game()
                case "position":
                    if user_input[1] == "startpos":
                        if user_input_len == 2:
                            chess.new_game()
                        elif user_input_len > 3:
                            if (user_input_len > 3 + len(chess.moves)) and ((user_input[3:(3 + len(chess.moves) + 1)] == chess.moves)):
                                chess.make_moves(
                                    c_lib,
                                    chess.white_board,
                                    chess.encoded_white_board,
                                    chess.encoded_black_board,
                                    chess.board_metadata,
                                    chess.en_passant_ctypes,
                                    chess.num_half_moves_ctypes,
                                    chess.moves,
                                    user_input[3:],
                                )
                            else:
                                chess.new_game(moves=user_input[3:])
                    elif user_input[1] == "fen":
                        if user_input_len == 8:
                            chess.new_game(fen_string=user_input[2:])
                        elif user_input_len > 9:
                            chess.new_game(fen_string=user_input[2:9], moves=user_input[9:])
                case "go":
                    nodes = engine_config["base_num_nodes"]

                    # Starting move
                    if engine_config["use_starting_move"] and (
                        chess.white_board == np.array(
                            [
                                -4, -2, -3, -5, -6, -3, -2, -4,
                                -1, -1, -1, -1, -1, -1, -1, -1,
                                0, 0, 0, 0, 0, 0, 0, 0,
                                0, 0, 0, 0, 0, 0, 0, 0,
                                0, 0, 0, 0, 0, 0, 0, 0,
                                0, 0, 0, 0, 0, 0, 0, 0,
                                1, 1, 1, 1, 1, 1, 1, 1,
                                4, 2, 3, 5, 6, 3, 2, 4
                            ], dtype=np.int8
                        )
                    ).all():
                        best_move = engine_config["starting_move"]

                    # Search
                    else: 

                        if chess.board_metadata[5] and "wtime" in user_input:
                            wtime_index = user_input.index("wtime")
                            wtime = int(user_input[wtime_index + 1])
                            nodes = engine_config["engine_move_time_factor"] * engine_config["nps"] * wtime / 1000
                        
                        elif (not chess.board_metadata[5]) and "btime" in user_input:
                            btime_index = user_input.index("btime")
                            btime = int(user_input[btime_index + 1])
                            nodes = engine_config["engine_move_time_factor"] * engine_config["nps"] * btime / 1000

                        nodes = int((nodes // engine_config["mcts_batch_size"] + 1) * engine_config["mcts_batch_size"])
                        if nodes > engine_config["max_num_nodes"]:
                            nodes = engine_config["max_num_nodes"]
                        best_move, wdl = chess.search(
                            chess.c_lib,
                            chess.model,
                            chess.device,
                            chess.white_board,
                            chess.encoded_white_board,
                            chess.encoded_black_board,
                            chess.board_metadata,
                            chess.en_passant_ctypes,
                            chess.num_half_moves_ctypes,
                            nodes,
                            chess.batched_encoded_boards,
                            chess.batch_size,
                            engine_config["mcts_batches_early_stop"],
                            chess.c_puct,
                            chess.c_fpu,
                            chess.virtual_loss,
                            engine_config["print_tree_info"],
                        )          
                    if show_wdl and (wdl is not None):
                        print(f"info wdl {wdl[0] * 1000.0 :.0f} {wdl[1] * 1000.0:.0f} {wdl[2] * 1000.0:.0f}")
                    print(f"bestmove {best_move}")
                case "setoption":
                    if len(user_input) >= 5 and user_input[1] == "name" and user_input[2] == "UCI_ShowWDL":
                        if (user_input[4] == "true"):
                            show_wdl = True
                        elif (user_input[4] == "false"):
                            show_wdl = False
                case "quit":
                    break
                case "debug":
                    chess.debug_print_board()
                case _:
                    print(f"Unknown command {user_input[0]}")

        


if __name__ == "__main__":
    main()
    










