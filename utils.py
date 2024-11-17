import ctypes
import datetime
import os
import random
import sys
import platform
import traceback
from pathlib import Path

import numpy as np
import torch


def load_c_lib():
    """
    Load the C dynamic-link library

    Returns
    -------
    c_lib : ctypes.CDLL
        C dynamic-link library object

    Raises
    ------
    SystemExit
        If the platform is not supported

    SystemExit
        If the user chose to exit the program
        after failed loading of the C library
    """
    if platform.system() == "Windows":
        c_lib_path = str(Path(__file__).parent / "c_lib.dll")
    elif platform.system() == "Darwin":
        c_lib_path = str(Path(__file__).parent / "c_lib.dylib")
    elif platform.system() == "Linux":
        c_lib_path = str(Path(__file__).parent / "c_lib.so")
    else:
        print(f"Platform {platform.system()} not supported. Integration mode is set to NumPy.")
        return None

    try:
        c_lib = ctypes.cdll.LoadLibrary(c_lib_path)
    except:
        traceback.print_exc()
        print()
        print(f"Loading c_lib failed (Path: {c_lib_path}).")
        print("Exiting the program...")

    c_lib.initialize_new_boards.restype = None
    c_lib.initialize_new_boards_batch.restype = None
    c_lib.flip_board.restype = None
    c_lib.flip_board_batch.restype = None
    c_lib.mirror_board.restype = None
    c_lib.mirror_board_batch.restype = None
    c_lib.encode_boards.restype = None
    c_lib.make_move.restype = None
    c_lib.check_king_capture.restype = ctypes.c_bool
    c_lib.check_move_legal.restype = ctypes.c_bool
    c_lib.check_castle_legal.restype = ctypes.c_bool
    c_lib.mask_illegal_and_softmax_policy.restype = None
    c_lib.get_best_move_from_policy.restype = None
    c_lib.is_terminal.restype = ctypes.c_int8

    c_lib.initialize_node_pool.restype = ctypes.c_int8
    c_lib.free_node_pool.restype = None
    c_lib.initialize_root.restype = None
    c_lib.reset_virtual_loss.restype = None
    c_lib.get_batch.restype = None
    c_lib.put_batch.restype = None
    c_lib.back_propagate_virtual_loss.restype = None
    c_lib.back_propagate.restype = None
    c_lib.batch_PUCT.restype = ctypes.c_bool
    c_lib.mcts_batches_mask_illegal_and_softmax_policy.restype = None
    c_lib.get_best_move_and_wdl.restype = None
    c_lib.get_tree_depth.restype = ctypes.c_int16
    c_lib.print_tree_info.restype = None

    return c_lib

def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

class Logger(object):
    def __init__(self, log_path):
        self.terminal = sys.stdout
        self.log_path = log_path
        Path(self.log_path.parent).mkdir(parents=True, exist_ok=True)

    def write(self, msg):
        time_stampped_msg = f"[{datetime.datetime.now().strftime('%y/%m/%d %H:%M:%S')}] {msg}"
        if msg != "\n":
            with open(self.log_path, "a") as file:
                file.write(time_stampped_msg)
            self.terminal.write(time_stampped_msg)
        else:
            with open(self.log_path, "a") as file:
                file.write("\n")
            self.terminal.write("\n")

    def flush(self):
        pass