import datetime
import gzip
import random
import struct
import sys
import timeit
import traceback
import yaml
from pathlib import Path

import numpy as np
import torch
from torch.optim import SGD
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from model import ResNetCBAM
import utils

RANDOM_STATE = 1

# Struct format for Leela Chess Zero V6 Training Data
V6_STRUCT = struct.Struct("4si7432s832sBBBBBBBbfffffffffffffffIHH4H")
V6_STRUCT_SIZE = V6_STRUCT.size


def main():
    try:
        utils.seed_everything(RANDOM_STATE)

        with open("config.yaml") as file:
            training_config = yaml.safe_load(file)["training"]

        log_dir = Path(training_config["log_dir"])
        logger = utils.Logger(log_dir / f"training-{datetime.datetime.now().strftime('%y%m%d-%H%M')}.log")
        sys.stdout = logger
        sys.stderr = logger

        print("Training script started.")
        print(f"Training configuration: {training_config}")
        print("Creating / Loading model...")

        device = training_config["device"]
        if training_config["create_new_model"]:
            model = ResNetCBAM()
            Path(training_config["model_dir"]).mkdir(parents=True, exist_ok=True)
            torch.save(model, Path(training_config["model_dir"]) / "model_epoch0.pt")
            model = model.to(device)
        else:
            model = torch.load(
                Path(training_config["model_dir"]) / training_config["load_model_name"],
                map_location=torch.device(device),
                weights_only=False,
            )
        if ("cuda" in device) and (not torch.cuda.is_available()):
            print("Warning: CUDA is not available. Using CPU instead.")
        print("Done!")

        print("Locating training data...")
        training_data_dir = Path(training_config["training_data_dir"])
        if not training_data_dir.exists():
            print("Training data directory not found. Exiting the program...")
            return
        training_data_files = list(training_data_dir.glob("training*"))
        if len(training_data_files) == 0:
            print("The training data directory is empty. Exiting the program...")
            return
        print(f"Found {len(training_data_files)} training data files.")

        print("Initializing training dataset and loader...")
        training_dataset = TrainingDataset(training_data_files, device)
        training_loader = DataLoader(
            training_dataset,
            batch_size=training_config["batch_size"],
            shuffle=True,
            num_workers=training_config["data_loader_num_workers"],
            pin_memory=training_config["data_loader_pin_memory"],
        )

        optimizer = SGD(model.parameters(), lr=training_config["learning_rate"], momentum=training_config["momentum"])
        cross_entropy_loss = torch.nn.CrossEntropyLoss()
        mse_loss = torch.nn.MSELoss()
        relu = torch.nn.ReLU()
        print("Done!")

        print("Training started...")
        policy_weight = training_config["policy_weight"]
        winner_weight = training_config["winner_weight"]
        mlh_weight = training_config["mlh_weight"]

        for epoch in range(
            training_config["current_epochs"] + 1, training_config["num_epochs"] + training_config["current_epochs"] + 1
        ):
            print(f"Epochs: {epoch}")
            start = timeit.default_timer()
            epoch_policy_loss = 0.0
            epoch_policy_loss_mse = 0.0
            epoch_winner_loss = 0.0
            epoch_mlh_loss = 0.0
            epoch_total_loss = 0.0
            batch_policy_loss = 0.0
            batch_policy_loss_mse = 0.0
            batch_winner_loss = 0.0
            batch_mlh_loss = 0.0
            batch_total_loss = 0.0
            model.train()
            for i, (planes, probs_targets, winner_targets, plies_left_targets) in enumerate(training_loader, start=1):
                planes = planes.to(device)
                probs_targets = probs_targets.to(device)
                winner_targets = winner_targets.to(device)
                plies_left_targets = plies_left_targets.to(device)

                optimizer.zero_grad(set_to_none=True)
                policy_logit, winner, plies_left = model(planes)
                policy_logit[probs_targets < 0.0] = -1e10    # Set invalid moves to a very negative value, so that it doesn't affect the softmax calculation

                probs_loss = cross_entropy_loss(policy_logit, relu(probs_targets))
                winner_loss = cross_entropy_loss(winner, winner_targets)
                mlh_loss = mse_loss(plies_left, plies_left_targets.unsqueeze(1))
                loss = policy_weight * probs_loss + winner_weight * winner_loss + mlh_weight * mlh_loss

                loss.backward()
                optimizer.step()

                epoch_policy_loss += probs_loss.item()
                epoch_policy_loss_mse += mse_loss(torch.nn.functional.softmax(policy_logit, dim=1), relu(probs_targets)).item()
                epoch_winner_loss += winner_loss.item()
                epoch_mlh_loss += mlh_loss.item()
                epoch_total_loss += loss.item()

                batch_policy_loss += probs_loss.item()
                batch_policy_loss_mse += mse_loss(torch.nn.functional.softmax(policy_logit, dim=1), relu(probs_targets)).item()
                batch_winner_loss += winner_loss.item()
                batch_mlh_loss += mlh_loss.item()
                batch_total_loss += loss.item()

                if (i % training_config["log_interval"]) == 0:
                    print(f"\tEpoch: {epoch} | Batch: {i} | Policy Loss: {policy_weight * batch_policy_loss / training_config['log_interval']:.3g} | Policy Loss (MSE): {policy_weight * batch_policy_loss_mse / training_config['log_interval']:.3g} | Winner Loss: {winner_weight * batch_winner_loss / training_config['log_interval']:.3g} | MLH Loss: {mlh_weight * batch_mlh_loss / training_config['log_interval']:.3g} | Total Weighted Loss: {batch_total_loss / training_config['log_interval']:.3g}")
                    batch_policy_loss = 0.0
                    batch_policy_loss_mse = 0.0
                    batch_winner_loss = 0.0
                    batch_mlh_loss = 0.0
                    batch_total_loss = 0.0
            end = timeit.default_timer()
            print(f"Epoch {epoch} finished. Policy Loss: {policy_weight * epoch_policy_loss / len(training_loader):.3g} | Policy Loss (MSE): {policy_weight * epoch_policy_loss_mse / len(training_loader):.3g} | Winner Loss: {winner_weight * epoch_winner_loss / len(training_loader):.3g} | MLH Loss: {mlh_weight * epoch_mlh_loss / len(training_loader):.3g} | Total Weighted Loss: {epoch_total_loss / len(training_loader):.3g} | Runtime: {end - start:.0f} seconds.")
            if (epoch % training_config["save_interval"]) == 0:
                print("Saving model...")
                torch.save(
                    model,
                    Path(training_config["model_dir"]) / f"model_epoch{epoch}.pt",
                )

        print("Training finished.")
        
    except:
        print(traceback.format_exc())


class TrainingDataset(Dataset):
    def __init__(self, file_list, device):
        self.file_list = file_list
        self.device = device

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        # Downsample: Only take one round per game
        with gzip.open(self.file_list[idx], "rb") as file:
            content = file.read(256 * V6_STRUCT_SIZE)
            sample = random.randrange(0, len(content) // V6_STRUCT_SIZE)
            content = content[sample * V6_STRUCT_SIZE : (sample + 1) * V6_STRUCT_SIZE]
            planes, probs, winner, plies_left = convert_v6_to_tuple(content)

        return (
            torch.tensor(planes, dtype=torch.float32),
            torch.tensor(probs, dtype=torch.float32),
            torch.tensor(winner, dtype=torch.float32),
            torch.tensor(plies_left, dtype=torch.float32),
        )


def reverse_expand_bits(plane):
    """
    Copied from https://github.com/LeelaChessZero/lczero-training/blob/master/tf/chunkparser.py
    """
    return (
        np.unpackbits(np.array([plane], dtype=np.uint8))[::-1]
        .astype(np.float32)
        .tobytes()
    )


def convert_v6_to_tuple(content):
    """
    Copied and modified from https://github.com/LeelaChessZero/lczero-training/blob/master/tf/chunkparser.py

    Unpack a v6 binary record to 5-tuple (state, policy pi, result, q, m)

    v6 struct format is (8356 bytes total):
                                size         1st byte index
    uint32_t version;                               0
    uint32_t input_format;                          4
    float probabilities[1858];  7432 bytes          8
    uint64_t planes[104];        832 bytes       7440
    uint8_t castling_us_ooo;                     8272
    uint8_t castling_us_oo;                      8273
    uint8_t castling_them_ooo;                   8274
    uint8_t castling_them_oo;                    8275
    uint8_t side_to_move_or_enpassant;           8276
    uint8_t rule50_count;                        8277
    // Bitfield with the following allocation:
    //  bit 7: side to move (input type 3)
    //  bit 6: position marked for deletion by the rescorer (never set by lc0)
    //  bit 5: game adjudicated (v6)
    //  bit 4: max game length exceeded (v6)
    //  bit 3: best_q is for proven best move (v6)
    //  bit 2: transpose transform (input type 3)
    //  bit 1: mirror transform (input type 3)
    //  bit 0: flip transform (input type 3)
    uint8_t invariance_info;                     8278
    uint8_t dep_result;                               8279
    float root_q;                                8280
    float best_q;                                8284
    float root_d;                                8288
    float best_d;                                8292
    float root_m;      // In plies.              8296
    float best_m;      // In plies.              8300
    float plies_left;                            8304
    float result_q;                              8308
    float result_d;                              8312
    float played_q;                              8316
    float played_d;                              8320
    float played_m;                              8324
    // The folowing may be NaN if not found in cache.
    float orig_q;      // For value repair.      8328
    float orig_d;                                8332
    float orig_m;                                8336
    uint32_t visits;                             8340
    // Indices in the probabilities array.
    uint16_t played_idx;                         8344
    uint16_t best_idx;                           8346
    uint64_t reserved;                           8348
    """
    # unpack the V6 content from raw byte array, arbitrarily chose 4 2-byte values
    # for the 8 "reserved" bytes
    (
        ver,
        input_format,
        probs,
        planes,
        us_ooo,
        us_oo,
        them_ooo,
        them_oo,
        stm,
        rule50_count,
        invariance_info,
        dep_result,
        root_q,
        best_q,
        root_d,
        best_d,
        root_m,
        best_m,
        plies_left,
        result_q,
        result_d,
        played_q,
        played_d,
        played_m,
        orig_q,
        orig_d,
        orig_m,
        visits,
        played_idx,
        best_idx,
        reserved1,
        reserved2,
        reserved3,
        reserved4,
    ) = V6_STRUCT.unpack(content)

    # Unpack bit planes and cast to 32 bit float
    planes = np.unpackbits(np.frombuffer(planes, dtype=np.uint8)).astype(np.float32)
    probs = np.frombuffer(probs, dtype=np.float32)

    # Concatenate all planes. Make the last plane all 1's so the NN can
    # detect edges of the board more easily
    planes = np.concatenate(
        [
            planes,
            np.full(64, us_oo, dtype=np.float32),
            np.full(64, us_ooo, dtype=np.float32),
            np.full(64, them_oo, dtype=np.float32),
            np.full(64, them_ooo, dtype=np.float32),
            np.full(64, stm, dtype=np.float32),
            np.full(64, rule50_count, dtype=np.float32),
            np.zeros(64, dtype=np.float32),
            np.ones(64, dtype=np.float32),
        ]
    ).reshape(112, 8, 8)

    winner = (
        0.5 * (1.0 - result_d + result_q),
        result_d,
        0.5 * (1.0 - result_d - result_q),
    )

    return (planes, probs, winner, plies_left)


if __name__ == "__main__":
    main()
