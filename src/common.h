#ifndef COMMON_H
#define COMMON_H

/**
 * common: Contains methods for ChessSL
 */

#include <stdbool.h>
#include <stdint.h>

typedef int8_t int8;
typedef int16_t int16;
typedef int32_t int32;
typedef int64_t int64;

#define PIECE_NONE 0
#define PIECE_WHITE_PAWN 1
#define PIECE_WHITE_KNIGHT 2
#define PIECE_WHITE_BISHOP 3
#define PIECE_WHITE_ROOK 4
#define PIECE_WHITE_QUEEN 5
#define PIECE_WHITE_KING 6
#define PIECE_BLACK_PAWN -1
#define PIECE_BLACK_KNIGHT -2
#define PIECE_BLACK_BISHOP -3
#define PIECE_BLACK_ROOK -4
#define PIECE_BLACK_QUEEN -5
#define PIECE_BLACK_KING -6

extern const int16 white_policy_idx_to_moves_from[1858];
extern const int16 white_policy_idx_to_moves_to[1858];
extern const int16 black_policy_idx_to_moves_from[1858];
extern const int16 black_policy_idx_to_moves_to[1858];

/**
 * @brief Indices for the possible pawn promotions to knight
 */
extern const int16 possible_knight_promotion_idx[22];

/**
 * @brief This array provides a way to get the index to the policy_idx_to_move array.
 *        Since the board starts at a8, the array starts with
 *        "a8a8", "a8b8", ..., "a8h8", "a8a7", ..., "h1h1".
 *        Then, it continues with "a7a8q", "a7a8r", "a7a8b", "a7a8", "a7b8q", ..., "h7h8b".
 *        (i.e. Queen, Rook, Bishop, Knight).
 */
extern const int16 policy_idx[4226];
extern const char *pos_to_utf8[64];


void initialize_new_boards(int8 *restrict white_board);
void initialize_new_boards_batch(int8 *restrict white_boards, int32 batch_size);
void flip_board(int8 *restrict original_board, int8 *restrict flipped_board);
void flip_board_batch(int8 *restrict original_boards, int8 *restrict flipped_boards, int32 batch_size);
void mirror_board(int8 *restrict original_board, int8 *restrict mirrored_board);
void mirror_board_batch(int8 *restrict original_boards, int8 *restrict mirrored_boards, int32 batch_size);
void encode_boards(
    float *restrict old_encoded_white_board,
    float *restrict old_encoded_black_board,
    float *restrict new_encoded_white_board,
    float *restrict new_encoded_black_board,
    int8 *restrict white_board,
    bool *restrict board_metadata,
    int8 num_half_moves
);
void make_move(
    int8 *restrict white_board,
    bool *restrict board_metadata,
    int8 *restrict en_passant, 
    int8 *restrict num_half_moves,
    int8 from_pos,
    int8 to_pos,
    int8 promotion
);
bool check_king_capture(
    int8 *restrict board,
    int8 white_king_pos,
    int8 black_king_pos
);
bool check_move_legal(
    int8 *restrict board,
    int8 white_king_pos,
    int8 black_king_pos,
    int8 from,
    int8 to
);
bool check_castle_legal(
    int8 *restrict board,
    int8 black_king_pos,
    int8 castle_id
);
void mask_illegal_and_softmax_policy(
    float *restrict policy,
    int8 *restrict white_board,
    bool *restrict board_metadata,
    int8 en_passant
);
void get_best_move_from_policy(
    float *restrict policy,
    int8 *restrict white_board,
    bool *restrict board_metadata,
    int8 *restrict best_move_utf8
);

int8 is_terminal(
    int8 *restrict white_board,
    bool *restrict board_metadata,
    int8 en_passant
);

#endif
