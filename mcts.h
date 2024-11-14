#ifndef MCTS_H
#define MCTS_H

#include "common.h"

#define MCTS_MAX_CHILDREN 218

typedef struct Node
{
    struct Node *parent;
    float wdl[3];

    int8 white_board[64];
    bool board_metadata[6];
    int8 en_passant;
    int8 num_half_moves;

    int8 previous_white_board_one_ply[64];
    int8 previous_white_board_two_ply[64];
    int8 previous_white_board_three_ply[64];
    int8 previous_white_board_four_ply[64];

    float encoded_white_board[64 * 112];
    float encoded_black_board[64 * 112];

    int8 legal_from[MCTS_MAX_CHILDREN];
    int8 legal_to[MCTS_MAX_CHILDREN];
    int8 legal_promotion[MCTS_MAX_CHILDREN];
    float legal_actions_prior[MCTS_MAX_CHILDREN];
    int16 num_legal_actions;

    struct Node *children[MCTS_MAX_CHILDREN];
    int16 children_idx[MCTS_MAX_CHILDREN];
    float total_policy_explored_children;
    int16 num_children;

    int32 num_descents;
    
    float num_descents_virtual_loss;
    float value_virtual_loss;
} Node;

int8 initialize_node_pool(Node **node_pool, int32 total_num_nodes);
void free_node_pool(Node *restrict node_pool);
void initialize_root(
    Node *root,
    float *restrict encoded_white_board,
    float *restrict encoded_black_board,
    int8 *restrict white_board,
    bool *restrict board_metadata,
    int8 en_passant,
    int8 num_half_moves,
    float *restrict policy,
    float w,
    float d,
    float l
);
void reset_virtual_loss(Node *node_pool, int32 num_nodes);
void get_batch(
    Node *node_pool,
    int32 batch_size,
    int32 batch_early_stop,
    float *restrict batched_encoded_boards,
    Node **restrict batch_nodes,
    int32 *restrict num_batched_nodes,
    int32 *restrict num_current_nodes,
    int32 num_total_nodes,
    float c_puct,
    float c_fpu,
    float virtual_loss
);
void put_batch(
    Node **restrict batch_nodes,
    int32 num_batched_nodes,
    float *restrict policy,
    float *restrict wdl
);
void back_propagate_virtual_loss(Node *restrict node, float virtual_loss);
void back_propagate(Node *restrict node, float w, float d, float l);
bool batch_PUCT(
    Node *root,
    Node **batch_node_ptr,
    int32 batch_early_stop,
    int32 *restrict batch_num_iterations,
    int32 *restrict num_current_nodes,
    float c_puct,
    float c_fpu,
    float virtual_loss
);
void mcts_batches_mask_illegal_and_softmax_policy(
    float *restrict policy,
    Node ** batch_nodes,
    int32 num_batched_nodes
);
void get_best_move_and_wdl(
    Node *root,
    int8 *restrict from,
    int8 *restrict to,
    int8 *restrict promotion,
    float *restrict w,
    float *restrict d,
    float *restrict l
);
int16 get_tree_depth(Node *root);
void print_tree_info(
    Node *root,
    int16 max_depth,
    int32 num_nodes,
    int32 num_current_nodes,
    float elapsed_time
);

#endif
