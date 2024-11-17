#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "common.h"
#include "mcts.h"

int8 initialize_node_pool(Node **node_pool, int32 total_num_nodes)
{
    // printf("Amount of memory to allocate in GB: %f\n", total_num_nodes * sizeof(Node) / 1024.0 / 1024.0 / 1024.0);
    *node_pool = malloc(total_num_nodes * sizeof(Node));
    if (*node_pool == NULL)
    {
        fprintf(stderr, "Error: Unable to allocate memory for node pool in initialize_node_pool()\n");
        return 1;
    }
    return 0;
}

void free_node_pool(Node *restrict node_pool)
{
    free(node_pool);
}

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
)
{
    root->parent = NULL;
    root->wdl[0] = w;
    root->wdl[1] = d;
    root->wdl[2] = l;

    memcpy(root->encoded_white_board, encoded_white_board, 112 * 64 * sizeof(float));
    memcpy(root->encoded_black_board, encoded_black_board, 112 * 64 * sizeof(float));
    memcpy(root->white_board, white_board, 64 * sizeof(int8));
    memcpy(root->board_metadata, board_metadata, 6 * sizeof(bool));
    root->en_passant = en_passant;
    root->num_half_moves = num_half_moves;

    root->num_legal_actions = 0;
    for (int16 i = 0; i < 1792; i++)
    {
        if (policy[i] == 0.0)
        {
            continue;
        }
        
        if (board_metadata[5])
        {
            root->legal_from[root->num_legal_actions] = white_policy_idx_to_moves_from[i];
            root->legal_to[root->num_legal_actions] = white_policy_idx_to_moves_to[i];
        }
        else
        {
            root->legal_from[root->num_legal_actions] = black_policy_idx_to_moves_from[i];
            root->legal_to[root->num_legal_actions] = black_policy_idx_to_moves_to[i];
        }

        /* Check if it is promoting to knight */
        bool is_promotion_to_knight = false;
        for (int16 j = 0; j < 22; j++)
        {
            if (i == possible_knight_promotion_idx[j])
            {
                if (board_metadata[5] && white_board[root->legal_from[root->num_legal_actions]] == PIECE_WHITE_PAWN)
                {
                    root->legal_promotion[root->num_legal_actions] = 4;
                    is_promotion_to_knight = true;
                    break;
                }
                else if (!board_metadata[5] && white_board[root->legal_from[root->num_legal_actions]] == PIECE_BLACK_PAWN)
                {
                    root->legal_promotion[root->num_legal_actions] = 4;
                    is_promotion_to_knight = true;
                    break;
                }
            }
        }
        if (!is_promotion_to_knight)
        {
            root->legal_promotion[root->num_legal_actions] = 0;
        }
        
        root->legal_actions_prior[root->num_legal_actions] = policy[i];
        root->num_legal_actions++;
        if (root->num_legal_actions >= MCTS_MAX_CHILDREN)
        {
            fprintf(stderr, "Error: Legal actions larger than MCTS_MAX_CHILDREN in initialize_root()\n");
            return;
        }
    }
    for (int16 i = 1792; i < 1858; i++)
    {
        if (policy[i] == 0.0)
        {
            continue;
        }
        
        if (board_metadata[5])
        {
            root->legal_from[root->num_legal_actions] = white_policy_idx_to_moves_from[i];
            root->legal_to[root->num_legal_actions] = white_policy_idx_to_moves_to[i];
        }
        else
        {
            root->legal_from[root->num_legal_actions] = black_policy_idx_to_moves_from[i];
            root->legal_to[root->num_legal_actions] = black_policy_idx_to_moves_to[i];
        }
        root->legal_promotion[root->num_legal_actions] = ((i - 1792) % 3) + 1;
        root->legal_actions_prior[root->num_legal_actions] = policy[i];
        root->num_legal_actions++;
    
        if (root->num_legal_actions >= MCTS_MAX_CHILDREN)
        {
            fprintf(stderr, "Error: Legal actions larger than MCTS_MAX_CHILDREN in initialize_root()\n");
            return;
        }
    }

    for (int16 i = 0; i < MCTS_MAX_CHILDREN; i++)
    {
        root->children[i] = NULL;
    }

    root->num_children = 0;
    root->num_descents = 1;

    root->num_descents_virtual_loss = 0.0;
    root->value_virtual_loss = 0.0;
}

void reset_virtual_loss(Node *restrict node_pool, int32 num_nodes)
{
    for (int32 i = 0; i < num_nodes; i++)
    {
        node_pool[i].num_descents_virtual_loss = 0.0;
        // node_pool[i].value_virtual_loss = 0.0;
    }
}

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
)
{
    int32 current_batch_idx = 0;
    int32 i = 0;
    while ((i < batch_early_stop) && (current_batch_idx < batch_size) && (*num_current_nodes < num_total_nodes))
    {
        bool is_new_node = batch_PUCT(node_pool, &(batch_nodes[current_batch_idx]), batch_early_stop, &i, num_current_nodes, c_puct, c_fpu, virtual_loss);
        if (is_new_node)
        {
            if (batch_nodes[current_batch_idx]->board_metadata[5])
            {
                memcpy(batched_encoded_boards + current_batch_idx * 64 * 112, batch_nodes[current_batch_idx]->encoded_white_board, 64 * 112 * sizeof(float));
            }
            else
            {
                memcpy(batched_encoded_boards + current_batch_idx * 64 * 112, batch_nodes[current_batch_idx]->encoded_black_board, 64 * 112 * sizeof(float));
            }
            current_batch_idx++;
        }
    }

    *num_batched_nodes = current_batch_idx;
}

void put_batch(
    Node **restrict batch_nodes,
    int32 num_batched_nodes,
    float *restrict policy,
    float *restrict wdl
)
{
    for (int32 i = 0; i < num_batched_nodes; i++)
    {
        batch_nodes[i]->num_legal_actions = 0;
        for (int16 j = 0; j < 1792; j++)
        {
            if (policy[i * 1858 + j] != 0.0)
            {
                if (batch_nodes[i]->board_metadata[5])
                {
                    batch_nodes[i]->legal_from[batch_nodes[i]->num_legal_actions] = white_policy_idx_to_moves_from[j];
                    batch_nodes[i]->legal_to[batch_nodes[i]->num_legal_actions] = white_policy_idx_to_moves_to[j];
                }
                else
                {
                    batch_nodes[i]->legal_from[batch_nodes[i]->num_legal_actions] = black_policy_idx_to_moves_from[j];
                    batch_nodes[i]->legal_to[batch_nodes[i]->num_legal_actions] = black_policy_idx_to_moves_to[j];
                }

                /* Check if it is promoting to knight */
                bool is_promotion_to_knight = false;
                for (int16 k = 0; k < 22; k++)
                {
                    if (j == possible_knight_promotion_idx[k])
                    {
                        if (batch_nodes[i]->board_metadata[5] && batch_nodes[i]->white_board[batch_nodes[i]->legal_from[batch_nodes[i]->num_legal_actions]] == PIECE_WHITE_PAWN)
                        {
                            batch_nodes[i]->legal_promotion[batch_nodes[i]->num_legal_actions] = 4;
                            is_promotion_to_knight = true;
                            break;
                        }
                        else if (!batch_nodes[i]->board_metadata[5] && batch_nodes[i]->white_board[batch_nodes[i]->legal_from[batch_nodes[i]->num_legal_actions]] == PIECE_BLACK_PAWN)
                        {
                            batch_nodes[i]->legal_promotion[batch_nodes[i]->num_legal_actions] = 4;
                            is_promotion_to_knight = true;
                            break;
                        }
                    }
                }
                if (!is_promotion_to_knight)
                {
                    batch_nodes[i]->legal_promotion[batch_nodes[i]->num_legal_actions] = 0;
                }
                
                batch_nodes[i]->legal_actions_prior[batch_nodes[i]->num_legal_actions] = policy[i * 1858 + j];
                batch_nodes[i]->num_legal_actions++;
            }
        }
        for (int16 j = 1792; j < 1858; j++)
        {
            if (policy[i * 1858 + j] != 0.0)
            {
                if (batch_nodes[i]->board_metadata[5])
                {
                    batch_nodes[i]->legal_from[batch_nodes[i]->num_legal_actions] = white_policy_idx_to_moves_from[j];
                    batch_nodes[i]->legal_to[batch_nodes[i]->num_legal_actions] = white_policy_idx_to_moves_to[j];
                }
                else
                {
                    batch_nodes[i]->legal_from[batch_nodes[i]->num_legal_actions] = black_policy_idx_to_moves_from[j];
                    batch_nodes[i]->legal_to[batch_nodes[i]->num_legal_actions] = black_policy_idx_to_moves_to[j];
                }
                batch_nodes[i]->legal_promotion[batch_nodes[i]->num_legal_actions] = ((j - 1792) % 3) + 1;
            }
        }

        back_propagate(batch_nodes[i], wdl[i * 3 + 0], wdl[i * 3 + 1], wdl[i * 3 + 2]);
    }
}

void back_propagate_virtual_loss(Node *restrict node, float virtual_loss)
{
    while (node != NULL)
    {
        node->num_descents_virtual_loss += virtual_loss;
        node = node->parent;
    }
}

void back_propagate(Node *restrict node, float w, float d, float l)
{
    while (node != NULL)
    {
        node->wdl[0] += w;
        node->wdl[1] += d;
        node->wdl[2] += l;
        node->num_descents += 1;
        node = node->parent;

        float temp = w;
        w = l;
        l = temp;
    }
}

bool batch_PUCT(
    Node *node_pool,
    Node **batch_node_ptr,
    int32 batch_early_stop,
    int32 *restrict batch_num_iterations,
    int32 *restrict num_current_nodes,
    float c_puct,
    float c_fpu,
    float virtual_loss
)
{
    bool is_new_node = false;
    Node *current_node = &(node_pool[0]);

    while (*batch_num_iterations < batch_early_stop)
    {
        /* PUCT search */    
        float best_puct = -INFINITY;
        int16 best_child = 0;
        for (int16 i = 0; i < current_node->num_legal_actions; i++)
        {
            float puct;
            /* Node not explored */
            if (current_node->children[i] == NULL)
            { 
                float v_c = (current_node->wdl[0] - current_node->wdl[2] + current_node->value_virtual_loss) / current_node->num_descents - c_fpu * sqrtf(current_node->total_policy_explored_children);
                puct = v_c + c_puct * current_node->legal_actions_prior[i] * sqrtf(current_node->num_descents);
            }

            /* Node not explored but already inside the batch */
            else if (current_node->children[i]->num_legal_actions == -1)
            {
                float v_c = (current_node->wdl[0] - current_node->wdl[2] + current_node->value_virtual_loss) / current_node->num_descents - c_fpu * sqrtf(current_node->total_policy_explored_children);
                puct = v_c + c_puct * current_node->legal_actions_prior[i] * sqrtf(current_node->num_descents) / (1 + current_node->children[i]->num_descents_virtual_loss);
            }

            /* Explored nodes */
            else
            {
                float v_c = (current_node->children[i]->wdl[2] - current_node->children[i]->wdl[0] + current_node->children[i]->value_virtual_loss) / current_node->children[i]->num_descents;
                puct = v_c + c_puct * current_node->legal_actions_prior[i] * sqrtf(current_node->num_descents) / (1 + current_node->children[i]->num_descents + current_node->children[i]->num_descents_virtual_loss);
            }

            if (puct > best_puct)
            {
                best_puct = puct;
                best_child = i;
            }
        }

        /* Expand */
        if (current_node->children[best_child] == NULL)
        {
            is_new_node = true;

            Node *new_node = &(node_pool[*num_current_nodes]);
            new_node->parent = current_node;
            new_node->wdl[0] = 0.0;
            new_node->wdl[1] = 0.0;
            new_node->wdl[2] = 0.0;

            memcpy(new_node->white_board, current_node->white_board, 64 * sizeof(int8));
            memcpy(new_node->board_metadata, current_node->board_metadata, 6 * sizeof(bool));
            new_node->num_half_moves = current_node->num_half_moves;

            memcpy(new_node->previous_white_board_one_ply, current_node->white_board, 64 * sizeof(int8));
            memcpy(new_node->previous_white_board_two_ply, current_node->previous_white_board_one_ply, 64 * sizeof(int8));
            memcpy(new_node->previous_white_board_three_ply, current_node->previous_white_board_two_ply, 64 * sizeof(int8));
            memcpy(new_node->previous_white_board_four_ply, current_node->previous_white_board_three_ply, 64 * sizeof(int8));
            
            make_move(
                new_node->white_board,
                new_node->board_metadata,
                &(new_node->en_passant),
                &(new_node->num_half_moves),
                current_node->legal_from[best_child],
                current_node->legal_to[best_child],
                current_node->legal_promotion[best_child]
            );

            /* Check for repetition */
            new_node->board_metadata[4] = (
                memcmp(new_node->white_board, current_node->previous_white_board_three_ply, 64 * sizeof(int8)) == 0
            );
            
            encode_boards(
                current_node->encoded_white_board,
                current_node->encoded_black_board,
                new_node->encoded_white_board,
                new_node->encoded_black_board,
                new_node->white_board,
                new_node->board_metadata,
                new_node->num_half_moves
            );

            new_node->num_legal_actions = -1;

            new_node->total_policy_explored_children = 0.0;
            new_node->num_children = 0;
            for (int16 j = 0; j < MCTS_MAX_CHILDREN; j++)
            {
                new_node->children[j] = NULL;
            }

            new_node->num_descents = 0;
            new_node->num_descents_virtual_loss = 0.0;
            new_node->value_virtual_loss = 0.0;

            current_node->children[best_child] = new_node;
            current_node->children_idx[current_node->num_children] = best_child;
            current_node->num_children += 1;
            current_node->total_policy_explored_children += current_node->legal_actions_prior[best_child];

            current_node = new_node;

            *num_current_nodes += 1;
        }
        else
        {
            current_node = current_node->children[best_child];

            /* Node already in the batch */
            if (current_node->num_legal_actions == -1)
            {
                *batch_num_iterations += 1;
                back_propagate_virtual_loss(current_node, virtual_loss);
                return false;
            }

            // If no legal actions, it means that it is a leaf node (Either checkmate or stalemate)
            if (current_node->num_legal_actions == 0)
            {
                if (current_node->wdl[1] > 0.0)
                {
                    *batch_num_iterations += 1;
                    back_propagate(current_node, 0.0, 1.0, 0.0);
                }
                else
                {
                    *batch_num_iterations += 1;
                    back_propagate(current_node, 0.0, 0.0, 1.0);
                }

                return false;
            }

            is_new_node = false;
        }

        if (is_new_node)
        {
            int8 terminal_check = is_terminal(
                current_node->white_board, current_node->board_metadata, current_node->en_passant
            );

            /* Stalemate */
            if (terminal_check == 0)
            {
                current_node->num_legal_actions = 0;
                *batch_num_iterations += 1;
                back_propagate(current_node, 0.0, 1.0, 0.0);
                return false;
            }
            /* Checkmate */
            if (terminal_check == 1)
            {
                current_node->num_legal_actions = 0;
                *batch_num_iterations += 1;
                back_propagate(current_node, 0.0, 0.0, 1.0);
                return false;
            }    

            /* Check for repetition */
            if (current_node->board_metadata[4])
            {
                current_node->num_legal_actions = 0;
                *batch_num_iterations += 1;
                back_propagate(current_node, 0.0, 1.0, 0.0);
                return false;
            }

            /* Check for 50 moves */
            if (current_node->num_half_moves >= 100)
            {
                current_node->num_legal_actions = 0;
                *batch_num_iterations += 1;
                back_propagate(current_node, 0.0, 1.0, 0.0);
                return false;
            }
                
            /* node need to be evaluated by NN */
            *batch_node_ptr = current_node;
            *batch_num_iterations += 1;
            back_propagate_virtual_loss(current_node, virtual_loss);
            return true;
        }
    }

    return false;
}

void mcts_batches_mask_illegal_and_softmax_policy(
    float *restrict policy,
    Node **batch_nodes,
    int32 num_batched_nodes
)
{
    for (int32 i = 0; i < num_batched_nodes; i++)
    {
        mask_illegal_and_softmax_policy(
            policy + i * 1858,
            batch_nodes[i]->white_board,
            batch_nodes[i]->board_metadata,
            batch_nodes[i]->en_passant
        );
    }
}

void get_best_move_and_wdl(
    Node *root,
    int8 *restrict from,
    int8 *restrict to,
    int8 *restrict promotion,
    float *restrict w,
    float *restrict d,
    float *restrict l
)
{
    if (root->num_children == 0)
    {
        int16 best_child = 0;
        for (int16 i = 0; i < root->num_legal_actions; i++)
        {
            if (root->legal_actions_prior[i] > root->legal_actions_prior[best_child])
            {
                best_child = i;
            }
        }
        *from = root->legal_from[best_child];
        *to = root->legal_to[best_child];
        *promotion = root->legal_promotion[best_child];
        *w = root->wdl[0]; 
        *d = root->wdl[1];
        *l = root->wdl[2];
        return;
    }
    
    int32 most_visits = 0;
    int16 best_child = 0;
    for (int16 i = 0; i < root->num_children; i++)
    {
        if (root->children[root->children_idx[i]]->num_descents > most_visits)
        {
            best_child = root->children_idx[i];
            most_visits = root->children[best_child]->num_descents;
        }
        // printf("Prior: %f, Num_visits: %d, value: %f, from: %d, to: %d, promotion: %d\n",
        //     root->legal_actions_prior[root->children_idx[i]], 
        //     root->children[root->children_idx[i]]->num_descents, 
        //     (
        //         root->children[root->children_idx[i]]->wdl[2] - root->children[root->children_idx[i]]->wdl[0]
        //     ) / root->children[root->children_idx[i]]->num_descents,
        //     root->legal_from[root->children_idx[i]],
        //     root->legal_to[root->children_idx[i]],
        //     root->legal_promotion[root->children_idx[i]]
        // );
    }

    *from = root->legal_from[best_child];
    *to = root->legal_to[best_child];
    *promotion = root->legal_promotion[best_child];

    // Win of children = loss of parent
    *l = root->children[best_child]->wdl[0] / root->children[best_child]->num_descents;
    *d = root->children[best_child]->wdl[1] / root->children[best_child]->num_descents;
    *w = root->children[best_child]->wdl[2] / root->children[best_child]->num_descents;
}

int16 get_tree_depth(Node *root)
{
    int16 max_depth = 0;
    int16 most_visits;
    int16 best_child;
    Node *current_node = root;
    while (true)
    {
        most_visits = 0;
        best_child = 0;
        if (current_node->num_children == 0)
        {
            break;
        }

        for (int16 i = 0; i < current_node->num_children; i++)
        {
            if (current_node->children[current_node->children_idx[i]]->num_descents > most_visits)
            {
                best_child = current_node->children_idx[i];
                most_visits = current_node->children[best_child]->num_descents;
            }
        }

        current_node = current_node->children[best_child];

        max_depth++;
    }

    return max_depth;
}

void print_tree_info(
    Node *root,
    int16 max_depth,
    int32 num_nodes,
    int32 num_current_nodes,
    float elapsed_time
)
{
    int32 most_visits;
    int16 best_child;
    Node *current_node = root;
    bool depth_zero_flag = true;

    const char *board_value_to_pos[] = {
        "a8", "b8", "c8", "d8", "e8", "f8", "g8", "h8",
        "a7", "b7", "c7", "d7", "e7", "f7", "g7", "h7",
        "a6", "b6", "c6", "d6", "e6", "f6", "g6", "h6",
        "a5", "b5", "c5", "d5", "e5", "f5", "g5", "h5",
        "a4", "b4", "c4", "d4", "e4", "f4", "g4", "h4",
        "a3", "b3", "c3", "d3", "e3", "f3", "g3", "h3",
        "a2", "b2", "c2", "d2", "e2", "f2", "g2", "h2",
        "a1", "b1", "c1", "d1", "e1", "f1", "g1", "h1"
    };
    printf("info depth %d ", max_depth);
    while (true)
    {
        most_visits = 0;
        best_child = 0;
        if (current_node->num_children == 0)
        {
            break;
        }

        for (int16 i = 0; i < current_node->num_children; i++)
        {
            if (current_node->children[current_node->children_idx[i]]->num_descents > most_visits)
            {
                best_child = current_node->children_idx[i];
                most_visits = current_node->children[best_child]->num_descents;
            }
        }

        if (depth_zero_flag)
        {
            float wdl[3];
            wdl[0] = current_node->children[best_child]->wdl[2] / current_node->children[best_child]->num_descents;
            wdl[1] = current_node->children[best_child]->wdl[1] / current_node->children[best_child]->num_descents;
            wdl[2] = current_node->children[best_child]->wdl[0] / current_node->children[best_child]->num_descents;
            printf("wdl %.0f %.0f %.0f ", wdl[0] * 1000.0, wdl[1] * 1000.0, wdl[2] * 1000.0);
            printf("nps %d ", (int) ((float) num_nodes / elapsed_time));
            printf("nodes %d ", num_current_nodes);
            printf("time %d ", (int) (elapsed_time * 1000.0));
            printf("pv ");
            depth_zero_flag = false;
        }
        int8 from = current_node->legal_from[best_child];
        int8 to = current_node->legal_to[best_child];
        int8 promotion = current_node->legal_promotion[best_child];

        if (promotion == 0)
        {
            printf("%s%s ", board_value_to_pos[from], board_value_to_pos[to]);
        }
        else
        {
            printf("%s%s%c ", board_value_to_pos[from], board_value_to_pos[to], "nbrq"[promotion - 1]);
        }
        current_node = current_node->children[best_child];
    }

    printf("\n");
}