import torch
from torch import nn, Tensor
import numpy as np
from torch.nn.init import normal_ as normal_init
import torch.nn.functional as F

from config import Config


class ScaledDotProductAttention(nn.Module):
    def __init__(self, embed_dim: int) -> None:
        super(ScaledDotProductAttention, self).__init__()
        self.embed_dim = embed_dim

    def forward(self, Q: Tensor, K: Tensor, V: Tensor, Q_position: Tensor = None, K_position: Tensor = None,
                V_position: Tensor = None) -> Tensor:
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.embed_dim)
        if Q_position is not None:
            pos_sim = torch.matmul(Q_position, K_position.transpose(-1, -2)) / np.sqrt(self.embed_dim)
            scores += pos_sim
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        return context


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int) -> None:
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.W_Q = nn.Linear(self.embed_dim, self.embed_dim * self.num_heads)
        self.W_K = nn.Linear(self.embed_dim, self.embed_dim * self.num_heads)
        self.W_V = nn.Linear(self.embed_dim, self.embed_dim * self.num_heads)
        self.W_QP = nn.Linear(self.embed_dim, self.embed_dim * self.num_heads)
        self.W_KP = nn.Linear(self.embed_dim, self.embed_dim * self.num_heads)
        self.W_VP = nn.Linear(self.embed_dim, self.embed_dim * self.num_heads)
        self.q_linear = nn.Linear(self.num_heads * self.embed_dim, self.embed_dim)
        self.v_linear = nn.Linear(self.num_heads * self.embed_dim, self.embed_dim)
        self.layer_norm = nn.LayerNorm(self.embed_dim)

        self.init_weight()

    def init_weight(self):
        normal_init(self.W_Q.weight, mean=0.0, std=0.01)
        normal_init(self.W_Q.bias, mean=0.0, std=0.01)
        normal_init(self.W_K.weight, mean=0.0, std=0.01)
        normal_init(self.W_K.bias, mean=0.0, std=0.01)
        normal_init(self.W_V.weight, mean=0.0, std=0.01)
        normal_init(self.W_V.bias, mean=0.0, std=0.01)

        normal_init(self.W_QP.weight, mean=0.0, std=0.01)
        normal_init(self.W_QP.bias, mean=0.0, std=0.01)
        normal_init(self.W_KP.weight, mean=0.0, std=0.01)
        normal_init(self.W_KP.bias, mean=0.0, std=0.01)
        normal_init(self.W_VP.weight, mean=0.0, std=0.01)
        normal_init(self.W_VP.bias, mean=0.0, std=0.01)

        normal_init(self.q_linear.weight, mean=0.0, std=0.01)
        normal_init(self.q_linear.bias, mean=0.0, std=0.01)
        normal_init(self.v_linear.weight, mean=0.0, std=0.01)
        normal_init(self.v_linear.bias, mean=0.0, std=0.01)

    def forward(self, Q: Tensor, K: Tensor, V: Tensor, Q_position: Tensor = None, K_position: Tensor = None,
                V_position: Tensor = None, x_flag: bool = False):
        if not x_flag:
            residual, batch_size = Q, Q.size(0)
            # Q = Q - K

            q_p_s = k_p_s = v_p_s = None
            if Q_position is not None and K_position is not None:
                if V_position is None:
                    V_position = K_position
                q_p_s = self.W_QP(Q_position).view(batch_size, -1, self.num_heads, self.embed_dim).transpose(1, 2)
                k_p_s = self.W_KP(K_position).view(batch_size, -1, self.num_heads, self.embed_dim).transpose(1, 2)
                v_p_s = self.W_VP(V_position).view(batch_size, -1, self.num_heads, self.embed_dim).transpose(1, 2)

            q_s = self.W_Q(Q).view(batch_size, -1, self.num_heads, self.embed_dim).transpose(1, 2)
            k_s = self.W_K(K).view(batch_size, -1, self.num_heads, self.embed_dim).transpose(1, 2)
            v_s = self.W_V(V).view(batch_size, -1, self.num_heads, self.embed_dim).transpose(1, 2)

            context = ScaledDotProductAttention(self.embed_dim)(q_s, k_s, v_s, q_p_s, k_p_s, v_p_s)
            context = context.transpose(1, 2).contiguous().view(batch_size, -1,
                                                                self.num_heads * self.embed_dim)
            v_s = v_s.transpose(1, 2).contiguous().view(batch_size, -1,
                                                        self.num_heads * self.embed_dim)
            output = self.q_linear(context)
            v_s = self.v_linear(v_s)
            return self.layer_norm(output + residual), v_s
        else:
            v_s = self.W_V(V).view(V.shape[0], -1, self.num_heads, self.embed_dim).transpose(1, 2)
            v_s = v_s.transpose(1, 2).contiguous().view(V.shape[0], -1,
                                                        self.num_heads * self.embed_dim)
            output = self.v_linear(v_s)
            output = output / output.shape[1]
            output = torch.split(output, 1, dim=1)[0]
            output = output.view(output.shape[0], output.shape[2])
            return output


class TiSANCR(torch.nn.Module):
    def __init__(self, n_users: int, n_items: int, n_time: int, config:Config) -> None:
        super(TiSANCR, self).__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.n_time = n_time
        self.config = config
        self.emb_size = self.config.emb_size
        self.dropout = self.config.dropout

        self.item_embeddings = torch.nn.Embedding(self.n_items, self.emb_size)
        self.user_embeddings = torch.nn.Embedding(self.n_users, self.emb_size)

        self.true_vector = torch.nn.Parameter(torch.from_numpy(
            np.random.uniform(0, 0.1, size=self.emb_size).astype(np.float32)),
            requires_grad=False)

        self.not_layer_1 = torch.nn.Linear(self.emb_size, self.emb_size)

        self.not_layer_2 = torch.nn.Linear(self.emb_size, self.emb_size)

        self.or_layer_1 = torch.nn.Linear(2 * self.emb_size, self.emb_size)

        self.or_layer_2 = torch.nn.Linear(self.emb_size, self.emb_size)

        self.and_layer_1 = torch.nn.Linear(2 * self.emb_size, self.emb_size)

        self.and_layer_2 = torch.nn.Linear(self.emb_size, self.emb_size)

        self.encoder_layer_1 = torch.nn.Linear(2 * self.emb_size, self.emb_size)

        self.encoder_layer_2 = torch.nn.Linear(self.emb_size, self.emb_size)

        self.dropout_layer = torch.nn.Dropout(self.dropout)

        self.attention_heads = self.config.attention_heads
        self.init_my()

        self.init_weights()

    def init_my(self) -> None:
        self.time_embeddings = nn.Embedding(self.n_time, self.emb_size)
        normal_init(self.time_embeddings.weight, mean=0.0, std=0.01)
        self.time_mean_embeddings = nn.Embedding(self.n_time, self.emb_size)
        normal_init(self.time_mean_embeddings.weight, mean=0.0, std=0.01)
        self.time_attention_layer = MultiHeadAttention(embed_dim=self.emb_size, num_heads=self.attention_heads)
        self.event_attention_layer = MultiHeadAttention(embed_dim=self.emb_size, num_heads=self.attention_heads)

        self.time_encoder_layer1 = nn.Linear(2 * self.emb_size, self.emb_size)
        self.time_encoder_layer2 = nn.Linear(self.emb_size, self.emb_size)
        normal_init(self.time_encoder_layer1.weight, mean=0.0, std=0.01)
        normal_init(self.time_encoder_layer1.bias, mean=0.0, std=0.01)
        normal_init(self.time_encoder_layer2.weight, mean=0.0, std=0.01)
        normal_init(self.time_encoder_layer2.bias, mean=0.0, std=0.01)

    def init_weights(self) -> None:
        normal_init(self.not_layer_1.weight, mean=0.0, std=0.01)
        normal_init(self.not_layer_1.bias, mean=0.0, std=0.01)
        normal_init(self.not_layer_2.weight, mean=0.0, std=0.01)
        normal_init(self.not_layer_2.bias, mean=0.0, std=0.01)

        normal_init(self.or_layer_1.weight, mean=0.0, std=0.01)
        normal_init(self.or_layer_1.bias, mean=0.0, std=0.01)
        normal_init(self.or_layer_2.weight, mean=0.0, std=0.01)
        normal_init(self.or_layer_2.bias, mean=0.0, std=0.01)

        normal_init(self.and_layer_1.weight, mean=0.0, std=0.01)
        normal_init(self.and_layer_1.bias, mean=0.0, std=0.01)
        normal_init(self.and_layer_2.weight, mean=0.0, std=0.01)
        normal_init(self.and_layer_2.bias, mean=0.0, std=0.01)

        normal_init(self.encoder_layer_1.weight, mean=0.0, std=0.01)
        normal_init(self.encoder_layer_1.bias, mean=0.0, std=0.01)
        normal_init(self.encoder_layer_2.weight, mean=0.0, std=0.01)
        normal_init(self.encoder_layer_2.bias, mean=0.0, std=0.01)

        normal_init(self.user_embeddings.weight, mean=0.0, std=0.01)
        normal_init(self.item_embeddings.weight, mean=0.0, std=0.01)

    def logic_not(self, vector: Tensor) -> Tensor:
        vector = F.relu(self.not_layer_1(vector))
        if self.training:
            vector = self.dropout_layer(vector)
        out = self.not_layer_2(vector)
        return out

    def logic_or(self, vector1: Tensor, vector2: Tensor, dim: int = 1) -> Tensor:
        vector = torch.cat((vector1, vector2), dim)
        vector = F.relu(self.or_layer_1(vector))
        if self.training:
            vector = self.dropout_layer(vector)
        out = self.or_layer_2(vector)
        return out

    def logic_and(self, vector1: Tensor, vector2: Tensor, dim: int = 1) -> Tensor:
        vector = torch.cat((vector1, vector2), dim)
        vector = F.relu(self.and_layer_1(vector))
        if self.training:
            vector = self.dropout_layer(vector)
        out = self.and_layer_2(vector)
        return out

    def encoder(self, ui_vector: Tensor) -> Tensor:
        event_vector = F.relu(self.encoder_layer_1(ui_vector))
        if self.training:
            event_vector = self.dropout_layer(event_vector)
        event_vector = self.encoder_layer_2(event_vector)
        return event_vector

    def time_encoder(self, et_vector: Tensor) -> Tensor:
        euvt = F.relu(self.time_encoder_layer1(et_vector))
        if self.training:
            euvt = self.dropout_layer(euvt)
        euvt = self.time_encoder_layer2(euvt)
        return euvt

    def forward(self, batch_data: tuple[Tensor]) -> tuple:
        user_ids, item_ids, histories, history_feedbacks, neg_item_ids, timestamp, history_timestamp = batch_data

        user_embs = self.user_embeddings(user_ids)
        item_embs = self.item_embeddings(item_ids)
        neg_item_embs = self.item_embeddings(neg_item_ids)

        right_side_events = self.encoder(torch.cat((user_embs, item_embs), dim=1))
        # timestamp_embs = self.time_embeddings(timestamp)
        # right_side_events += timestamp_embs

        left_side_events = user_embs.view(user_embs.size(0), 1, user_embs.size(1))
        left_side_events = left_side_events.expand(user_embs.size(0), histories.size(1), user_embs.size(1))
        history_item_embs = self.item_embeddings(histories)
        left_side_events = self.encoder(torch.cat((left_side_events, history_item_embs), dim=2))

        # TODO
        _timestamp = timestamp.view(timestamp.size(0), 1)
        _timestamp = _timestamp.expand(_timestamp.size(0), history_timestamp.size(1))
        relative_time = _timestamp - history_timestamp
        relative_time_embs = self.time_embeddings(relative_time)
        attn_out, _ = self.time_attention_layer(relative_time_embs, relative_time_embs,
                                                relative_time_embs)
        left_side_events += attn_out
        # left_side_events = self.time_encoder(torch.cat((left_side_events, attn_out), dim=2))
        lx = self.time_attention_layer(None, None, relative_time_embs, x_flag=True)
        right_side_events += lx
        # right_side_events = self.time_encoder(torch.cat((right_side_events, lx), dim=1))

        exp_user_embs = user_embs.view(user_embs.size(0), 1, user_embs.size(1))
        exp_user_embs = exp_user_embs.expand(user_embs.size(0), neg_item_embs.size(1),
                                             user_embs.size(1))
        right_side_neg_events = self.encoder(torch.cat((exp_user_embs, neg_item_embs), dim=2))

        # TODO
        lx = lx.view(lx.shape[0], 1, lx.shape[1])
        lx = lx.expand(right_side_neg_events.shape[0], right_side_neg_events.shape[1], self.emb_size)
        right_side_neg_events += lx
        # right_side_neg_events = self.time_encoder(torch.cat((right_side_neg_events, lx), dim=2))

        left_side_neg_events = self.logic_not(left_side_events)

        constraints = list([left_side_events])
        constraints.append(left_side_neg_events)

        feedback_tensor = history_feedbacks.view(history_feedbacks.size(0), history_feedbacks.size(1), 1)
        feedback_tensor = feedback_tensor.expand(history_feedbacks.size(0), history_feedbacks.size(1), self.emb_size)

        # double not
        left_side_events = feedback_tensor * left_side_events + (1 - feedback_tensor) * left_side_neg_events
        left_side_events = self.logic_not(left_side_events)

        tmp_vector = left_side_events[:, 0]

        shuffled_history_idx = list(range(1, histories.size(1)))
        np.random.shuffle(shuffled_history_idx)
        for i in shuffled_history_idx:
            tmp_vector = self.logic_or(tmp_vector, left_side_events[:, i])
            constraints.append(tmp_vector.view(histories.size(0), -1, self.emb_size))
        left_side_events = tmp_vector

        constraints.append(right_side_events.view(histories.size(0), -1, self.emb_size))

        constraints.append(right_side_neg_events)

        expression_events = self.logic_or(left_side_events, right_side_events)
        constraints.append(expression_events.view(histories.size(0), -1, self.emb_size))

        exp_left_side_events = left_side_events.view(left_side_events.size(0), 1, left_side_events.size(1))
        exp_left_side_events = exp_left_side_events.expand(left_side_events.size(0), right_side_neg_events.size(1),
                                                           left_side_events.size(1))

        expression_neg_events = self.logic_or(exp_left_side_events, right_side_neg_events, dim=2)
        constraints.append(expression_neg_events)

        positive_predictions = F.cosine_similarity(expression_events,
                                                   self.true_vector.view([1, -1])) * 10

        reshaped_expression_neg_events = expression_neg_events.reshape(expression_neg_events.size(0) *
                                                                       expression_neg_events.size(1),
                                                                       expression_neg_events.size(2))
        negative_predictions = F.cosine_similarity(reshaped_expression_neg_events, self.true_vector.view([1, -1])) * 10
        negative_predictions = negative_predictions.reshape(expression_neg_events.size(0),
                                                            expression_neg_events.size(1))

        constraints = torch.cat(constraints, dim=1)

        constraints = constraints.view(constraints.size(0) * constraints.size(1), constraints.size(2))

        return positive_predictions, negative_predictions, constraints
