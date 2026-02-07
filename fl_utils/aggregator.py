import copy
import math
import sys

from sklearn.decomposition import PCA

sys.path.append("../")
import torch
import numpy as np
from torch.nn.utils import vector_to_parameters, parameters_to_vector


class Aggregator:
    def __init__(self, helper):
        self.helper = helper
        self.Wt = None
        self.krum_client_ids = []
        self.W = {}
        # 错误反馈向量
        for name, data in self.helper.global_model.state_dict().items():
            self.W[name] = torch.zeros_like(data, dtype=torch.float32)

    def flatten_update(self, update_dict):
        return torch.cat([v.view(-1).float().cpu() for v in update_dict.values()])

    def h(self, theta_t, scaled_update):
        # 用 PCA 的第一个分量长度作为拓扑特征的代理（你也可以用 persistent entropy / bottleneck distance 等）
        pca = PCA(n_components=1)
        vec = scaled_update.numpy().reshape(1, -1)
        pca.fit(vec)
        return pca.explained_variance_[0]  # 返回方差作为拓扑特征值

    def selective_aggregation(self, global_model, weight_accumulator_by_client, clean_ids, all_ids):
        model_state = global_model.state_dict()
        agg_result = {k: torch.zeros_like(v) for k, v in model_state.items()}
        num_clients = len(all_ids)

        # 计算每个参数在所有客户端中的平均变化幅度
        avg_abs_update = {}
        for name in model_state:
            stacked_updates = torch.stack([weight_accumulator_by_client[i][name].abs().float() for i in range(num_clients)])
            avg_abs_update[name] = stacked_updates.mean(dim=0)

        # 为每一层生成 mask，表示哪些参数是变化最小的 topk%
        importance_mask = {}
        for name, avg_update in avg_abs_update.items():
            flat = avg_update.flatten()
            k = max(1, int(0.1 * flat.numel()))
            topk_indices = torch.topk(flat, k, largest=False).indices  # 最小的 k 个
            mask = torch.zeros_like(flat, dtype=torch.bool)
            mask[topk_indices] = True
            importance_mask[name] = mask.view_as(avg_update)

        # 聚合更新
        for i, client_id in enumerate(all_ids):
            update = weight_accumulator_by_client[i]
            is_clean = client_id in clean_ids

            for name, delta in update.items():
                if is_clean:
                    # 干净客户端：全参数聚合
                    agg_result[name] += delta
                else:
                    # 异常客户端：只聚合变化最小的 20%
                    mask = importance_mask[name]
                    agg_result[name] += delta * mask  # 非 topk 区域为 0

        # 平均化聚合结果
        for name in agg_result:
            agg_result[name] = agg_result[name].to(dtype=torch.float32) / num_clients

        return agg_result

    def agg(self, global_model, weight_accumulator, weight_accumulator_by_client, client_models, sampled_participants, epoch):
        if self.helper.config.agg_method == 'avg':
            return self.average_shrink_models(global_model, weight_accumulator)
        elif self.helper.config.agg_method == 'clip':
            self.clip_updates(weight_accumulator)
            return self.average_shrink_models(global_model, weight_accumulator)
        elif self.helper.config.agg_method == 'robustLR':
            return self.robustLR(weight_accumulator_by_client, global_model, sampled_participants, weight_accumulator)
        elif self.helper.config.agg_method == 'CRFL':
            return self.CRFL(global_model,weight_accumulator)
        elif self.helper.config.agg_method == 'SparseFed':
            return self.sparsefed_update(weight_accumulator, global_model)
        else:
            return self.agg_avg(weight_accumulator_by_client,sampled_participants,global_model)

    def average_shrink_models(self, global_model, weight_accumulator):
        """
        Perform FedAvg algorithm and perform some clustering on top of it.
        """
        lr = 1

        for name, data in global_model.state_dict().items():
            if name == 'decoder.weight':
                continue
            update_per_layer = weight_accumulator[name] * \
                               (1 / self.helper.config.num_sampled_participants) * lr
            update_per_layer = update_per_layer.clone().detach().to(dtype=data.dtype)
            data.add_(update_per_layer.cuda())

        return True

    def clip_updates(self, agent_updates_dict):
        for key in agent_updates_dict:
            if 'num_batches_tracked' not in key:
                update = agent_updates_dict[key]
                l2_update = torch.norm(update, p=2)
                update.div_(max(1, l2_update / self.helper.config.clip_factor))
        return

    def krum_updates(self, weight_accumulator_by_client, sampled_participants):
        n_clients = len(sampled_participants)
        m = n_clients - self.helper.config.krum_k - 2
        distances = []

        # 计算客户端更新之间的成对距离
        for i in range(n_clients):
            client_i_update = weight_accumulator_by_client[i]
            dists = []
            for j in range(n_clients):
                if i != j:
                    client_j_update = weight_accumulator_by_client[j]
                    dist = 0
                    for key in client_i_update:
                        # 确保数据类型为浮点型
                        client_i_update[key] = client_i_update[key].float()
                        client_j_update[key] = client_j_update[key].float()
                        dist += torch.norm(client_i_update[key] - client_j_update[key]) ** 2
                    dists.append(dist)
            dists.sort()
            distances.append((i, sum(dists[:m])))  # 计算最近的'm'个距离的和

        # 选择具有最小距离和的客户端
        krum_client_idx = min(distances, key=lambda x: x[1])[0]
        krum_client_id = sampled_participants[krum_client_idx]
        self.krum_client_ids.append(krum_client_id)

        # 打印选择的客户端及其更新以调试
        print(f"Selected Krum Client ID: {krum_client_id}")

        # 使用选定客户端的更新作为全局更新
        krum_client_update = weight_accumulator_by_client[krum_client_idx]
        for key in krum_client_update:
            weight_accumulator_by_client[krum_client_idx][key].copy_(krum_client_update[key])

        return weight_accumulator_by_client[krum_client_idx]

    def robustLR(self, weight_accumulator_by_client, global_model, sampled_participants, weight_accumulator):
        # 计算每个参数的调整学习率
        lr_adjustment_vector = self.compute_robustLR(weight_accumulator_by_client, global_model)

        # 逐层更新模型参数
        for name, data in global_model.state_dict().items():
            if name == 'decoder.weight':
                continue
            data = data.float()
            update_per_layer = weight_accumulator[name] * lr_adjustment_vector[name] / len(sampled_participants)

            data.add_(update_per_layer)

        return

    def compute_robustLR(self, weight_accumulator_by_client, global_model):
        lr_adjustment_dict = {}

        # 遍历每一层的更新
        for key, update in weight_accumulator_by_client[0].items():
            # 收集该层中所有客户端的参数更新
            stacked_updates = torch.stack([client_update[key] for client_update in weight_accumulator_by_client])

            # 计算每个参数的位置上的符号和
            agent_updates_sign = torch.sign(stacked_updates)
            update_sum = torch.sum(agent_updates_sign, dim=0)  # 对客户端维度求和

            # 计算每个参数的绝对值
            value = abs(update_sum)

            # 根据每个参数的位置调整学习率
            lr_adjustment = torch.where(value < 2, -self.helper.config.server_lr, self.helper.config.server_lr)

            # 确保调整值的形状与该层的参数一致
            lr_adjustment_dict[key] = lr_adjustment.to()

        return lr_adjustment_dict

    def agg_avg(self, weight_accumulator_by_client, sampled_participants, global_model):
        """
        Standard FedAvg-style aggregation using per-client updates
        """
        n_clients = len(sampled_participants)

        # 1. 初始化聚合更新
        agg_update = {
            key: torch.zeros_like(weight_accumulator_by_client[0][key], dtype=torch.float32)
            for key in weight_accumulator_by_client[0].keys()
            if key != 'decoder.weight'
        }

        # 2. 累加客户端更新
        for i in range(n_clients):
            for key in agg_update:
                agg_update[key] += weight_accumulator_by_client[i][key].float()

        # 3. 平均
        for key in agg_update:
            agg_update[key] /= n_clients

        # 4. 应用到 global_model
        with torch.no_grad():
            for name, param in global_model.state_dict().items():
                if name == 'decoder.weight':
                    continue
                param.add_(agg_update[name].to(param.device).to(param.dtype))

        return True

    def agg_late_guard(
        self,
        global_model,
        weight_accumulator_by_client,
        sampled_participants,
        suspicious_ids,
        mask_by_suspicious_id=None,
        drop_suspicious=False,
        drop_ids=None,
    ):
        """
        LATE-Guard aggregation:
        - suspicious_ids: clients whose update should be masked (soft defense)
        - drop_ids: subset of clients whose whole update should be dropped when drop_suspicious=True (hard defense)
        IMPORTANT: even in hard_drop mode, we still mask the remaining suspicious clients that are not dropped.
        """
        device = next(global_model.parameters()).device
        model_state = global_model.state_dict()

        susp_set = set(suspicious_ids or [])
        drop_set = set(drop_ids or [])

        # ---- build accumulator only for floating-point entries ----
        agg_update = {}
        for name, t in model_state.items():
            if name == "decoder.weight":
                continue
            if torch.is_floating_point(t):
                agg_update[name] = torch.zeros_like(t, device=device, dtype=torch.float32)

        used = 0  # number of updates actually aggregated

        for i, cid in enumerate(sampled_participants):
            is_susp = cid in susp_set
            is_drop = cid in drop_set

            if drop_suspicious and is_drop:
                # hard defense: drop only the chosen subset
                continue

            upd = weight_accumulator_by_client[i]
            used += 1

            for name, buf in agg_update.items():
                if name not in upd:
                    continue

                delta0 = upd[name]
                if not torch.is_floating_point(delta0):
                    continue

                delta = delta0.to(device=device, dtype=torch.float32)

                # masked defense: always apply to suspicious (even when hard_drop is on),
                # except those already dropped above.
                if is_susp and mask_by_suspicious_id is not None:
                    if cid in mask_by_suspicious_id and name in mask_by_suspicious_id[cid]:
                        m = mask_by_suspicious_id[cid][name].to(device=device)
                        if m.dtype != torch.bool:
                            m = m != 0
                        delta = delta * m.to(dtype=delta.dtype)

                buf.add_(delta)

        denom = max(1, used)
        for name in agg_update:
            agg_update[name].div_(denom)

        # apply update to global model
        with torch.no_grad():
            sd = global_model.state_dict()
            for name, upd in agg_update.items():
                if name not in sd:
                    continue
                sd[name].add_(upd.to(dtype=sd[name].dtype, device=sd[name].device))
            global_model.load_state_dict(sd, strict=False)

        return True

    def CRFL(self, global_model, weight_accumulator):
        self.clip_updates(weight_accumulator)
        self.average_shrink_models(global_model, weight_accumulator)
        for name, param in global_model.state_dict().items():
            param = param.float()
            param.add_(self.dp_noise(param, self.helper.config.sigma_param))
        return True
    def dp_noise(self, param, sigma):

        noise = torch.normal(0, sigma, size=param.shape, device=param.device)
        return noise

    def sparsefed_update(self, weight_accumulator, target_model):
        """实现 SparseFed 更新"""

        k = 0.95
        for name, data in target_model.state_dict().items():
            if name == 'decoder.weight':
                continue
            weight_accumulator[name] = weight_accumulator[name] * \
                               (1 / self.helper.config.num_sampled_participants)

        # 错误反馈
        for name in weight_accumulator:
            self.W[name] += weight_accumulator[name]

        # 提取前k个坐标作为稀疏更新
        gradient = {}
        for name, w in weight_accumulator.items():
            flat_w = w.view(-1).abs()
            _, indices = torch.topk(flat_w, int(k * len(flat_w)), largest=True)
            mask = torch.zeros_like(flat_w)
            mask[indices] = 1
            gradient[name] = w * mask.view_as(w)

        # 累积剩余的误差
        for name in gradient:
            self.W[name] -= gradient[name]


        for name, data in target_model.state_dict().items():
            if name == 'decoder.weight':
                continue
            update_per_layer = gradient[name]
            update_per_layer = update_per_layer.clone().detach().to(dtype=data.dtype)
            data.add_(update_per_layer.cuda())

        return target_model
