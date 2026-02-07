import math
import sys

from mpmath import norm
from sklearn.covariance import MinCovDet

sys.path.append("../")
import time
import wandb

import torch
import torch.nn.functional as F

import random
import numpy as np
import copy
import os

from .attacker import Attacker
from .aggregator import Aggregator
from math import ceil
import pickle


class FLer:
    def __init__(self, helper):
        os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

        self.helper = helper
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.criterion = torch.nn.CrossEntropyLoss()
        self.cos_sim = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        self.attack_sum = 0
        self.aggregator = Aggregator(self.helper)
        self.start_time = time.time()
        self.attacker_criterion = torch.nn.CrossEntropyLoss()
        self.abnormal_scores = {}
        self.trigger_cache = None
        self.synthetic_data = None
        self.client_history = {i: [] for i in range(self.helper.config.num_total_participants)}

        if self.helper.config.is_poison:
            self.attacker = Attacker(self.helper)
        else:
            self.attacker = None

        if self.helper.config.sample_method == 'random_updates':
            self.init_advs()

        if self.helper.config.load_benign_model:
            model_path = f'../saved/benign_new/{self.helper.config.dataset}_{self.helper.config.poison_start_epoch}_{self.helper.config.agg_method}.pt'
            self.helper.global_model.load_state_dict(torch.load(model_path, map_location='cuda')['model'])

        return

            
    def init_advs(self):
        num_updates = self.helper.config.num_sampled_participants * self.helper.config.poison_epochs
        num_poison_updates = ceil(self.helper.config.sample_poison_ratio * num_updates)
        updates = list(range(num_updates))
        advs = np.random.choice(updates, num_poison_updates, replace=False)
        print(f'Using random updates, sampled {",".join([str(x) for x in advs])}')
        adv_dict = {}
        for adv in advs:
            epoch = (adv // self.helper.config.num_sampled_participants) + self.helper.config.poison_start_epoch
            idx = (adv % self.helper.config.num_sampled_participants) + self.helper.config.poison_start_epoch
            if epoch in adv_dict:
                adv_dict[epoch].append(idx)
            else:
                adv_dict[epoch] = [idx]
        self.advs = adv_dict

    def test_once(self, poison=False):
        model = self.helper.global_model
        model.eval()

        total_loss = 0.0
        correct = 0
        num_data = 0

        with torch.no_grad():
            data_source = self.helper.test_data

            if self.helper.config.agg_method == 'CRFL':
                disturbance_models = self.create_disturbance_models(model, n_models=5, noise_std=0.002)

            for batch_id, batch in enumerate(data_source):
                data, targets = batch
                data, targets = data.cuda(), targets.cuda()

                if poison:
                    if self.helper.config.attack_method == 'DBA':
                        poison_pattern = []
                        for i in range(0, self.helper.config.trigger_num):
                            poison_pattern_name = f"poison_pattern_{i}"
                            poison_pattern += getattr(self.helper, poison_pattern_name)
                        data, targets = self.attacker.DBA_poison_input(data, targets, poison_pattern, eval=True)
                    elif self.helper.config.attack_method == 'CerP':
                        data, targets = self.attacker.cerp_poison_input(data, targets, eval=True)
                    else:
                        data, targets = self.attacker.poison_input(data, targets, eval=True)

                if self.helper.config.agg_method == 'CRFL':
                    preds = []
                    for m in disturbance_models:
                        out_m = m(data)
                        pred_m = out_m.data.max(1)[1]
                        preds.append(pred_m.cpu().numpy())

                    majority_pred = self.majority_voting(preds)
                    correct += int((majority_pred == targets.cpu().numpy()).sum())
                    output = model(data)
                else:
                    output = model(data)
                    pred = output.data.max(1)[1]
                    correct += pred.eq(targets.data).sum().item()

                batch_loss = self.criterion(output, targets)
                total_loss += batch_loss.item() * targets.size(0)
                num_data += targets.size(0)

        acc = 100.0 * correct / num_data
        loss = total_loss / num_data
        model.train()
        return loss, acc


    def create_disturbance_models(self, global_model, n_models=5, noise_std=0.002):
        """根据高斯噪声生成多个扰动模型"""
        disturbance_models = []
        for _ in range(n_models):
            disturbance_model = self.add_gaussian_noise_to_model(global_model, noise_std)
            disturbance_models.append(disturbance_model)
        return disturbance_models

    def add_gaussian_noise_to_model(self, model, noise_std=0.002):
        """添加高斯噪声到模型参数"""
        noisy_model = copy.deepcopy(model)
        # 创建模型副本
        for param in noisy_model.parameters():
            noise = torch.randn_like(param) * noise_std
            param.data += noise
        return noisy_model

    def majority_voting(self, predictions):
        """基于多数投票机制返回最终的预测结果"""
        predictions = np.array(predictions)
        # 计算每个样本的多数投票结果
        majority_prediction = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=predictions)
        return majority_prediction

    def log_once(self, epoch, loss, acc, bkd_loss, bkd_acc):
        log_dict = {
            'epoch': epoch,
            'test_acc': acc,
            'test_loss': loss,
            'bkd_acc': bkd_acc,
            'bkd_loss': bkd_loss
        }
        print('|'.join([f'{k}:{float(log_dict[k]):.3f}' for k in log_dict]))
        self.save_model(epoch, log_dict)

    def save_model(self, epoch, log_dict):
        if epoch % self.helper.config.save_every == 0:
            log_dict['model'] = self.helper.global_model.state_dict()
            # if self.helper.config.is_poison:
            #     pass
            # else:
            assert self.helper.config.lr_method == 'linear'
            save_path = f'../saved/benign_new/{self.helper.config.dataset}_{epoch}_{self.helper.config.agg_method}.pt'
            torch.save(log_dict, save_path)
            print(f'Model saved at {save_path}')

    def save_res(self, accs, asrs):
        log_dict = {
            'accs': accs,
            'asrs': asrs
        }
        atk_method = self.helper.config.attacker_method
        if self.helper.config.sample_method == 'random':
            file_name = f'{self.helper.config.dataset}/{self.helper.config.agg_method}_{atk_method}_r_{self.helper.config.num_adversaries}_{self.helper.config.poison_epochs}_ts{self.helper.config.trigger_size}.pkl'
        else:
            raise NotImplementedError
        save_path = os.path.join(f'../saved/res/{file_name}')
        f_save = open(save_path, 'wb')
        pickle.dump(log_dict, f_save)
        f_save.close()
        print(f'results saved at {save_path}')
        
    def update_dynamic_params(self, epoch, total_epochs):
        # 指数衰减或线性策略可选
        eps_c = max(0.3, 1 * (1 - epoch / total_epochs))
        eps_s = max(0.3, 1 * (1 - epoch / total_epochs))
        decay_base = 0.0001
        min_threshold = 1e-20
        decay_rate = math.log(decay_base / min_threshold) / total_epochs
        importance_threshold = max(min_threshold, decay_base * math.exp(-decay_rate * epoch))

        return eps_c, eps_s, importance_threshold
    
    def _flatten_update(self, upd):
        return torch.cat([v.view(-1) for v in upd.values()])

    def score_update_alignment(self, upd, clean_grad_dict, topk_ratio=0.01, eps=1e-12):
        """
        返回:
          cos:  Δ 与 g 的 cosine
          l2:   ||Δ||
          r:    topk 能量集中度 = ||topk(|Δ|)|| / ||Δ||
        """
        vec_d = []
        vec_g = []
        for name, delta in upd.items():
            if name == "decoder.weight":
                continue
            if name not in clean_grad_dict:
                continue
            if not torch.is_floating_point(delta):
                continue
            d = delta.detach().float().view(-1).cpu()
            g = clean_grad_dict[name].detach().float().view(-1).cpu()
            vec_d.append(d)
            vec_g.append(g)

        if len(vec_d) == 0:
            return 0.0, 0.0, 0.0

        d = torch.cat(vec_d, dim=0)
        g = torch.cat(vec_g, dim=0)

        d_norm = torch.norm(d) + eps
        g_norm = torch.norm(g) + eps
        cos = float(torch.dot(d, g) / (d_norm * g_norm))

        # 能量集中度
        k = max(1, int(topk_ratio * d.numel()))
        topk = torch.topk(d.abs(), k, largest=True).values
        r = float(torch.norm(topk) / d_norm)

        return cos, float(d_norm), r

    def flag_by_gsa_metrics(self, sampled_participants, weight_accumulator_by_client, clean_g,
                            topk_ratio=0.01, cos_bad_thr=0.02, r_mad_k=3.0):
        """
        用 GSA 相关指标做“硬规则”可疑标记：
        1) cos 很小（接近 0）或为负：可能是伪装/反向更新
        2) topk-r 特别大：更新能量集中，常见于后门/异常梯度
        返回：flag_ids（set）
        """
        cos_list = []
        r_list = []
        for i, cid in enumerate(sampled_participants):
            cos, l2, r = self.score_update_alignment(weight_accumulator_by_client[i], clean_g, topk_ratio=topk_ratio)
            cos_list.append(cos)
            r_list.append(r)

        # r 的 robust 阈值：median + k * MAD
        r_arr = np.array(r_list, dtype=np.float32)
        r_med = float(np.median(r_arr))
        r_mad = float(np.median(np.abs(r_arr - r_med)) + 1e-12)
        r_thr = r_med + r_mad_k * 1.4826 * r_mad

        flag_ids = set()
        for i, cid in enumerate(sampled_participants):
            cos = float(cos_list[i])
            r = float(r_list[i])

            # 规则 1：cos 太小/为负（你可以把 0.02 调成 0.05 更狠）
            if cos < cos_bad_thr:
                flag_ids.add(cid)

            # 规则 2：topk-r 异常偏大
            if r > r_thr:
                flag_ids.add(cid)

        return flag_ids

    def train(self):
        print('Training')
        accs = []
        asrs = []
        poisonupdate_dict = dict()
        start_epoch = self.helper.config.poison_start_epoch if self.helper.config.load_benign_model else 0

        # ---------------- Trigger-based LATE-Guard state ----------------
        from collections import deque

        # 这些参数都支持在 config.yaml 里配置（不存在则用默认值）
        window = getattr(self.helper.config, "guard_window", 50)              # 滚动窗口长度
        warmup = getattr(self.helper.config, "guard_warmup", 50)              # warmup 期：只学习基线，不轻易 hard
        sus_rate_hi = getattr(self.helper.config, "guard_sus_rate_hi", 0.2)  # 可疑比例阈值（触发 alert）
        cos_med_thr = getattr(self.helper.config, "guard_cos_med_thr", -0.02) # median cosine 绝对阈值
        cos_drop_thr = getattr(self.helper.config, "guard_cos_drop_thr", 0.06)# 相对基线下降阈值

        calm_patience = getattr(self.helper.config, "guard_calm_patience", 10)  # 连续 calm N 轮退出 alert
        repair_steps = getattr(self.helper.config, "guard_repair_steps", 50)
        repair_every = getattr(self.helper.config, "guard_repair_every", 5)
        clip_mult = getattr(self.helper.config, "guard_clip_mult", 2.0)

        # detect 参数（你也可在 yaml 里配）
        eta = getattr(self.helper.config, "guard_eta", 0.2)
        topk_ratio = getattr(self.helper.config, "guard_topk_ratio", 0.01)
        sign_thr = getattr(self.helper.config, "guard_sign_thr", 0.75)
        q_normal = getattr(self.helper.config, "guard_q_normal", 0.6)
        q_alert  = getattr(self.helper.config, "guard_q_alert", 0.5)

        # clean-grad / probe batch
        clean_batches_normal = getattr(self.helper.config, "guard_clean_batches_normal", 2)
        clean_batches_alert  = getattr(self.helper.config, "guard_clean_batches_alert", 10)
        probe_batches_normal = getattr(self.helper.config, "guard_probe_batches_normal", 10)
        probe_batches_alert  = getattr(self.helper.config, "guard_probe_batches_alert", 20)

        # GSA 参数（建议保持温和，避免 benign 掉点）
        tau = getattr(self.helper.config, "tau", 0.1)
        gsa_alpha = getattr(self.helper.config, "gsa_alpha", getattr(self.helper.config, "alpha", 0.90))
        gsa_beta = getattr(self.helper.config, "gsa_beta", getattr(self.helper.config, "beta", 0.0))

        if not hasattr(self, "_late_guard_state"):
            self._late_guard_state = {
                "cos_hist": deque(maxlen=window),
                "sus_hist": deque(maxlen=window),
                "alert_level": 0,      # 0=normal, 1=alert
                "calm_streak": 0,
                "trigger_streak": 0,   # 连续触发计数
                "base_cos_ema": None,  # 基线 EMA
            }


        st = self._late_guard_state

        for epoch in range(start_epoch, self.helper.config.epochs):
            sampled_participants = self.sample_participants(epoch)
            is_optimize = False

            self.helper.config.eps_c, self.helper.config.eps_s, self.helper.config.importance_threshold = \
                self.update_dynamic_params(epoch, self.helper.config.epochs)

            weight_accumulator, weight_accumulator_by_client = self.train_once(
                epoch, sampled_participants, is_optimize, poisonupdate_dict
            )

            if self.helper.config.agg_method == 'new_algorithm':
                # ---------------- 0) 选择 normal/alert 的强度 ----------------
                in_alert = (st["alert_level"] > 0)
                n = len(sampled_participants)

                # suspicious 选择：不再固定 topk=2/3（攻击者多时会漏网）
                max_susp_ratio_normal = getattr(self.helper.config, "guard_max_susp_ratio_normal", 0.4)
                max_susp_ratio_alert = getattr(self.helper.config, "guard_max_susp_ratio_alert", 0.6)
                max_susp_k = int(max(1, math.ceil((max_susp_ratio_alert if in_alert else max_susp_ratio_normal) * n)))
                min_susp_k = int(getattr(self.helper.config, "guard_min_susp_k", 1))
                z_thr = float(
                    getattr(self.helper.config, "guard_susp_z_thr_alert", 0.7)
                    if in_alert else getattr(self.helper.config, "guard_susp_z_thr_normal", 1.0)
                )
                clean_frac = float(getattr(self.helper.config, "guard_clean_frac", 0.5))

                probe_b = probe_batches_alert if in_alert else probe_batches_normal
                clean_b = clean_batches_alert if in_alert else clean_batches_normal
                q_use = q_alert if in_alert else q_normal

                # ---------------- 1) 估计 clean 更新方向（-grad） ----------------
                self.helper.global_model.eval()
                max_batches = getattr(self.helper.config, "max_batches", clean_b)
                clean_g = self.compute_clean_grad_dict(self.helper.global_model, max_batches=max_batches)

                # ---------------- 2) 统计 pre-score（不依赖 oracle） ----------------
                pre_scores = {}
                for i, cid in enumerate(sampled_participants):
                    cos, l2, r = self.score_update_alignment(
                        weight_accumulator_by_client[i], clean_g, topk_ratio=topk_ratio
                    )
                    pre_scores[cid] = (cos, l2, r)

                pre_cos_list = [pre_scores[c][0] for c in sampled_participants]
                pre_med_cos = float(np.median(pre_cos_list)) if len(pre_cos_list) else float("nan")
                pre_min_cos = float(np.min(pre_cos_list)) if len(pre_cos_list) else float("nan")

                # ---------------- 3) GSA（温和净化一次） ----------------
                weight_accumulator_by_client = self.sanitize_updates_gsa(
                    weight_accumulator_by_client,
                    clean_g,
                    tau=tau,
                    alpha=gsa_alpha,
                    beta=gsa_beta
                )

                # ---------------- 4) post-score（看 benign 是否被伤到） ----------------
                post_scores = {}
                for i, cid in enumerate(sampled_participants):
                    cos, l2, r = self.score_update_alignment(
                        weight_accumulator_by_client[i], clean_g, topk_ratio=topk_ratio
                    )
                    post_scores[cid] = (cos, l2, r)

                post_cos_list = [post_scores[c][0] for c in sampled_participants]
                post_med_cos = float(np.median(post_cos_list)) if len(post_cos_list) else float("nan")
                post_min_cos = float(np.min(post_cos_list)) if len(post_cos_list) else float("nan")

                print(
                    f"[GSA-DBG] epoch={epoch} adv_ids=[] | "
                    f"ADV cos nan->nan, ADV ||Δ|| nan->nan, ADV topk-r nan->nan | "
                    f"BEN cos {pre_med_cos:.3f}->{post_med_cos:.3f}"
                )

                # ---------------- 5) detect（默认 top_k_susp=；alert 才加大） ----------------
                eta_cap = getattr(self.helper.config, "guard_eta_cap", None)
                if eta_cap is None:
                    eta_cap = (
                        getattr(self.helper.config, "guard_eta_cap_min", 0.02),
                        getattr(self.helper.config, "guard_eta_cap_max", 0.2),
                    )
                suspicious_ids, mask_by_sid = self.detect_suspicious_and_build_mask(
                    self.helper.global_model,
                    weight_accumulator_by_client,
                    sampled_participants,
                    eta=eta,
                    topk_ratio=topk_ratio,
                    sign_thr=sign_thr,
                    q=q_use,
                    max_probe_batches=probe_b,
                    top_k_susp=max_susp_k,
                    z_thr=z_thr,
                    min_k=min_susp_k,
                    clean_frac=clean_frac,
                    probe_trigger_strength=getattr(self.helper.config, "guard_probe_trigger_strength", 1.0),
                    probe_trigger_seed=getattr(self.helper.config, "guard_probe_trigger_seed", 1234),
                    tsp_lambda=getattr(self.helper.config, "guard_tsp_lambda", 0.5),
                    eta_cap=eta_cap,
                )

                suspicious_ids = list(suspicious_ids)

                sus_rate = float(len(suspicious_ids)) / max(1, len(sampled_participants))
                print(f"[LATE-Guard] epoch={epoch} suspicious_ids={suspicious_ids}, sampled={sampled_participants}")

                # ---------------- 6) Trigger：用滚动基线决定是否进入/维持 alert（改：连续触发 + EMA 基线） ----------------
                # 触发需要连续 K 轮（避免 benign 单轮抖动直接进 alert）
                trigger_need = getattr(self.helper.config, "guard_trigger_need", 3)
                min_hist = getattr(self.helper.config, "guard_min_hist", 20)  # hist 不足时不启用 drop trigger
                ema_m = getattr(self.helper.config, "guard_base_ema_m", 0.9)  # 基线 EMA momentum（越大越慢）

                # 初始化 EMA 基线（第一次有值就设）
                if (st.get("base_cos_ema", None) is None) and (not math.isnan(post_med_cos)):
                    st["base_cos_ema"] = float(post_med_cos)

                baseline_cos = float(st["base_cos_ema"]) if st.get("base_cos_ema", None) is not None else float(post_med_cos)
                cos_drop = baseline_cos - post_med_cos

                # 是否允许用 "drop" 作为 trigger（hist 太短时不允许）
                use_drop_trigger = (len(st["cos_hist"]) >= min_hist)

                trigger = False
                if epoch >= warmup:
                    cond_sus = (sus_rate >= sus_rate_hi)
                    cond_abs = (post_med_cos < cos_med_thr)
                    cond_drop = (use_drop_trigger and (cos_drop > cos_drop_thr))
                    trigger = (cond_sus or cond_abs or cond_drop)

                # 连续触发计数（hysteresis）
                if trigger:
                    st["trigger_streak"] = min(trigger_need, st.get("trigger_streak", 0) + 1)
                else:
                    st["trigger_streak"] = max(0, st.get("trigger_streak", 0) - 1)

                # 进入/退出 alert 的状态机（进入：连续 K 轮；退出：连续 calm_patience 轮不触发）
                if st["alert_level"] == 0:
                    if st["trigger_streak"] >= trigger_need:
                        st["alert_level"] = 1
                        st["calm_streak"] = 0
                else:
                    if trigger:
                        st["calm_streak"] = 0
                    else:
                        st["calm_streak"] += 1
                        if st["calm_streak"] >= calm_patience:
                            st["alert_level"] = 0
                            st["calm_streak"] = 0
                            st["trigger_streak"] = 0

                in_alert = (st["alert_level"] > 0)

                # ---------------- 基线更新策略（关键改动）
                # 1) normal：正常更新 hist + EMA
                # 2) alert：也允许"慢速"更新 EMA（但要求 sus_rate 不高，避免攻击期污染）
                if (not math.isnan(post_med_cos)):
                    if (not in_alert):
                        st["cos_hist"].append(float(post_med_cos))
                        st["sus_hist"].append(float(sus_rate))
                        st["base_cos_ema"] = ema_m * float(st["base_cos_ema"]) + (1 - ema_m) * float(post_med_cos)
                    else:
                        # alert 时仅在 sus_rate 较低（更像 benign 抖动）才慢速回归
                        if sus_rate <= getattr(self.helper.config, "guard_alert_ema_sus_cap", 0.2):
                            st["base_cos_ema"] = ema_m * float(st["base_cos_ema"]) + (1 - ema_m) * float(post_med_cos)

                baseline_cos = float(st["base_cos_ema"])
                cos_drop = baseline_cos - post_med_cos

                # ---------------- 7) gsa_flags：只在"detect 已经明显偏高"时才 union，避免 benign 扩张 ----------------
                gsa_flags = []
                if in_alert:
                    # 只有当 detect 本身已经较多，才用 gsa 再补抓
                    if sus_rate >= getattr(self.helper.config, "guard_gsa_union_sus_rate", 0.2):
                        gsa_flags = self.flag_by_gsa_metrics(
                            sampled_participants,
                            weight_accumulator_by_client,
                            clean_g,
                            topk_ratio=topk_ratio,
                            cos_bad_thr=getattr(self.helper.config, "guard_cos_bad_thr", 0.02),
                            r_mad_k=getattr(self.helper.config, "guard_r_mad_k", 3.0)
                        )
                        suspicious_ids = list(set(suspicious_ids).union(set(gsa_flags)))

                # 为新加入的可疑客户端补齐 mask（否则 agg_late_guard 里可能 key 不存在）
                for cid in suspicious_ids:
                    if cid not in mask_by_sid:
                        mask_by_sid[cid] = {}

                # ---------------- 8) hard-drop 决策：不再等于 in_alert（关键改动）
                # 只有"强证据"才 hard drop：sus_rate 高 或 绝对 cos 很差 或 连续 alert 很久
                hard_drop = False
                if in_alert:
                    hard_drop = (
                        (sus_rate >= getattr(self.helper.config, "guard_hard_sus_rate", 0.35)) or
                        (post_med_cos <= getattr(self.helper.config, "guard_hard_cos_med", 0.0)) or
                        (st.get("trigger_streak", 0) >= trigger_need)  # 已连续触发达到门槛
                    )

                print(
                    f"[LATE-Guard+] epoch={epoch} "
                    f"suspicious_ids={sorted(suspicious_ids)} "
                    f"(alert={in_alert}, hard_drop={hard_drop}, sus_rate={sus_rate:.3f}, "
                    f"med_cos={post_med_cos:.4f}, base_cos={baseline_cos:.4f}, drop={cos_drop:.4f}, "
                    f"trig_streak={st.get('trigger_streak', 0)}/{trigger_need}) "
                    f"flags(gsa)={sorted(list(gsa_flags)) if gsa_flags else []}, "
                    f"tau={tau:.4f}, gsa_alpha={gsa_alpha:.2f}, clean_batches={max_batches}"
                )

                # clip：仅 hard_drop 才启用（否则 benign 会被过度压缩）
                if hard_drop:
                    self.clip_client_updates(weight_accumulator_by_client, clip_mult=clip_mult)

                # 聚合：normal/alert 都 soft(mask)，hard_drop 时只 drop 一部分（但仍 mask 其余 suspicious）
                n = len(sampled_participants)
                normal_drop_ratio = float(getattr(self.helper.config, "guard_max_drop_ratio_normal", 0.3))
                alert_drop_ratio = float(getattr(self.helper.config, "guard_max_drop_ratio_alert", 0.5))
                max_drop = int(max(1, math.ceil((alert_drop_ratio if in_alert else normal_drop_ratio) * n)))

                if hard_drop:
                    # drop 数量随可疑比例自适应，但不超过 max_drop
                    drop_k = int(min(max_drop, max(1, math.ceil(sus_rate * n))))
                    drop_k = min(drop_k, len(suspicious_ids))
                    drop_ids = suspicious_ids[:drop_k]  # suspicious_ids 已按可疑分数降序
                else:
                    drop_ids = []

                self.aggregator.agg_late_guard(
                    self.helper.global_model,
                    weight_accumulator_by_client,
                    sampled_participants,
                    suspicious_ids=suspicious_ids,        # 用于 mask（软防御）
                    drop_ids=drop_ids,                    # hard_drop 时仅 drop 这部分
                    mask_by_suspicious_id=mask_by_sid,
                    drop_suspicious=hard_drop
                )

                # repair：仅 hard_drop 且按频率做
                if hard_drop and (repair_steps > 0) and (repair_every > 0) and (epoch % repair_every == 0):
                    self.repair_global_model(
                        self.helper.global_model,
                        steps=repair_steps,
                        lr=getattr(self.helper.config, "guard_repair_lr", 0.001),
                        lam_kl=getattr(self.helper.config, "guard_lam_kl", 1.0),
                        beta_prox=getattr(self.helper.config, "guard_beta_prox", 1e-4)
                    )
            else:
                self.aggregator.agg(
                    self.helper.global_model,
                    weight_accumulator,
                    weight_accumulator_by_client,
                    self.helper.client_models,
                    sampled_participants,
                    epoch
                )

            loss, acc = self.test_once()
            bkd_loss, bkd_acc = self.test_once(poison=self.helper.config.is_poison)

            self.log_once(epoch, loss, acc, bkd_loss, bkd_acc)
            accs.append(acc)
            asrs.append(bkd_acc)

        if self.helper.config.is_poison:
            self.save_res(accs, asrs)

        # 计算平均test_acc和bkd_acc并打印
        if accs:  # 确保列表不为空
            avg_acc = sum(accs) / len(accs)
            print(f"\n{'='*60}")
            print(f"Training finished! Average test accuracy: {avg_acc:.4f}")

            if asrs:  # 确保列表不为空
                avg_asr = sum(asrs) / len(asrs)
                print(f"Average backdoor attack success rate (ASR): {avg_asr:.4f}")
            print(f"{'='*60}")
        else:
            print("No accuracy values recorded.")

        # 返回结果，方便后续使用
        return accs, asrs


            
    
    def compute_clean_grad_dict(self, model, max_batches=2):
        """
        用服务器 repair_data 估计“干净更新方向” g_clean。
        关键：返回的是 -grad（因为 SGD 更新方向是 -∇L）。
        """
        device = self.device
        model.eval()  # 稳定 BN/Dropout（不影响 backward）
        model.zero_grad(set_to_none=True)

        n = 0
        for b, (x, y) in enumerate(self.helper.repair_data):
            if b >= max_batches:
                break
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = self.criterion(out, y)
            loss.backward()
            n += 1

        g = {}
        for name, p in model.named_parameters():
            if name == "decoder.weight":
                continue
            if p.grad is None:
                continue
            if not torch.is_floating_point(p.grad):
                continue
            # 关键：取负号，作为“更新方向参考”
            g[name] = -(p.grad.detach() / max(1, n)).clone()

        model.zero_grad(set_to_none=True)
        return g


    @torch.no_grad()
    def sanitize_updates_gsa(
        self,
        weight_accumulator_by_client,
        clean_grad_dict,
        tau=0.1,          # 对齐阈值（cos>=tau：放行）
        alpha=0.3,        # 轻度衰减（0<=cos<tau）
        beta=0.0,         # 强衰减（cos<0）
        eps=1e-12,
    ):
        """
        GSA：梯度子空间对齐（更新净化）
        - 先计算整体 cos(Δ, g_clean) 决定 lam（每个客户端一个 lam）
        - 再对每一层做“按层投影” Δ_l = Δ_par + Δ_perp，压 Δ_perp
        """

        sanitized = []
        for upd in weight_accumulator_by_client:
            # --------- (1) 先算整体 cosine，用来选择 lam ----------
            dot = 0.0
            upd_norm2 = 0.0
            g_norm2 = 0.0

            for name, delta in upd.items():
                if name == "decoder.weight":
                    continue
                if name not in clean_grad_dict:
                    continue
                if not torch.is_floating_point(delta):
                    continue
                d = delta.detach().float().view(-1)
                g = clean_grad_dict[name].detach().float().to(d.device).view(-1)

                dot += float((d * g).sum().item())
                upd_norm2 += float((d * d).sum().item())
                g_norm2 += float((g * g).sum().item())

            upd_norm = (max(upd_norm2, eps)) ** 0.5
            g_norm = (max(g_norm2, eps)) ** 0.5
            cos = dot / (upd_norm * g_norm + eps)

            if cos >= tau:
                lam = 1.0
            elif cos >= 0.0:
                lam = alpha
            else:
                lam = beta

            # --------- (2) 按层投影 + 压正交分量 ----------
            new_upd = {}
            for name, delta in upd.items():
                if name == "decoder.weight" or (not torch.is_floating_point(delta)):
                    new_upd[name] = delta
                    continue

                if name not in clean_grad_dict:
                    # 没有参考方向的层：保守轻衰减（也可以原样保留）
                    new_upd[name] = (delta.float() * alpha).to(dtype=delta.dtype)
                    continue

                d = delta.detach().float()
                g = clean_grad_dict[name].detach().float().to(d.device)

                gn2 = float((g.view(-1) * g.view(-1)).sum().item()) + eps
                dl = float((d.view(-1) * g.view(-1)).sum().item())
                coef_l = dl / gn2

                d_par = coef_l * g
                d_perp = d - d_par
                d_new = d_par + lam * d_perp

                new_upd[name] = d_new.to(dtype=delta.dtype)

            sanitized.append(new_upd)

        return sanitized

    
    def rebuild_weight_accumulator(self, weight_accumulator_by_client):
        """
        将 per-client 的 update 重新求和得到 weight_accumulator（与原逻辑一致）
        """
        wa = self.create_weight_accumulator()
        for upd in weight_accumulator_by_client:
            for name in wa:
                if name in upd and torch.is_floating_point(upd[name]):
                    wa[name].add_(upd[name].to(wa[name].device).to(wa[name].dtype))
        return wa
    
    def clip_update(self, update_dict, clip_norm=5.0):
        flat = torch.cat([v.view(-1) for v in update_dict.values()])
        n = torch.norm(flat, p=2) + 1e-12
        scale = min(1.0, clip_norm / n.item())
        if scale < 1.0:
            for k in update_dict:
                update_dict[k] = update_dict[k] * scale
        return update_dict

    def _random_patch_perturb(self, x, p=0.5):
        """
        x: [B,C,H,W]
        随机矩形区域做遮挡/噪声扰动（不是固定触发器）
        """
        if torch.rand(1).item() > p:
            return x

        x2 = x.clone()
        B, C, H, W = x2.shape
        # patch size：大概 1/8~1/4
        ph = int(H * (0.125 + 0.125 * torch.rand(1).item()))
        pw = int(W * (0.125 + 0.125 * torch.rand(1).item()))
        top = torch.randint(0, max(1, H - ph), (1,)).item()
        left = torch.randint(0, max(1, W - pw), (1,)).item()

        # 50% 置零，50% 加噪声
        if torch.rand(1).item() < 0.5:
            x2[:, :, top:top+ph, left:left+pw] = 0.0
        else:
            noise = torch.randn_like(x2[:, :, top:top+ph, left:left+pw]) * 0.1
            x2[:, :, top:top+ph, left:left+pw] = x2[:, :, top:top+ph, left:left+pw] + noise
        return x2
    
    @torch.no_grad()
    def detect_suspicious_and_build_mask(self, global_model, weight_accumulator_by_client, sampled_participants,
                                     eta=0.1, topk_ratio=0.01, sign_thr=0.7, q=0.75,
                                     max_probe_batches=2, top_k_susp=1,
                                     z_thr=1.0, min_k=1, clean_frac=0.5,
                                     probe_trigger_strength=1.0,
                                     probe_trigger_seed=1234,
                                     tsp_lambda=0.5,
                                     eta_cap=(0.02, 0.2)):
        def _normalize_eta_cap(x, default=(0.02, 0.2)):
            if x is None:
                return default
            # already a pair
            if isinstance(x, (list, tuple)) and len(x) == 2:
                return (float(x[0]), float(x[1]))
            # string like "[0.02,0.2]" or "0.02,0.2" or "(0.02, 0.2)"
            if isinstance(x, str):
                s = x.strip().replace("(", "").replace(")", "").replace("[", "").replace("]", "")
                parts = [p.strip() for p in s.split(",") if p.strip()]
                if len(parts) == 2:
                    return (float(parts[0]), float(parts[1]))
                if len(parts) == 1:
                    vmax = float(parts[0])
                    return (min(default[0], vmax), vmax)
                return default
            # scalar float/int -> interpret as eta_max
            if isinstance(x, (float, int)):
                vmax = float(x)
                return (min(default[0], vmax), vmax)

            return default

        eta_cap = _normalize_eta_cap(eta_cap, default=(0.02, 0.2))
        eta_min, eta_max = float(eta_cap[0]), float(eta_cap[1])
        device = self.device
        global_model.eval()

        # ---------- A) 几何分数：top-k 能量集中度 ----------
        geom_scores = []
        flat_cache = []
        for i, cid in enumerate(sampled_participants):
            vecs = []
            for name, delta in weight_accumulator_by_client[i].items():
                if name == "decoder.weight":
                    continue
                if not torch.is_floating_point(delta):
                    continue
                vecs.append(delta.detach().view(-1).float().cpu())
            if len(vecs) == 0:
                v = torch.zeros(1)
            else:
                v = torch.cat(vecs, dim=0)
            flat_cache.append(v)

            l2 = torch.norm(v) + 1e-12
            k = max(1, int(topk_ratio * v.numel()))
            topk = torch.topk(v.abs(), k, largest=True).values
            r = torch.norm(topk) / l2
            geom_scores.append(r.item())

        # ---------- Trigger Sensitivity Probe (TSP)
        # 针对 My/FedESP：它会让 clean loss 基本不变 + 缩放更新避免检测
        # 所以我们探测：某客户端更新是否“更偏向让触发器样本的行为变化”，同时惩罚 clean-loss 变差（保 ACC）

        def _build_probe_trigger(chw, device):
            C, H, W = chw
            ts = int(getattr(self.helper.config, "trigger_size", 5))
            px = int(getattr(self.helper.config, "trigger_pos_x", 0))
            py = int(getattr(self.helper.config, "trigger_pos_y", max(0, W - ts)))  # 默认右上角

            g = torch.Generator(device=device)
            g.manual_seed(int(probe_trigger_seed))

            # 固定噪声 pattern（右上角），强度由 probe_trigger_strength 控制
            noise = torch.randn((1, C, H, W), generator=g, device=device).clamp(-2.0, 2.0)
            trig = noise * float(probe_trigger_strength)

            m = torch.zeros((1, C, H, W), device=device)
            m[:, :, px:px + ts, py:py + ts] = 1.0
            return trig, m

        def _apply_probe_trigger(x, trig, m):
            x_t = trig * m + x * (1.0 - m)
            # clamp 到输入本身的范围，避免探针变成“分布外噪声”
            xmin = x.amin(dim=(1,2,3), keepdim=True)
            xmax = x.amax(dim=(1,2,3), keepdim=True)
            return torch.max(torch.min(x_t, xmax), xmin)


        # 取一批 repair_data 推断图像形状
        for b, (x0, y0) in enumerate(self.helper.repair_data):
            break
        C, H, W = int(x0.shape[1]), int(x0.shape[2]), int(x0.shape[3])
        probe_trig, probe_mask = _build_probe_trigger((C, H, W), device)

        # 1) 先算全局模型基线：clean vs triggered
        base_clean, base_trig, n_batches = 0.0, 0.0, 0
        for b, (x, y) in enumerate(self.helper.repair_data):
            if b >= max_probe_batches:
                break
            x, y = x.to(device), y.to(device)

            out = global_model(x)
            base_clean += self.criterion(out, y).item()

            x_t = _apply_probe_trigger(x, probe_trig, probe_mask)
            out_t = global_model(x_t)
            base_trig += self.criterion(out_t, y).item()

            n_batches += 1

        base_clean /= max(1, n_batches)
        base_trig  /= max(1, n_batches)

        # 2) 计算每个客户端更新的 L2，用于“自适应 eta”（让缩放隐蔽失效）
        l2_list = []
        for i, cid in enumerate(sampled_participants):
            vecs = []
            for name, delta in weight_accumulator_by_client[i].items():
                if name == "decoder.weight":
                    continue
                if not torch.is_floating_point(delta):
                    continue
                vecs.append(delta.detach().view(-1).float().cpu())
            v = torch.cat(vecs, dim=0) if len(vecs) else torch.zeros(1)
            l2_list.append(float(torch.norm(v).item()))
        l2_med = float(np.median(l2_list)) + 1e-12
        eta_min, eta_max = float(eta_cap[0]), float(eta_cap[1])

        # 3) 对每个客户端：虚拟应用更新 -> 再测 clean / triggered loss
        tsp_scores = []
        for i, cid in enumerate(sampled_participants):
            tmp = copy.deepcopy(global_model).to(device)
            sd = tmp.state_dict()

            l2_i = float(l2_list[i]) + 1e-12
            eta_i = float(eta) * (l2_med / l2_i)   # 更新越小，probe 时放大越多
            eta_i = max(eta_min, min(eta_max, eta_i))

            for name in sd:
                if name == "decoder.weight":
                    continue
                if not torch.is_floating_point(sd[name]):
                    continue
                if name not in weight_accumulator_by_client[i]:
                    continue
                delta = weight_accumulator_by_client[i][name].to(device)
                if not torch.is_floating_point(delta):
                    continue
                sd[name].add_(eta_i * delta.to(sd[name].dtype))
            tmp.load_state_dict(sd, strict=False)
            tmp.eval()

            clean_i, trig_i, nb = 0.0, 0.0, 0
            for b, (x, y) in enumerate(self.helper.repair_data):
                if b >= max_probe_batches:
                    break
                x, y = x.to(device), y.to(device)

                out = tmp(x)
                clean_i += self.criterion(out, y).item()

                x_t = _apply_probe_trigger(x, probe_trig, probe_mask)
                out_t = tmp(x_t)
                trig_i += self.criterion(out_t, y).item()
                nb += 1

            clean_i /= max(1, nb)
            trig_i  /= max(1, nb)

            # TSP 分数：触发器损失变差 - λ * clean 损失变差（保 ACC）
            tsp = (trig_i - base_trig) - float(tsp_lambda) * (clean_i - base_clean)
            tsp_scores.append(tsp)

        # ---------- C) robust z-score 合成总分 ----------
        def robust_z(arr):
            arr = np.array(arr, dtype=np.float32)
            med = np.median(arr)
            mad = np.median(np.abs(arr - med)) + 1e-12
            return (arr - med) / (1.4826 * mad)

        s = robust_z(geom_scores) + robust_z(tsp_scores)

        # 先按总分从大到小排序
        order = np.argsort(-s)  # 大到小
        n = len(order)

        # 1) threshold + bounds 选择 suspicious（避免固定 top-k 在多攻击者时漏网）
        max_k = int(min(max(1, top_k_susp), n))
        min_k = int(min(max(1, min_k), max_k))
        thr_z = float(z_thr)

        # 按排序顺序先取所有超过阈值的
        cand = [int(idx) for idx in order if float(s[int(idx)]) >= thr_z]
        if len(cand) < min_k:
            suspicious_indices = [int(order[t]) for t in range(min_k)]
        else:
            suspicious_indices = cand[:max_k]

        # 保持 suspicious_ids 与 suspicious_indices 同序（desc）
        suspicious_ids = [sampled_participants[i] for i in suspicious_indices]

        # ---------- D) 用“inlier(低分)子集”构建共识 mask ----------
        # 不再用“非可疑=干净”的补集（攻击者多时会污染共识）
        susp_set = set(suspicious_indices)
        clean_k = int(max(1, round(float(clean_frac) * len(sampled_participants))))
        low_order = np.argsort(s)  # 小到大
        avail = [int(idx) for idx in low_order if int(idx) not in susp_set]
        if len(avail) == 0:
            # 极端情况：无法获得 inlier（上层可选择 hard_drop）
            clean_indices = []
        else:
            clean_indices = avail[: min(clean_k, len(avail))]

        mask_by_suspicious_id = {}
        if len(clean_indices) == 0:
            # 极端情况：全部被判为可疑，直接返回（上层可以选择 drop）
            return suspicious_ids, mask_by_suspicious_id

        for idx in suspicious_indices:
            cid = sampled_participants[idx]
            mask_by_suspicious_id[cid] = {}

        for name in global_model.state_dict().keys():
            if name == "decoder.weight":
                continue

            # 只对浮点参数做 mask
            if not torch.is_floating_point(global_model.state_dict()[name]):
                continue

            clean_stack = torch.stack(
                [weight_accumulator_by_client[i][name].detach().float().cpu()
                 for i in clean_indices
                 if name in weight_accumulator_by_client[i]
                 and torch.is_floating_point(weight_accumulator_by_client[i][name])],
                dim=0
            )  # [n_clean, ...]

            if clean_stack.numel() == 0:
                continue

            n_clean = clean_stack.shape[0]

            # sign agreement
            sign_sum = torch.sign(clean_stack).sum(dim=0).abs()
            agree = sign_sum / max(1, n_clean)
            mask_sign = agree >= sign_thr

            # magnitude threshold
            abs_clean = clean_stack.abs()
            thr = torch.quantile(abs_clean, q=q, dim=0)

            # 为每个可疑客户端生成 mask
            for idx in suspicious_indices:
                cid = sampled_participants[idx]
                if name not in weight_accumulator_by_client[idx]:
                    continue
                susp_delta = weight_accumulator_by_client[idx][name].detach().float().cpu()
                if not torch.is_floating_point(susp_delta):
                    continue
                mask_mag = susp_delta.abs() <= thr
                mask = mask_sign & mask_mag
                mask_by_suspicious_id[cid][name] = mask

        return suspicious_ids, mask_by_suspicious_id


    def train_once(self, epoch, sampled_participants, is_optimize, poisonupdate_dict):
        weight_accumulator = self.create_weight_accumulator()
        weight_accumulator_by_client = []
        client_count = 0
        attacker_idxs = []
        global_model_copy = self.create_global_model_copy()
        local_asr = []
        first_adversary = self.contain_adversary(epoch, sampled_participants)
        if first_adversary >= 0 and ('sin' in self.helper.config.attacker_method):
            model = self.helper.local_model
            self.copy_params(model, global_model_copy)
        if first_adversary >= 0:
            self.attack_sum += 1
            print(f'Epoch {epoch}, poisoning by {first_adversary}, attack sum {self.attack_sum}.')
        else:
            print(f'Epoch {epoch}, no adversary.')
        for participant_id in sampled_participants:
            model = self.helper.local_model
            self.copy_params(model, global_model_copy)
            model.train()
            if not self.if_adversary(epoch, participant_id, sampled_participants):
                self.train_benign(participant_id, model, epoch)
            else:
                attacker_idxs.append(client_count)
                if self.helper.config.attack_method == 'A3FL':
                    self.attacker.search_trigger(model, self.helper.train_data[first_adversary], 'outter',
                                                 first_adversary,
                                                 epoch)
                elif self.helper.config.attack_method == 'My':
                    self.attacker.my_search_trigger(model, self.helper.train_data[first_adversary], 'outter',
                                                    first_adversary, epoch)
                self.train_malicious(participant_id, model, epoch, poisonupdate_dict, is_optimize)
            weight_accumulator, single_wa = self.update_weight_accumulator(model, weight_accumulator)
            weight_accumulator_by_client.append(single_wa)
            self.helper.client_models[participant_id].load_state_dict(model.state_dict())
            client_count += 1
        return weight_accumulator, weight_accumulator_by_client

    
    def repair_global_model(self, global_model, steps=100, lr=0.001, lam_kl=1.0, beta_prox=1e-4):
        """
        聚合后轻量修复：CE + 一致性KL + prox
        """
        device = self.device
        global_model.train()

        # 保存前一版本参数用于 prox
        prev = {}
        for k, v in global_model.state_dict().items():
            if k == "decoder.weight":
                continue
            if not torch.is_floating_point(v):
                continue
            prev[k] = v.detach().clone()


        opt = torch.optim.SGD(global_model.parameters(), lr=lr, momentum=0.0, weight_decay=0.0)

        it = 0
        for x, y in self.helper.repair_data:
            x, y = x.to(device), y.to(device)

            # forward on clean
            out = global_model(x)
            loss_ce = self.criterion(out, y)

            with torch.no_grad():
                p = torch.softmax(out, dim=1)

            # perturbed
            x2 = self._random_patch_perturb(x, p=1.0)
            out2 = global_model(x2)
            p2 = torch.log_softmax(out2, dim=1)

            loss_kl = torch.nn.functional.kl_div(p2, p, reduction='batchmean')

            # prox
            prox = 0.0
            for name, param in global_model.state_dict().items():
                if name == "decoder.weight":
                    continue
                if name not in prev:
                    continue
                if not torch.is_floating_point(param):
                    continue
                prox = prox + torch.mean((param - prev[name].to(param.device)) ** 2)


            loss = loss_ce + lam_kl * loss_kl + beta_prox * prox

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            it += 1
            if it >= steps:
                break

    def update_weight_accumulator(self, model, weight_accumulator):
        single_weight_accumulator = dict()
        for name, data in model.state_dict().items():
            if name == 'decoder.weight' or '__' in name:
                continue
            weight_accumulator[name].add_(data - self.helper.global_model.state_dict()[name])
            single_weight_accumulator[name] = data - self.helper.global_model.state_dict()[name]
        return weight_accumulator, single_weight_accumulator

    def contain_adversary(self, epoch, sampled_participants):
        if self.helper.config.is_poison and \
                self.helper.config.poison_start_epoch <= epoch <= self.helper.config.poison_start_epoch + self.helper.config.poison_epochs and epoch >= 0:
            if self.helper.config.sample_method == 'random':
                for p in sampled_participants:
                    if p < self.helper.config.num_adversaries:
                        return p
            elif self.helper.config.sample_method == 'random_updates':
                if epoch in self.advs:
                    return self.advs[epoch][0]
        return -1

    def if_adversary(self, epoch, participant_id, sampled_participants):
        if self.helper.config.is_poison and self.helper.config.poison_start_epoch <= epoch <= self.helper.config.poison_start_epoch + self.helper.config.poison_epochs and epoch >= 0:
            if self.helper.config.sample_method == 'random' and participant_id < self.helper.config.num_adversaries:
                return True
            elif self.helper.config.sample_method == 'random_updates':
                if epoch in self.advs:
                    for idx in self.advs[epoch]:
                        if sampled_participants[idx] == participant_id:
                            return True
        else:
            return False

    def create_global_model_copy(self):
        global_model_copy = dict()
        for name, param in self.helper.global_model.named_parameters():
            global_model_copy[name] = self.helper.global_model.state_dict()[name].clone().detach().requires_grad_(False)
        return global_model_copy

    def create_weight_accumulator(self):
        weight_accumulator = dict()
        for name, data in self.helper.global_model.state_dict().items():
            ### don't scale tied weights:
            if name == 'decoder.weight' or '__' in name:
                continue
            weight_accumulator[name] = torch.zeros_like(data)
        return weight_accumulator

    def train_benign(self, participant_id, model, epoch):
        lr = self.get_lr(epoch)
        optimizer = torch.optim.SGD(model.parameters(), lr=lr,
                                    momentum=self.helper.config.momentum,
                                    weight_decay=self.helper.config.decay)
        for internal_epoch in range(self.helper.config.retrain_times):
            total_loss = 0.0
            for inputs, labels in self.helper.train_data[participant_id]:
                inputs, labels = inputs.cuda(), labels.cuda()
                output = model(inputs)
                loss = self.criterion(output, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    def train_malicious(self, participant_id, model, epoch, poisonupdate_dict, is_optimize):
        # 设置模型为训练模式
        model.train()
        lr = self.get_lr(epoch)
        optimizer = torch.optim.SGD(model.parameters(), lr=lr,
                                    momentum=self.helper.config.poison_momentum,
                                    weight_decay=self.helper.config.decay)
        benign_model = copy.deepcopy(model)
        benign_model.train()
        benign_optimizer = torch.optim.SGD(benign_model.parameters(), lr=lr,
                                           momentum=self.helper.config.momentum,
                                           weight_decay=self.helper.config.decay)
        for internal_epoch in range(self.helper.config.retrain_times):
            for inputs, labels in self.helper.train_data[participant_id]:
                inputs, labels = inputs.cuda(), labels.cuda()
                output = benign_model(inputs)
                loss = self.attacker_criterion(output, labels)
                benign_optimizer.zero_grad()
                loss.backward()
                benign_optimizer.step()
        for internal_epoch in range(self.helper.config.attacker_retrain_times):
            for inputs, labels in self.helper.train_data[participant_id]:
                inputs, labels = inputs.cuda(), labels.cuda()
                benign_inputs, benign_labels = inputs.cuda(), labels.cuda()
                # A3FL
                if self.helper.config.attack_method == 'A3FL':
                    inputs, labels = self.attacker.poison_input(inputs, labels)
                    output = model(inputs)
                    loss = self.attacker_criterion(output, labels)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                elif self.helper.config.attack_method == 'My':
                    inputs, labels = self.attacker.poison_input(inputs, labels)
                    output = model(inputs)
                    loss = self.attacker_criterion(output, labels)

                    optimizer.zero_grad()
                    loss.backward()
                    k = 0.95
                    # 保存每层的梯度绝对值并生成掩码
                    mask_grad_list = self.generate_gradient_masks(model, benign_model, k)
                    amplification_factor = self.calculate_amplified_difference(model, benign_model, mask_grad_list, 30)
                    # 遍历模型参数并应用掩码
                    for param, benign_param, mask in zip(model.parameters(), benign_model.parameters(), mask_grad_list):
                        if param.grad is not None and benign_param.grad is not None:
                            # 对掩码为1的部分放大恶意梯度
                            param.grad.data = mask * param.grad.data * amplification_factor
                    # 更新恶意模型的参数
                    optimizer.step()
                elif self.helper.config.attack_method == 'Neurotoxin':
                    inputs, labels = self.attacker.poison_input(inputs, labels)
                    output = model(inputs)
                    # Neurotoxin 攻击方法
                    loss = self.attacker_criterion(output, labels)

                    optimizer.zero_grad()
                    loss.backward()

                    # 收集所有参数的梯度并拼接为一个大向量
                    grad_list = []
                    for name, parms in benign_model.named_parameters():
                        if parms.requires_grad and parms.grad is not None:
                            grad_list.append(parms.grad.view(-1))

                    full_grad_vector = torch.cat(grad_list)

                    # 计算全模型梯度绝对值的top-k值（k由比例ratio决定）
                    k = int(0.95 * full_grad_vector.numel())
                    top_k_values, _ = torch.topk(full_grad_vector.abs(), k)
                    threshold = top_k_values[-1].item()  # 最小的top-k梯度值作为阈值

                    # 创建梯度掩码（梯度绝对值大于等于阈值的位置为1，其余为0）
                    mask_grad_list = []
                    for name, parms in benign_model.named_parameters():
                        if parms.requires_grad and parms.grad is not None:
                            mask = (parms.grad.abs() >= threshold).float()
                            mask_grad_list.append(mask)

                    # 将掩码应用到模型梯度中
                    mask_grad_list_copy = iter(mask_grad_list)
                    for name, parms in model.named_parameters():
                        if parms.requires_grad and parms.grad is not None:
                            parms.grad *= next(mask_grad_list_copy)

                    # 更新恶意模型参数
                    optimizer.step()
                elif self.helper.config.attack_method == 'DBA':
                    poison_pattern = []
                    if participant_id == 0:
                        poison_pattern = self.helper.poison_pattern_0
                    elif participant_id == 1:
                        poison_pattern = self.helper.poison_pattern_1
                    elif participant_id == 2:
                        poison_pattern = self.helper.poison_pattern_2
                    elif participant_id == 3:
                        poison_pattern = self.helper.poison_pattern_3
                    inputs, labels = self.attacker.DBA_poison_input(benign_inputs, benign_labels, poison_pattern)
                    output = model(inputs)
                    loss = self.attacker_criterion(output, labels)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                elif self.helper.config.attack_method == 'CerP':
                    if not is_optimize:
                        self.attacker.noise_trigger = self.optimize_trigger(model)
                        is_optimize = True
                    inputs, labels = self.attacker.cerp_poison_input(benign_inputs, benign_labels)
                    output = model(inputs)
                    class_loss = self.attacker_criterion(output, labels)
                    sum_cs = 0
                    for otherAd in range(0, self.helper.config.num_adversaries):
                        if otherAd == participant_id:
                            continue
                        else:
                            if otherAd in poisonupdate_dict:
                                otherAd_variables = poisonupdate_dict[otherAd]
                                sum_cs += self.model_cosine_similarity(model, otherAd_variables)
                    malDistance_Loss = self.model_dist_norm_var(model, benign_model)
                    loss = class_loss + self.helper.config.alpha_loss * malDistance_Loss + \
                           self.helper.config.beta_loss * sum_cs
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
        if self.helper.config.attack_method == 'CerP':
            local_model = dict()
            for name, data in model.named_parameters():
                local_model[name] = data
            poisonupdate_dict[participant_id] = local_model
        #     将model模型保存到poisonupdate_dict中

    def calculate_amplified_difference(self, model, benign_model, mask_grad_list, threshold=0.01):
        # 初始放大因子范围
        min_factor = 0.0
        max_factor = 5.0  # 可以调整为一个适当的最大值
        amplification_factor = (min_factor + max_factor) / 2.0
        tol = 1e-4
        while (max_factor - min_factor) > tol:
            total_difference = 0.0

            for param, benign_param, mask in zip(model.parameters(), benign_model.parameters(), mask_grad_list):
                if param.grad is not None and benign_param.grad is not None:
                    # 根据当前放大因子计算最终梯度
                    malicious_grad = param.grad  # 恶意梯度
                    difference = mask * malicious_grad * amplification_factor
                    # 计算最终梯度与q全局梯度的差异
                    total_difference += difference.pow(2).sum().item()  # 累积差值的平方和

            # 计算当前的 Frobenius 范数
            current_norm = total_difference ** 0.5
            # 检查是否满足阈值条件
            if current_norm < threshold:
                min_factor = amplification_factor  # 增大放大因子范围
            else:
                max_factor = amplification_factor  # 减小放大因子范围

            amplification_factor = (min_factor + max_factor) / 2.0  # 更新放大因子为中点值

        return amplification_factor

    def generate_gradient_masks(self, model, benign_model, trigger_sensitivity_ratio):
        mask_grad_list = []

        for param, benign_param in zip(model.parameters(), benign_model.parameters()):
            if param.grad is not None and benign_param.grad is not None:
                # 获取恶意模型和良性模型的梯度
                malicious_grad = param.grad
                benign_grad = benign_param.grad

                # 计算触发器敏感性（梯度差异）
                trigger_sensitivity = (malicious_grad - benign_grad).abs()

                # 获取触发器敏感性的排序和比例选择
                num_elements = trigger_sensitivity.numel()
                num_select = int(trigger_sensitivity_ratio * num_elements)

                # 根据触发器敏感性选择重要参数
                _, sensitive_indices = torch.topk(trigger_sensitivity.view(-1), num_select, largest=True)
                mask = torch.zeros_like(trigger_sensitivity.view(-1))
                mask[sensitive_indices] = 1.0  # 选择敏感的参数

                # 恢复掩码到参数的原始形状
                mask = mask.view(param.shape)
                mask_grad_list.append(mask)

        return mask_grad_list

    def get_lr(self, epoch):
        if self.helper.config.lr_method == 'exp':
            tmp_epoch = epoch
            if self.helper.config.is_poison and self.helper.config.load_benign_model:
                tmp_epoch += self.helper.config.poison_start_epoch
            lr = self.helper.config.lr * (self.helper.config.gamma ** tmp_epoch)
        elif self.helper.config.lr_method == 'linear':
            if self.helper.config.is_poison or epoch > 1900:
                lr = 0.002
            else:
                lr_init = self.helper.config.lr
                target_lr = self.helper.config.target_lr
                if epoch <= self.helper.config.epochs / 2.:
                    lr = epoch * (target_lr - lr_init) / (self.helper.config.epochs / 2. - 1) + lr_init - (
                            target_lr - lr_init) / (self.helper.config.epochs / 2. - 1)
                else:
                    lr = (epoch - self.helper.config.epochs / 2) * (-target_lr) / (
                            self.helper.config.epochs / 2) + target_lr

                if lr <= 0.002:
                    lr = 0.002
        return lr

    def sample_participants(self, epoch):
        if self.helper.config.sample_method in ['random', 'random_updates']:
            sampled_participants = random.sample(
                range(self.helper.config.num_total_participants),
                self.helper.config.num_sampled_participants)
        elif self.helper.config.sample_method == 'fix-rate':
            start_index = (
                                  epoch * self.helper.config.num_sampled_participants) % self.helper.config.num_total_participants
            sampled_participants = list(range(start_index, start_index + self.helper.config.num_sampled_participants))
        else:
            raise NotImplementedError
        assert len(sampled_participants) == self.helper.config.num_sampled_participants
        print("Sampled participants:", sampled_participants)
        return sampled_participants

    def copy_params(self, model, target_params_variables):
        for name, layer in model.named_parameters():
            layer.data = copy.deepcopy(target_params_variables[name])

    def optimize_trigger(self, local_model):
        # load model
        model = copy.deepcopy(local_model)
        model.load_state_dict(self.helper.global_model.state_dict())
        model.eval()
        aa = copy.deepcopy(self.attacker.trigger0).cuda()
        noise = self.attacker.noise_trigger.clone().detach().cuda()
        noise.requires_grad = True
        optimizer = torch.optim.SGD([noise], lr=0.01)  # Using SGD to update the noise
        for e in range(1):
            for poison_id in range(0, self.helper.config.num_adversaries):
                for inputs, labels in self.helper.train_data[poison_id]:
                    inputs = inputs.cuda()
                    labels = labels.cuda()
                    # 将lables都修改为target_class
                    labels[:] = self.helper.config.target_class
                    # 创建一个新的 Tensor 以避免直接修改原始输入
                    modified_inputs = inputs.clone()

                    # 使用修改后的输入和噪声进行训练
                    output = model(modified_inputs + noise)

                    class_loss = self.criterion(output, labels)
                    optimizer.zero_grad()  # Clear previous gradients
                    class_loss.backward()
                    optimizer.step()
                    # 限制噪声更新到特定区域
                    with torch.no_grad():
                        delta_noise = noise - aa
                        noise.data = aa + self.proj_lp(delta_noise, 10, 2).data
        return noise

    def proj_lp(self, delta, max_norm, p=2):
        """
        Project a vector to an Lp-ball of radius max_norm.
        """
        norm = torch.norm(delta, p=p, dim=-1, keepdim=True)
        scale = torch.clamp(norm / max_norm, max=1.0)  # Clamps the norm to max_norm
        return delta / scale

    def model_dist_norm_var(self, model, benign_model, norm=2):
        size = 0
        for name, layer in model.named_parameters():
            size += layer.view(-1).shape[0]

        sum_var = torch.FloatTensor(size).fill_(0)
        sum_var = sum_var.to(device=self.device)

        size = 0
        for (name, layer), (_, benign_layer) in zip(model.named_parameters(), benign_model.named_parameters()):
            sum_var[size:size + layer.view(-1).shape[0]] = (layer - benign_layer).view(-1)
            size += layer.view(-1).shape[0]

        return torch.norm(sum_var, norm)

    def model_cosine_similarity(self, model, target_params_variables, model_id='attacker'):
        cs_list = list()

        for name, data in model.named_parameters():
            # 忽略特定层
            if name == 'decoder.weight':
                continue

            # 获取模型和目标模型的展平参数
            model_flat = data.view(-1)
            target_flat = target_params_variables[name].view(-1)

            # 计算模型参数与目标参数之间的余弦相似度
            cs = F.cosine_similarity(model_flat, target_flat, dim=0)

            # 将余弦相似度值转换为标量，并添加到列表中
            cs_list.append(cs.item())

        # 计算所有余弦相似度的平均值
        average_cos_similarity = sum(cs_list) / len(cs_list) if cs_list else 0.0

        return average_cos_similarity
    
    def clip_client_updates(self, weight_accumulator_by_client, clip_mult=2.0):
        # 计算每个更新的 L2 norm
        norms = []
        for upd in weight_accumulator_by_client:
            s = 0.0
            for name, t in upd.items():
                if not torch.is_floating_point(t):
                    continue
                s += (t.float().view(-1) ** 2).sum().item()
            norms.append((s ** 0.5) + 1e-12)

        med = float(np.median(norms)) + 1e-12

        # 裁剪到 clip_mult * median
        for i, upd in enumerate(weight_accumulator_by_client):
            n = norms[i]
            max_n = clip_mult * med
            if n > max_n:
                scale = max_n / n
                for name, t in upd.items():
                    if not torch.is_floating_point(t):
                        continue
                    upd[name] = t * scale
