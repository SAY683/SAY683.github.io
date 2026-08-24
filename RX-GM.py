# --- GM4.2.py (Enhanced Negative Feedback & Balanced Interactions) ---
import dash
# ... (其他导入与 GM4.1.1 相同) ...
from dash import dcc, html, Input, Output, State, callback_context, no_update
from dash.exceptions import PreventUpdate
import plotly.graph_objects as go
import numpy as np
import random
import json
import time
from collections import defaultdict

# --- 1. 全局常量和辅助函数 (与 GM4.1.1 相同) ---
DIM_KEYS = ['b1_resource', 'b2_limitation', 'y1_clarity', 'y2_drive', 'y3_aspiration',
            'h1_possibilities', 'h2_innovation', 'h3_risk_appetite']

DIMENSION_LABELS_MAP_ZH = {
    'b1_resource': "本然 B1: 资源/条件", 'b2_limitation': "本然 B2: 限制/障碍",
    'y1_clarity': "应然 Y1: 价值清晰度", 'y2_drive': "应然 Y2: 驱动力", 'y3_aspiration': "应然 Y3: 理想高度",
    'h1_possibilities': "或然 H1: 可能性广度", 'h2_innovation': "或然 H2: 创新能力", 'h3_risk_appetite': "或然 H3: 风险承受"
}

AXIS_LABELS_ZH = {
    'simplified': {'x': '本然 B1 (资源)', 'y': '应然 Y2 (驱动)', 'z': '或然 H1 (可能)'},
    'composite': {'x': '本然 (综合)', 'y': '应然 (综合)', 'z': '或然 (综合)'}
}

MAX_LOG_LINES = 150


def sigmoid(x, k=1, x0=5):
    # Ensure x is a number to prevent overflow with np.exp if x is too large or small an array
    if isinstance(x, (np.ndarray, list)):  # Basic check, might need more robust for mixed types
        x = np.array(x, dtype=float)
        # Clip x to prevent overflow in exp
    x_clipped = np.clip(x, -500, 500)  # Adjust clipping range as needed
    return 1 / (1 + np.exp(-k * (x_clipped - x0)))


def scale_value(value, old_min=0, old_max=10, new_min=0, new_max=1):
    if old_max == old_min: return new_min
    return new_min + (value - old_min) * (new_max - new_min) / (old_max - old_min)


# --- 2. WorldState 类 (核心算法修改) ---
class WorldState:
    def __init__(self, name_zh, name_en, b1_res, b2_lim, y1_cla, y2_dri, y3_asp, h1_pos, h2_inn, h3_ris):
        self.name_zh = name_zh
        self.name_en = name_en
        dim_values = {  # 从构造函数参数创建字典
            'b1_resource': b1_res, 'b2_limitation': b2_lim,
            'y1_clarity': y1_cla, 'y2_drive': y2_dri, 'y3_aspiration': y3_asp,
            'h1_possibilities': h1_pos, 'h2_innovation': h2_inn, 'h3_risk_appetite': h3_ris
        }
        for key in DIM_KEYS:
            value = dim_values.get(key, 0)  # 安全获取
            setattr(self, key, np.clip(float(value), 0, 10))
        self.history = []
        self.neighbors = []
        self.active_effects_log = []

    def get_display_name(self):  # 无变化
        return f"{self.name_zh} ({self.name_en})"

    def get_coords_for_plot(self, coord_type='simplified', wb=(0.6, 0.4), wy=(0.4, 0.4, 0.2),
                            wh=(0.4, 0.4, 0.2)):  # 无变化
        if coord_type == 'simplified':
            return (self.b1_resource, self.y2_drive, self.h1_possibilities)
        elif coord_type == 'composite':
            b_composite = wb[0] * self.b1_resource - wb[1] * self.b2_limitation
            y_composite = wy[0] * self.y1_clarity + wy[1] * self.y2_drive + wy[2] * self.y3_aspiration
            h_composite = wh[0] * self.h1_possibilities + wh[1] * self.h2_innovation + wh[2] * self.h3_risk_appetite
            return (np.clip(b_composite, 0, 10), np.clip(y_composite, 0, 10), np.clip(h_composite, 0, 10))
        return (self.b1_resource, self.y1_clarity, self.h1_possibilities)

    def _apply_boundary_effect(self, current_value, delta_value, min_val=0, max_val=10, boundary_threshold=0.5):  # 无变化
        if current_value <= min_val + boundary_threshold and delta_value < 0:
            return delta_value * scale_value(current_value, min_val, min_val + boundary_threshold, 0, 1)
        elif current_value >= max_val - boundary_threshold and delta_value > 0:
            return delta_value * scale_value(current_value, max_val - boundary_threshold, max_val, 1, 0)
        return delta_value

    # --- Delta 计算方法 (核心修改) ---
    def _calculate_delta_b1(self, k, params, avg_b1_others, neighbor_effect_b1, successful_risk_this_step):
        # B1: 资源/条件
        effect_h2 = k['b1_from_h2'] * sigmoid(self.h2_innovation, k=0.7, x0=5)  # 创新效果，饱和
        effect_y2 = k['b1_from_y2'] * scale_value(self.y2_drive, 0, 10, 0.3, 1)  # 驱动力转化资源效率，有基础值

        loss_b2 = k['b1_loss_b2'] * (self.b2_limitation / 10) ** 1.8  # 限制的负面影响更强

        # 创新和高驱动力的成本
        cost_h2_activity = k.get('b1_cost_h2_activity', 0.005) * self.h2_innovation  # 创新活动消耗资源
        cost_y2_sustain = k.get('b1_cost_y2_sustain', 0.003) * self.y2_drive  # 维持高驱动消耗资源

        social_pressure_b1 = 0  # 全局平均资源带来的压力/激励
        if avg_b1_others is not None:
            social_pressure_b1 = k.get('b1_social_pressure', 0.005) * np.clip(avg_b1_others - self.b1_resource, -3, 3)

        # 风险失败的成本
        cost_failed_risk = 0
        if not successful_risk_this_step and self.h3_risk_appetite > params.get('h3_active_threshold_for_cost', 4):
            cost_failed_risk = k.get('b1_cost_failed_risk', 0.08) * (self.h3_risk_appetite / 10)  # 风险越高，失败成本越大

        delta = (effect_h2 + effect_y2 - loss_b2 - params[
            'base_consumption'] - cost_h2_activity - cost_y2_sustain - cost_failed_risk +
                 social_pressure_b1 + neighbor_effect_b1)
        return self._apply_boundary_effect(self.b1_resource, delta)

    def _calculate_delta_b2(self, k, params, neighbor_effect_b2, successful_risk_this_step):
        # B2: 限制/障碍
        reduction_y2 = k['b2_reduce_y2'] * sigmoid(self.y2_drive, k=0.8, x0=3)  # 驱动力克服限制，效果饱和
        reduction_h2 = k['b2_reduce_h2'] * sigmoid(self.h2_innovation, k=0.8, x0=3)  # 创新克服限制，效果饱和

        random_event_b2 = 0
        if random.random() < params.get('b2_random_event_chance', 0.04):  # 降低一点随机负面事件概率
            random_event_b2 = random.uniform(0, 0.8) * k.get('b2_random_factor', 0.15)  # 减小随机事件强度

        # 风险失败增加限制
        increase_from_failed_risk = 0
        if not successful_risk_this_step and self.h3_risk_appetite > params.get('h3_active_threshold_for_cost', 4):
            increase_from_failed_risk = k.get('b2_from_failed_risk', 0.05) * (self.h3_risk_appetite / 10)

        delta = -(reduction_y2 + reduction_h2) + random_event_b2 + increase_from_failed_risk + neighbor_effect_b2
        return self._apply_boundary_effect(self.b2_limitation, delta)

    def _calculate_delta_y1(self, k, params, delta_b1_raw, neighbor_effect_y1):
        # Y1: 价值清晰度/一致性
        b1_growth_factor = 1 if delta_b1_raw > params['b1_growth_threshold_for_y1'] else -0.5  # B1不增长甚至下降时，清晰度受损
        effect_b1_growth = k['y1_from_b1_growth'] * b1_growth_factor * \
                           (1 + 0.3 * sigmoid(self.y2_drive, k=0.7, x0=6))  # 高驱动协同B1增长提升Y1，饱和

        loss_b2 = k['y1_loss_b2'] * (self.b2_limitation / 7) ** 2.0  # 高限制强烈冲击清晰度

        # 理想与现实差距过大导致困惑
        reality_measure = (self.b1_resource + self.h1_possibilities) / 2  # 简化的现实衡量
        aspiration_reality_gap = self.y3_aspiration - reality_measure
        loss_from_gap = 0
        if aspiration_reality_gap > params.get('y1_gap_threshold_for_loss', 3.0):
            loss_from_gap = k.get('y1_loss_aspiration_gap', 0.01) * (
                    aspiration_reality_gap - params.get('y1_gap_threshold_for_loss', 3.0))

        delta = effect_b1_growth - loss_b2 - loss_from_gap + neighbor_effect_y1
        return self._apply_boundary_effect(self.y1_clarity, delta)

    def _calculate_delta_y2(self, k, params, delta_b1_raw, neighbor_effect_y2, successful_risk_this_step):
        # Y2: 驱动力/动机强度
        effect_y1 = k['y2_from_y1'] * sigmoid(self.y1_clarity, k=0.8, x0=4)  # 清晰度提供方向，饱和

        # 驱动力试图弥合与理想的差距，但如果理想过高不切实际，驱动力也可能受挫
        aspiration_gap_y2 = self.y3_aspiration - self.y2_drive
        effect_y3 = k['y2_from_y3'] * sigmoid(aspiration_gap_y2, k=0.5, x0=0) * (
                10 - self.y2_drive) / 10  # 差距越大，驱动力增长潜力越大，但自身越高越难增长
        # x0=0 意味着只要Y3>Y2就有正向力

        b1_growth_factor = 1 if delta_b1_raw > params['b1_growth_threshold_for_y1'] else -0.2  # 资源不增长，驱动力轻微受损
        effect_b1_growth = k['y2_from_b1_growth'] * b1_growth_factor

        loss_b2 = k['y2_loss_b2'] * (self.b2_limitation / 9) ** 1.5  # 限制消耗驱动力
        loss_low_y1 = k['y2_loss_low_y1'] * (1 - sigmoid(self.y1_clarity, k=1, x0=2.0))  # Y1极低时，驱动力严重受损

        # 风险失败打击驱动力
        loss_failed_risk = 0
        if not successful_risk_this_step and self.h3_risk_appetite > params.get('h3_active_threshold_for_cost', 4):
            loss_failed_risk = k.get('y2_loss_failed_risk', 0.06)

        delta = effect_y1 + effect_y3 + effect_b1_growth - loss_b2 - loss_low_y1 - loss_failed_risk + neighbor_effect_y2
        return self._apply_boundary_effect(self.y2_drive, delta)

    def _calculate_delta_y3(self, k, params, neighbor_effect_y3):
        # Y3: 理想高度/道德标准
        # 缓慢向当前驱动力调整，但受清晰度(Y1)和资源(B1)支持
        adjustment_from_y2 = k['y3_adjust_y2'] * (self.y2_drive - self.y3_aspiration) * 0.03  # 更慢的调整
        boost_y1 = k.get('y3_boost_y1', 0.015) * sigmoid(self.y1_clarity - 5.5, k=0.8, x0=0)  # Y1高于5.5才提升Y3
        support_b1 = k.get('y3_support_b1', 0.01) * sigmoid(self.b1_resource - 4, k=0.7, x0=0)  # B1高于4才支持Y3提升

        # 如果现实(B1)远低于理想(Y3)，理想可能被迫降低（现实打击）
        reality_crush = 0
        if self.b1_resource < self.y3_aspiration - params.get('y3_reality_gap_threshold', 4.0):
            reality_crush = k.get('y3_loss_reality_gap', 0.02) * (
                    self.y3_aspiration - self.b1_resource - params.get('y3_reality_gap_threshold', 4.0))

        delta = adjustment_from_y2 + boost_y1 + support_b1 - reality_crush + neighbor_effect_y3
        return self._apply_boundary_effect(self.y3_aspiration, delta)

    def _calculate_delta_h1(self, k, params, neighbor_effect_h1):
        # H1: 可能性广度/选择多样性
        effect_b1 = k['h1_from_b1'] * sigmoid(self.b1_resource, k=0.6, x0=4)  # 资源带来的可能性，饱和
        effect_h2 = k['h1_from_h2'] * sigmoid(self.h2_innovation, k=0.7, x0=3.5)  # 创新拓展可能性，饱和
        loss_b2 = k['h1_loss_b2'] * (self.b2_limitation / 6) ** 2.0  # 高限制严重压缩可能性空间

        # 低驱动力(Y2)或低清晰度(Y1)可能导致无法感知或利用可能性
        loss_low_y2_y1 = k.get('h1_loss_low_y_factor', 0.01) * \
                         ((1 - scale_value(self.y2_drive, 0, 4, 0, 1)) + (
                                 1 - scale_value(self.y1_clarity, 0, 4, 0, 1))) / 2

        delta = effect_b1 + effect_h2 - loss_b2 - loss_low_y2_y1 + neighbor_effect_h1
        return self._apply_boundary_effect(self.h1_possibilities, delta)

    def _calculate_delta_h2(self, k, params, neighbor_effect_h2):
        # H2: 创新/适应能力
        # 受驱动力(Y2)、风险承受(H3)和价值清晰度(Y1)协同影响
        synergy_y1_y2 = self.y1_clarity * self.y2_drive / 100  # Y1和Y2的协同效应
        effect_y2_h3 = (k['h2_from_y2'] * self.y2_drive + k['h2_from_h3'] * self.h3_risk_appetite) * \
                       (0.5 + 0.5 * sigmoid(synergy_y1_y2, k=0.1, x0=(6 * 7 / 100)))  # Y1*Y2协同，x0设为中等偏上值

        # 维持创新能力需要持续投入（高Y2或H3），否则衰减
        practice_factor = (scale_value(self.y2_drive, 0, 10, 0.2, 1) + scale_value(self.h3_risk_appetite, 0, 10, 0.2,
                                                                                   1)) / 2
        decay = k['h2_decay_no_practice'] * (1 - practice_factor) * (self.h2_innovation / 8)  # 实践不足则衰减，衰减与当前H2有关

        delta = effect_y2_h3 - decay + neighbor_effect_h2
        return self._apply_boundary_effect(self.h2_innovation, delta)

    def _calculate_delta_h3(self, k, params, delta_b1_raw, neighbor_effect_h3, successful_risk_this_step):
        # H3: 风险承受意愿/探索精神
        effect_risk_success = k[
                                  'h3_from_risk_success'] * successful_risk_this_step  # successful_risk_this_step (0 or 1)

        # 清晰的价值(Y1)和强驱动(Y2)支持冒险
        effect_y1 = k['h3_from_y1'] * sigmoid(self.y1_clarity, k=0.7, x0=5.5)
        effect_y2 = k['h3_from_y2'] * sigmoid(self.y2_drive, k=0.7, x0=5.5)

        loss_b2 = k['h3_loss_b2'] * (self.b2_limitation / 10) ** 1.2  # 限制降低冒险意愿，影响相对温和

        # 风险失败的惩罚
        penalty_failed_risk = 0
        if not successful_risk_this_step and self.h3_risk_appetite > params.get('h3_active_threshold_for_cost', 4):
            penalty_failed_risk = k.get('h3_penalty_failed_risk', 0.08) * (self.h3_risk_appetite / 10)

        delta = effect_risk_success + effect_y1 + effect_y2 - loss_b2 - penalty_failed_risk + neighbor_effect_h3
        return self._apply_boundary_effect(self.h3_risk_appetite, delta)

    def _calculate_neighbor_effects(self, params, all_states_objects_dict):
        # (与GM4.1基本一致，但可以调整社会交互系数)
        k_social = params['coefficients'].get('social_interactions', {})
        effects = {key: 0.0 for key in DIM_KEYS}
        if not self.neighbors: return effects

        num_valid_neighbors = 0
        avg_neighbor_y1 = 0
        avg_neighbor_y2 = 0

        for neighbor_name in self.neighbors:
            if neighbor_name in all_states_objects_dict and neighbor_name != self.name_en:
                neighbor_obj = all_states_objects_dict[neighbor_name]
                num_valid_neighbors += 1
                avg_neighbor_y1 += neighbor_obj.y1_clarity
                avg_neighbor_y2 += neighbor_obj.y2_drive

                # B1 竞争/合作 (更强的竞争效应)
                b1_diff = neighbor_obj.b1_resource - self.b1_resource
                if b1_diff > k_social.get('b1_comp_diff_thresh', 1.0):  # 竞争阈值降低
                    effects['b1_resource'] -= k_social.get('b1_comp_loss_factor', 0.01) * b1_diff  # 竞争损失增强
                elif abs(b1_diff) < k_social.get('b1_coop_diff_thresh', 0.5) and self.b1_resource > 2.5:  # 合作条件微调
                    effects['b1_resource'] += k_social.get('b1_coop_gain_factor', 0.002) * min(self.b1_resource,
                                                                                               neighbor_obj.b1_resource)

        if num_valid_neighbors > 0:
            avg_neighbor_y1 /= num_valid_neighbors
            avg_neighbor_y2 /= num_valid_neighbors

            # Y1, Y2 对齐/传染 (引入阻尼，防止无限放大)
            effects['y1_clarity'] += k_social.get('y1_alignment_factor', 0.008) * (avg_neighbor_y1 - self.y1_clarity) * \
                                     (1 - sigmoid(self.y1_clarity, k=1, x0=8))  # 自身Y1很高时，受影响减弱
            effects['y2_drive'] += k_social.get('y2_contagion_factor', 0.015) * (avg_neighbor_y2 - self.y2_drive) * \
                                   (1 - sigmoid(self.y2_drive, k=1, x0=8))  # 自身Y2很高时，受影响减弱
        return effects

    def evolve(self, params, all_states_objects_dict, active_event_effects_on_self):
        # (演化主逻辑与GM4.1基本一致，但现在调用的是更新后的 _calculate_delta_X 方法)
        k = params['coefficients']
        noise_level = params.get('noise_level', 0.05)
        lr = params.get('learning_rate', 0.1)

        avg_b1_others = None
        all_b1s = [obj.b1_resource for name, obj in all_states_objects_dict.items() if name != self.name_en]
        if all_b1s: avg_b1_others = np.mean(all_b1s)

        neighbor_effects = self._calculate_neighbor_effects(params, all_states_objects_dict)

        # 判定风险是否成功 (需要一个对B1增长的预期或衡量)
        # 简化：如果本轮内部驱动的_raw_delta_b1_for_feedback为正，且H2和H3都活跃，则认为风险有一定成功可能
        # 这个判定需要在所有delta计算之前，或者传递给所有需要它的delta计算函数
        _conceptual_raw_delta_b1 = self._calculate_delta_b1(k, params, avg_b1_others, 0,
                                                            False)  # Pass False for successful_risk initially
        successful_risk_this_step = 1 if self.h2_innovation > params.get('h2_threshold_for_risk_success', 3.5) and \
                                         self.h3_risk_appetite > params.get('h3_active_threshold_for_cost', 4) and \
                                         _conceptual_raw_delta_b1 > params.get('b1_growth_threshold_for_h3', 0.02) \
            else 0

        # 重新计算 _raw_delta_b1_for_feedback，这次代入 risk_success
        _raw_delta_b1_for_feedback = self._calculate_delta_b1(k, params, avg_b1_others, 0, successful_risk_this_step)

        deltas = {
            'b1_resource': self._calculate_delta_b1(k, params, avg_b1_others, neighbor_effects.get('b1_resource', 0),
                                                    successful_risk_this_step),
            'b2_limitation': self._calculate_delta_b2(k, params, neighbor_effects.get('b2_limitation', 0),
                                                      successful_risk_this_step),
            'y1_clarity': self._calculate_delta_y1(k, params, _raw_delta_b1_for_feedback,
                                                   neighbor_effects.get('y1_clarity', 0)),
            'y2_drive': self._calculate_delta_y2(k, params, _raw_delta_b1_for_feedback,
                                                 neighbor_effects.get('y2_drive', 0), successful_risk_this_step),
            'y3_aspiration': self._calculate_delta_y3(k, params, neighbor_effects.get('y3_aspiration', 0)),
            'h1_possibilities': self._calculate_delta_h1(k, params, neighbor_effects.get('h1_possibilities', 0)),
            'h2_innovation': self._calculate_delta_h2(k, params, neighbor_effects.get('h2_innovation', 0)),
            'h3_risk_appetite': self._calculate_delta_h3(k, params, _raw_delta_b1_for_feedback,
                                                         neighbor_effects.get('h3_risk_appetite', 0),
                                                         successful_risk_this_step),
        }

        self.active_effects_log.clear()
        for key in DIM_KEYS:
            current_val = getattr(self, key)
            total_delta_from_sources = deltas.get(key, 0)
            noise = random.uniform(-noise_level, noise_level)  # Corrected noise application range
            effective_delta = total_delta_from_sources + noise
            new_val_before_event = current_val + effective_delta * lr
            final_val_after_event = new_val_before_event

            if key in active_event_effects_on_self:
                for event_eff in active_event_effects_on_self[key]:
                    self.active_effects_log.append(
                        f"事件 '{event_eff['name']}': {key} {event_eff['type']} {event_eff.get('val', 0):.2f}")
                    eff_type = event_eff.get('type')
                    eff_val = event_eff.get('val', 0)
                    if eff_type == 'add_abs':
                        final_val_after_event += eff_val
                    elif eff_type == 'set_abs':
                        final_val_after_event = eff_val
                    elif eff_type == 'multiply_abs':
                        final_val_after_event *= eff_val

            setattr(self, key, np.clip(final_val_after_event, 0, 10))

    def record_history(self, coord_type, wb, wy, wh, max_history=50):  # 无变化
        coords = self.get_coords_for_plot(coord_type, wb, wy, wh)
        if len(self.history) >= max_history: self.history.pop(0)
        self.history.append(coords)

    def clear_history(self):
        self.history = []  # 无变化

    def to_dict(self):  # 无变化
        data = {key: getattr(self, key) for key in DIM_KEYS}
        data.update({'name_zh': self.name_zh, 'name_en': self.name_en,
                     'history': self.history, 'neighbors': self.neighbors})
        return data

    @classmethod
    def from_dict(cls, data):  # 无变化 (GM4.1.1的健壮版本)
        obj = cls(
            data.get('name_zh', '未知状态'), data.get('name_en', f'Unknown_{random.randint(1000, 9999)}'),
            data.get('b1_resource', 0), data.get('b2_limitation', 0), data.get('y1_clarity', 0),
            data.get('y2_drive', 0), data.get('y3_aspiration', 0), data.get('h1_possibilities', 0),
            data.get('h2_innovation', 0), data.get('h3_risk_appetite', 0)
        )
        obj.history = data.get('history', [])
        obj.neighbors = data.get('neighbors', [])
        return obj

    def __repr__(self):
        return f"<WorldState: {self.get_display_name()}>"  # 无变化


# --- 3. EventManager 类 (与GM4.1.1版本一致，确保其逻辑正确) ---
class Event:
    # (GM4.1.1 Event Class code here - no changes in this iteration)
    def __init__(self, name, trigger_type, trigger_params, target_selector, effects, duration=1, one_time=False):
        self.name = name
        self.trigger_type = trigger_type
        self.trigger_params = trigger_params
        self.target_selector = target_selector
        self.effects = effects
        self.duration = duration
        self.one_time = one_time
        self.triggered_this_step = False

    def check_trigger(self, all_states_objects_dict, global_metrics):
        if self.trigger_type == 'probabilistic':
            return random.random() < self.trigger_params.get('prob', 0.01)
        elif self.trigger_type == 'conditional_global':
            metric_name = self.trigger_params.get('dim')
            if not metric_name: return False
            metric_val = global_metrics.get(metric_name)
            if metric_val is None: return False
            op = self.trigger_params.get('op')
            thresh_val = self.trigger_params.get('val')
            if op is None or thresh_val is None: return False
            if op == '<': return metric_val < thresh_val
            if op == '>': return metric_val > thresh_val
            if op == '==': return metric_val == thresh_val
            return False
        return False

    def select_targets(self, all_states_objects_dict):
        targets = []
        if self.target_selector == 'all':
            targets = list(all_states_objects_dict.values())
        elif isinstance(self.target_selector, dict) and 'type' in self.target_selector:
            selector_type = self.target_selector['type']
            if selector_type == 'random_n':
                n = self.target_selector.get('n', 1)
                population = list(all_states_objects_dict.values())
                if population: targets = random.sample(population, min(n, len(population)))
            elif selector_type == 'conditional_individual':
                dim = self.target_selector.get('dim')
                op = self.target_selector.get('op')
                val = self.target_selector.get('val')
                max_t = self.target_selector.get('max_targets', float('inf'))
                if not dim or not op or val is None: return []
                eligible_targets = []
                for state_obj in all_states_objects_dict.values():
                    s_val = getattr(state_obj, dim, None)
                    if s_val is not None:
                        if (op == '>' and s_val > val) or \
                                (op == '<' and s_val < val) or \
                                (op == '==' and s_val == val):
                            eligible_targets.append(state_obj)
                if eligible_targets:
                    targets = random.sample(eligible_targets, min(int(max_t), len(eligible_targets)))
        return targets

    def get_effects_for_target(self):
        processed_effects = []
        for eff_def in self.effects:
            effect_val = eff_def.get('val', 0)
            rand_range = eff_def.get('rand_range', 0)
            if rand_range > 0:
                effect_val += random.uniform(-rand_range, rand_range) * abs(effect_val)
            processed_effects.append({
                'name': self.name, 'dim': eff_def.get('dim'), 'type': eff_def.get('type'),
                'val': effect_val, 'duration': self.duration
            })
        return processed_effects


class EventManager:
    # (GM4.1.1 EventManager Class code here - no changes in this iteration)
    def __init__(self, event_definitions):
        self.event_definitions_template = event_definitions
        self.events = [Event(**ed) for ed in self.event_definitions_template]
        self.active_timed_effects = defaultdict(list)

    def reset_events(self):
        self.events = [Event(**ed) for ed in self.event_definitions_template]
        self.active_timed_effects.clear()

    def process_step(self, all_states_objects_dict, global_metrics):
        effects_to_apply_this_step = defaultdict(lambda: defaultdict(list))
        log_messages = []
        events_to_remove_indices = []
        for i, event_obj in enumerate(self.events):
            event_obj.triggered_this_step = False
            if event_obj.check_trigger(all_states_objects_dict, global_metrics):
                event_obj.triggered_this_step = True
                log_messages.append(f"事件 '{event_obj.name}' 已触发.")  # Chinese log
                targets = event_obj.select_targets(all_states_objects_dict)
                for target_state in targets:
                    for effect_data in event_obj.get_effects_for_target():
                        effect_dim = effect_data.get('dim')
                        if not effect_dim: continue
                        effects_to_apply_this_step[target_state.name_en][effect_dim].append(effect_data)
                        if effect_data['duration'] > 1:
                            self.active_timed_effects[target_state.name_en].append({
                                'effect_data': effect_data, 'remaining_duration': effect_data['duration'] - 1
                            })
                if event_obj.one_time and event_obj.triggered_this_step:
                    events_to_remove_indices.append(i)
        for index_to_remove in sorted(events_to_remove_indices, reverse=True):
            del self.events[index_to_remove]
        new_active_timed_effects = defaultdict(list)
        for state_name, timed_effects_list in self.active_timed_effects.items():
            for timed_effect in timed_effects_list:
                effect_data = timed_effect['effect_data']
                effect_dim = effect_data.get('dim')
                if not effect_dim: continue
                if timed_effect['remaining_duration'] > 0:
                    effects_to_apply_this_step[state_name][effect_dim].append(effect_data)
                    timed_effect['remaining_duration'] -= 1
                    if timed_effect['remaining_duration'] > 0:
                        new_active_timed_effects[state_name].append(timed_effect)
                    else:
                        log_messages.append(
                            f"定时效果 '{effect_data.get('name', '未命名效果')}' 在 {state_name} 上已到期.")  # Chinese log
        self.active_timed_effects = new_active_timed_effects
        return effects_to_apply_this_step, log_messages


# --- 4. 初始化数据 (GM4.2 - 参数调整) ---
initial_states_templates_gm42 = [  # 与 GM4.1.1 结构相同，但用于 GM4.2
    {'name_zh': "抑郁状态", 'name_en': "Depressed", 'b1_res': 2.5, 'b2_lim': 7.5, 'y1_cla': 2, 'y2_dri': 1,
     'y3_asp': 2.5, 'h1_pos': 1.5, 'h2_inn': 1, 'h3_ris': 1},
    {'name_zh': "创业先锋", 'name_en': "Entrepreneur", 'b1_res': 6, 'b2_lim': 3.5, 'y1_cla': 7, 'y2_dri': 8,
     'y3_asp': 8.5, 'h1_pos': 7, 'h2_inn': 8, 'h3_ris': 7.5},
    {'name_zh': "平均状态", 'name_en': "Average", 'b1_res': 4.5, 'b2_lim': 5, 'y1_cla': 5, 'y2_dri': 4.5, 'y3_asp': 5,
     'h1_pos': 4.5, 'h2_inn': 4, 'h3_ris': 4},
    # ... 其他状态点可以微调初始值 ...
    {'name_zh': "挣扎艺术家", 'name_en': "ArtistStruggling", 'b1_res': 3, 'b2_lim': 6.5, 'y1_cla': 6.5, 'y2_dri': 7,
     'y3_asp': 8, 'h1_pos': 2.5, 'h2_inn': 3.5, 'h3_ris': 2},
    {'name_zh': "稳定学者", 'name_en': "ScholarStable", 'b1_res': 5.5, 'b2_lim': 4, 'y1_cla': 7.5, 'y2_dri': 6,
     'y3_asp': 7, 'h1_pos': 5.5, 'h2_inn': 6, 'h3_ris': 3.5},
    {'name_zh': "探索青年", 'name_en': "YouthExplorer", 'b1_res': 5, 'b2_lim': 4.5, 'y1_cla': 5.5, 'y2_dri': 6.5,
     'y3_asp': 6, 'h1_pos': 6, 'h2_inn': 5.5, 'h3_ris': 6.5}
]

initial_states_obj_list_gm42 = [WorldState(**s_data) for s_data in initial_states_templates_gm42]

# 正确定义 neighbor_config_gm42
neighbor_config_gm42 = {
    "Entrepreneur": ["ArtistStruggling", "YouthExplorer"],
    "ArtistStruggling": ["Entrepreneur"],
    "YouthExplorer": ["Entrepreneur", "Average"],
    "Depressed": ["Average"],
    "Average": ["Depressed", "YouthExplorer", "ScholarStable"]
    # 可以根据 GM4.2 的需要调整这个配置
}

for state_obj in initial_states_obj_list_gm42:  # 现在这里使用的是正确的 neighbor_config_gm42
    state_obj.neighbors = neighbor_config_gm42.get(state_obj.name_en, [])

initial_world_states_store_data_gm42 = {s.name_en: s.to_dict() for s in initial_states_obj_list_gm42}

default_evolution_params_gm42 = {
    # ... (参数定义如 GM4.2 所示) ...
    'learning_rate': 0.06,  # 降低学习率，减缓整体变化
    'noise_level': 0.025,  # 适中噪声
    'base_consumption': 0.03,  # 提高基础资源消耗
    'b1_growth_threshold_for_y1': 0.05,  # B1增长一点才能鼓舞Y1
    'b1_growth_threshold_for_h3': 0.03,  # B1增长一点才能鼓励H3
    'h2_threshold_for_risk_success': 4.0,  # H2需要较高才能认为风险尝试是“有准备的”
    'h3_active_threshold_for_cost': 3.5,  # H3高于此值且失败，会产生额外成本/限制
    'b2_random_event_chance': 0.035,
    'y1_gap_threshold_for_loss': 3.5,  # Y1因理想现实差距过大而受损的阈值
    'y3_reality_gap_threshold': 4.5,  # Y3因现实过低而被迫降低的阈值

    'coefficients': {
        # B1 (资源)
        'b1_from_h2': 0.10, 'b1_from_y2': 0.09, 'b1_loss_b2': 0.12,  # 降低H2/Y2对B1的直接贡献，提高B2的损失
        'b1_cost_h2_activity': 0.008, 'b1_cost_y2_sustain': 0.005,  # 增加创新和驱动的成本
        'b1_cost_failed_risk': 0.10,  # 风险失败的资源成本
        'b1_social_pressure': 0.003,  # 减弱全局平均资源的直接影响
        # B2 (限制)
        'b2_reduce_y2': 0.07, 'b2_reduce_h2': 0.06, 'b2_random_factor': 0.12,
        'b2_from_failed_risk': 0.06,  # 风险失败增加限制
        # Y1 (清晰度)
        'y1_from_b1_growth': 0.18, 'y1_loss_b2': 0.11,
        'y1_loss_aspiration_gap': 0.015,  # 理想现实差距对Y1的负面影响
        # Y2 (驱动力)
        'y2_from_y1': 0.08, 'y2_from_y3': 0.03, 'y2_from_b1_growth': 0.11, 'y2_loss_b2': 0.09,
        'y2_loss_low_y1': 0.22, 'y2_loss_failed_risk': 0.08,  # 风险失败打击驱动力
        # Y3 (理想)
        'y3_adjust_y2': 0.07, 'y3_boost_y1': 0.01, 'y3_support_b1': 0.008,
        'y3_loss_reality_gap': 0.025,  # 现实过低对理想的打击
        # H1 (可能性)
        'h1_from_b1': 0.10, 'h1_from_h2': 0.12, 'h1_loss_b2': 0.13,
        'h1_loss_low_y_factor': 0.015,  # 低Y1/Y2对H1的负面影响
        # H2 (创新)
        'h2_from_y2': 0.11, 'h2_from_h3': 0.08, 'h2_decay_no_practice': 0.06,  # 创新衰减加快
        # H3 (风险承受)
        'h3_from_risk_success': 0.18, 'h3_from_y1': 0.06, 'h3_from_y2': 0.09, 'h3_loss_b2': 0.11,
        'h3_penalty_failed_risk': 0.10,  # 风险失败对H3的惩罚

        # 社会交互 (调整以避免过度趋同或竞争)
        'social_interactions': {
            'y2_contagion_factor': 0.01, 'b1_comp_diff_thresh': 1.2,
            'b1_comp_loss_factor': 0.012, 'b1_coop_diff_thresh': 0.6,
            'b1_coop_gain_factor': 0.0015, 'y1_alignment_factor': 0.005,
        }
    },
    'plot_weights': {'b': (0.6, 0.4), 'y': (0.4, 0.4, 0.2), 'h': (0.4, 0.4, 0.2)}
}

event_definitions_gm42 = [
    {
        'name': "经济小幅提振 (Global Stimulus)", 'trigger_type': "probabilistic", 'trigger_params': {'prob': 0.02},
        'target_selector': "all",
        'effects': [{'dim': 'b1_resource', 'type': 'add_abs', 'val': 0.5, 'rand_range': 0.2}],
        'duration': 1, 'one_time': False  # 确保 one_time 根据需要设置
    },
    {
        'name': "科技灵感迸发 (Innovation Spark)", 'trigger_type': "probabilistic", 'trigger_params': {'prob': 0.015},
        'target_selector': {'type': 'random_n', 'n': 2},
        'effects': [{'dim': 'h2_innovation', 'type': 'add_abs', 'val': 1.5, 'rand_range': 0.3},
                    {'dim': 'h1_possibilities', 'type': 'add_abs', 'val': 1.0, 'rand_range': 0.2}],
        'duration': 3, 'one_time': False
    },
    {
        'name': "突发困境 (Sudden Setback)", 'trigger_type': "probabilistic", 'trigger_params': {'prob': 0.025},
        'target_selector': {'type': 'random_n', 'n': 1},
        'effects': [{'dim': 'b2_limitation', 'type': 'add_abs', 'val': 2.0, 'rand_range': 0.25},
                    {'dim': 'y2_drive', 'type': 'add_abs', 'val': -1.0, 'rand_range': 0.1}],
        'duration': 1, 'one_time': False
    },
    {
        'name': "价值观觉醒 (挣扎者)",
        'trigger_type': "conditional_global",
        'trigger_params': {'dim': 'avg_y1_clarity', 'op': '<', 'val': 4.5},
        'target_selector': {'type': 'conditional_individual', 'dim': 'y1_clarity', 'op': '<', 'val': 3.0,
                            'max_targets': 1},
        'effects': [{'dim': 'y1_clarity', 'type': 'set_abs', 'val': 5.0, 'rand_range': 0.1},
                    {'dim': 'y3_aspiration', 'type': 'add_abs', 'val': 1.0}],
        'duration': 1, 'one_time': True  # 这个事件是一次性的
    }
    # 可以根据 GM4.2 的需要添加或修改更多事件
]
event_manager = EventManager(event_definitions_gm42)  # 现在使用正确的版本

# --- 5. Dash App Layout (使用 _gm42 后缀的初始数据) ---
app = dash.Dash(__name__, suppress_callback_exceptions=True)
app.title = "三世界理论 - v4.2 (反馈增强版)"
# (Layout 与 GM4.1.1 结构相同, 仅修改引用的初始数据变量名)
# ... (Layout code from GM4.1.1, but ensure default values for sliders etc.
#      use default_evolution_params_gm42 and initial_world_states_store_data_gm42)
app.layout = html.Div([
    html.H1(app.title, style={'textAlign': 'center', 'color': '#2c3e50'}),
    dcc.Store(id='world-states-store', data=initial_world_states_store_data_gm42),  # Use new initial data
    dcc.Store(id='selected-point-id-store', data=None),
    dcc.Store(id='evolution-params-store', data=default_evolution_params_gm42),  # Use new initial data
    dcc.Store(id='event-log-store', data=[]),
    dcc.Interval(id='evolution-interval', interval=1000, n_intervals=0, disabled=True),
    html.Div([
        html.Div([  # Left Panel
            html.H3("控制面板", style={'textAlign': 'center', 'borderBottom': '1px solid #ccc', 'paddingBottom': '10px',
                                       'marginBottom': '15px'}),
            html.Label("坐标系类型:", style={'fontWeight': 'bold'}),
            dcc.Dropdown(id='coord-type-dropdown',
                         options=[{'label': '简化坐标 (B1, Y2, H1)', 'value': 'simplified'},
                                  {'label': '综合坐标 (B_c, Y_c, H_c)', 'value': 'composite'}],
                         value='simplified', clearable=False, style={'marginBottom': '15px'}),
            html.Label("选择状态点:", style={'fontWeight': 'bold'}),
            dcc.Dropdown(id='point-selector-dropdown', clearable=False, style={'marginBottom': '15px'}),
            html.Div(id='edit-panel-div', style={'padding': '10px', 'border': '1px solid #ddd', 'borderRadius': '5px',
                                                 'marginBottom': '15px', 'backgroundColor': '#f9f9f9'}),
            html.Div(id='neighbor-info-div', style={'fontSize': 'small', 'marginBottom': '15px', 'padding': '5px',
                                                    'border': '1px dashed #ccc'}),
            html.H4("动态演化控制", style={'marginTop': '20px', 'borderTop': '1px solid #ccc', 'paddingTop': '15px'}),
            html.Div([
                html.Button('开始/暂停演化', id='toggle-evolution-button', n_clicks=0, className='button-primary',
                            style={'marginRight': '10px'}),
                html.Button('演化一步', id='step-evolution-button', n_clicks=0, style={'marginRight': '10px'}),
                html.Button('重置所有状态', id='reset-states-button', n_clicks=0, className='button-danger'),
            ], style={'marginBottom': '10px'}),
            html.Div([
                html.Label("演化速度 (ms/步): ", style={'display': 'inline-block', 'marginRight': '5px'}),
                dcc.Input(id='evolution-interval-input', type='number', value=1200, min=100, step=100,
                          style={'width': '80px'})], style={'marginTop': '10px'}),  # Default speed slower
            html.Label("学习率 (lr):", style={'fontWeight': 'bold', 'marginTop': '10px'}),
            dcc.Slider(id='lr-slider', min=0.01, max=0.15, step=0.005,
                       value=default_evolution_params_gm42['learning_rate'],  # Use new default
                       marks={i / 100: str(i / 100) for i in range(1, 16, 2)},
                       tooltip={"placement": "bottom", "always_visible": True}),
            html.Label("噪声水平:", style={'fontWeight': 'bold', 'marginTop': '10px'}),
            dcc.Slider(id='noise-slider', min=0, max=0.05, step=0.001,
                       value=default_evolution_params_gm42['noise_level'],  # Use new default
                       marks={i / 1000: str(i / 1000) for i in range(0, 51, 10)},
                       tooltip={"placement": "bottom", "always_visible": True}),
            html.Div(id='n-intervals-display', style={'marginTop': '15px', 'fontSize': 'small', 'color': 'gray'})
        ], style={'width': '30%', 'float': 'left', 'padding': '20px', 'boxSizing': 'border-box',
                  'backgroundColor': '#f0f4f8', 'borderRight': '1px solid #ccc', 'maxHeight': '90vh',
                  'overflowY': 'auto'}),
        html.Div([  # Right Panel
            dcc.Graph(id='main-3d-scatter-plot', style={'height': '70vh'}),
            html.Div([
                html.H5("事件与状态日志:", style={'marginTop': '10px', 'marginBottom': '5px'}),
                dcc.Textarea(id='event-log-textarea', value="", readOnly=True,
                             style={'width': '100%', 'height': '12vh', 'fontSize': 'small', 'border': '1px solid #ddd',
                                    'backgroundColor': '#fafafa'})])
        ], style={'width': '70%', 'float': 'right', 'padding': '10px', 'boxSizing': 'border-box'})]),
    html.Div(style={'clear': 'both'}),
    html.Footer(f"三世界理论模型 - v4.2 (反馈增强) - {time.strftime('%Y-%m-%d %H:%M:%S')}",
                style={'textAlign': 'center', 'marginTop': '20px', 'padding': '10px', 'fontSize': 'x-small',
                       'color': '#888'})
], style={'fontFamily': "'Segoe UI', Tahoma, Geneva, Verdana, sans-serif", 'maxWidth': '1800px', 'margin': 'auto',
          'backgroundColor': '#e9ecef'})


# --- 6. Callbacks (使用 _gm42 后缀, 引用新的初始数据和参数) ---
# (Callbacks from GM4.1.1 are largely reusable, just ensure they use the _gm42 suffixed global variables
# for default parameters and initial data if they need to reference them directly,
# e.g., in reset_all_states or for default plot_weights if evo_params_store is faulty.)

@app.callback(
    Output('neighbor-info-div', 'children'),
    [Input('point-selector-dropdown', 'value'), Input('world-states-store', 'data')]
)
def display_neighbor_info_gm42(selected_id, states_data_json):
    if not selected_id or not states_data_json or selected_id not in states_data_json:
        return "选择一个状态点查看邻居信息。"
    state_data = states_data_json[selected_id]
    neighbors = state_data.get('neighbors', [])
    if not neighbors: return f"{state_data.get('name_zh', 'N/A')} 没有定义邻居。"
    neighbor_display_names = [
        states_data_json[n_id].get('name_zh', n_id) if n_id in states_data_json else f"{n_id} (数据丢失)" for n_id in
        neighbors]
    return f"{state_data.get('name_zh', 'N/A')} 的邻居: {', '.join(neighbor_display_names)}"


@app.callback(
    [Output('point-selector-dropdown', 'options'), Output('point-selector-dropdown', 'value')],
    [Input('world-states-store', 'data')], [State('selected-point-id-store', 'data')]
)
def update_point_selector_gm42(states_data_json, selected_point_id):
    if not states_data_json: return [], None
    options = [{'label': f"{s_data['name_zh']} ({s_data['name_en']})", 'value': s_data['name_en']}
               for s_id, s_data in states_data_json.items() if
               isinstance(s_data, dict) and 'name_en' in s_data and 'name_zh' in s_data]
    valid_ids = [opt['value'] for opt in options]
    current_value = selected_point_id if selected_point_id in valid_ids else (options[0]['value'] if options else None)
    return options, current_value


@app.callback(
    Output('edit-panel-div', 'children'),
    Input('point-selector-dropdown', 'value'), State('world-states-store', 'data')
)
def update_edit_panel_gm42(selected_id, states_data_json):
    if not selected_id or not states_data_json or selected_id not in states_data_json:
        return html.P("请选择一个状态点进行编辑。", style={'color': 'orange'})
    state_data_for_selected_id = states_data_json[selected_id]
    if not isinstance(state_data_for_selected_id, dict):
        return html.P(f"加载状态点 '{selected_id}' 数据格式错误。", style={'color': 'red'})
    try:
        state_obj = WorldState.from_dict(state_data_for_selected_id)
    except Exception as e:
        return html.P(f"加载状态点 '{selected_id}' 数据时出错: {e}", style={'color': 'red'})
    panel_children = [
        html.H4(f"编辑: {state_obj.get_display_name()}", style={'marginTop': '0', 'marginBottom': '10px'})]
    for key_dim in DIM_KEYS:
        label_zh = DIMENSION_LABELS_MAP_ZH.get(key_dim, key_dim.replace('_', ' ').title())
        current_dim_value = getattr(state_obj, key_dim, 0)
        panel_children.extend([
            html.Label(label_zh,
                       style={'fontWeight': 'normal', 'fontSize': 'small', 'display': 'block', 'marginBottom': '2px'}),
            dcc.Slider(id={'type': 'dim-slider', 'index': key_dim}, min=0, max=10, step=0.1, value=current_dim_value,
                       marks={i: str(i) for i in range(0, 11, 2)},
                       tooltip={"placement": "bottom", "always_visible": False},
                       className='dim-slider-style'),
            html.Div(style={'marginBottom': '5px'})])
    return panel_children


@app.callback(
    Output('world-states-store', 'data', allow_duplicate=True),
    Input({'type': 'dim-slider', 'index': dash.ALL}, 'value'),
    State({'type': 'dim-slider', 'index': dash.ALL}, 'id'),
    State('point-selector-dropdown', 'value'), State('world-states-store', 'data'),
    prevent_initial_call=True
)
def update_state_from_sliders_gm42(slider_values, slider_ids_obj_list, selected_id, states_data_json):
    ctx = callback_context
    if not ctx.triggered or not selected_id or not states_data_json or selected_id not in states_data_json: return no_update
    triggered_input = ctx.triggered[0]
    slider_key, slider_value = None, None
    if isinstance(ctx.triggered_id, dict) and 'index' in ctx.triggered_id:
        slider_key = ctx.triggered_id['index']
        slider_value = triggered_input['value']
    elif slider_ids_obj_list and slider_values:
        prop_id_str = triggered_input['prop_id']
        for i, id_obj in enumerate(slider_ids_obj_list):
            if isinstance(id_obj, dict) and json.dumps(id_obj, sort_keys=True) in prop_id_str:
                slider_key = id_obj.get('index')
                slider_value = slider_values[i]
                break
    if not slider_key or slider_value is None or slider_key not in DIM_KEYS:
        return no_update
    updated_states_data = states_data_json.copy()
    point_to_update_dict = updated_states_data[selected_id].copy()
    point_to_update_dict[slider_key] = float(slider_value)
    updated_states_data[selected_id] = point_to_update_dict
    return updated_states_data


@app.callback(Output('selected-point-id-store', 'data'), Input('point-selector-dropdown', 'value'))
def update_selected_point_id_store_val_gm42(selected_id): return selected_id


@app.callback(
    [Output('evolution-interval', 'disabled'), Output('toggle-evolution-button', 'children')],
    [Input('toggle-evolution-button', 'n_clicks')], [State('evolution-interval', 'disabled')]
)
def toggle_evolution_gm42(n_clicks, current_disabled_state):
    if n_clicks == 0: return True, '开始演化'
    is_now_disabled = not current_disabled_state
    return is_now_disabled, '暂停演化' if not is_now_disabled else '开始演化'


@app.callback(Output('evolution-interval', 'interval'), Input('evolution-interval-input', 'value'))
def update_evolution_interval_time_gm42(value): return int(value) if value and int(value) >= 100 else 1000


@app.callback(
    Output('evolution-params-store', 'data'),
    [Input('lr-slider', 'value'), Input('noise-slider', 'value')],
    [State('evolution-params-store', 'data')]
)
def update_evolution_hyperparams_gm42(lr, noise, current_params_json):
    if lr is None or noise is None: raise PreventUpdate
    new_params = json.loads(json.dumps(current_params_json))
    new_params['learning_rate'] = float(lr)
    new_params['noise_level'] = float(noise)
    return new_params


@app.callback(
    [Output('world-states-store', 'data'),
     Output('n-intervals-display', 'children'),
     Output('event-log-store', 'data')],
    [Input('evolution-interval', 'n_intervals'), Input('step-evolution-button', 'n_clicks')],
    [State('world-states-store', 'data'), State('evolution-params-store', 'data'),
     State('evolution-interval', 'disabled'), State('coord-type-dropdown', 'value'),
     State('event-log-store', 'data')]
)
def run_evolution_step_advanced_gm42(n_intervals_auto, n_clicks_manual, states_data_json,
                                     evo_params_json, interval_disabled, coord_type, current_event_log_list):
    ctx = callback_context
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None
    if not triggered_id or (triggered_id == 'evolution-interval' and interval_disabled) or \
            not states_data_json or not evo_params_json:
        raise PreventUpdate

    all_states_objects_dict = {name: WorldState.from_dict(data) for name, data in states_data_json.items()}
    global_metrics = {
        'avg_b1_resource': np.mean(
            [s.b1_resource for s in all_states_objects_dict.values() if s]) if all_states_objects_dict else 0,
        'avg_y1_clarity': np.mean(
            [s.y1_clarity for s in all_states_objects_dict.values() if s]) if all_states_objects_dict else 0,
    }
    effects_to_apply_by_event, event_trigger_messages = event_manager.process_step(all_states_objects_dict,
                                                                                   global_metrics)
    updated_states_data_json_out = {}
    all_step_internal_event_logs = []

    for state_name_en, state_obj in all_states_objects_dict.items():
        active_event_effects_on_this_state = effects_to_apply_by_event.get(state_name_en, {})
        try:
            state_obj.evolve(evo_params_json, all_states_objects_dict, active_event_effects_on_this_state)
            plot_weights = evo_params_json.get('plot_weights', default_evolution_params_gm42['plot_weights'])
            state_obj.record_history(coord_type, plot_weights['b'], plot_weights['y'], plot_weights['h'])
            updated_states_data_json_out[state_name_en] = state_obj.to_dict()
            if state_obj.active_effects_log:
                all_step_internal_event_logs.extend(
                    [f"状态 '{state_obj.name_zh}': {log}" for log in state_obj.active_effects_log])
        except Exception as e:
            print(f"演化状态 {state_name_en} 出错: {e}")  # Chinese log
            updated_states_data_json_out[state_name_en] = states_data_json[state_name_en]

    if not isinstance(current_event_log_list, list): current_event_log_list = []
    new_log_entries = []
    if event_trigger_messages: new_log_entries.extend(event_trigger_messages)
    if all_step_internal_event_logs: new_log_entries.extend(all_step_internal_event_logs)
    if new_log_entries:
        log_timestamp = f"--- 步骤 {n_intervals_auto if triggered_id == 'evolution-interval' else '(手动)'} ({time.strftime('%H:%M:%S')}) ---"  # Chinese log
        combined_log_list = current_event_log_list + [log_timestamp] + new_log_entries
    else:
        combined_log_list = current_event_log_list
    final_log_list_for_store = combined_log_list[-MAX_LOG_LINES:]

    step_info = f"自动迭代: {n_intervals_auto}" if triggered_id == 'evolution-interval' else f"手动步进 (总: {n_clicks_manual})"
    return updated_states_data_json_out, step_info, final_log_list_for_store


@app.callback(Output('event-log-textarea', 'value'), Input('event-log-store', 'data'))
def update_event_log_display_gm42(log_data_list):
    if isinstance(log_data_list, list): return "\n".join(log_data_list)
    return "事件日志为空或格式错误."


@app.callback(
    [Output('world-states-store', 'data', allow_duplicate=True),
     Output('evolution-interval', 'n_intervals', allow_duplicate=True),
     Output('event-log-store', 'data', allow_duplicate=True)],
    [Input('reset-states-button', 'n_clicks')], prevent_initial_call=True
)
def reset_all_states_gm42(n_clicks):
    if n_clicks is None or n_clicks == 0: raise PreventUpdate

    fresh_initial_data_store_local = {}
    temp_states_list_local = [WorldState(**s_dict_template) for s_dict_template in initial_states_templates_gm42]

    for state_obj_local in temp_states_list_local:
        state_obj_local.neighbors = neighbor_config_gm42.get(state_obj_local.name_en, [])
        fresh_initial_data_store_local[state_obj_local.name_en] = state_obj_local.to_dict()

    event_manager.reset_events()
    return fresh_initial_data_store_local, 0, ["状态和事件已重置."]


@app.callback(
    Output('main-3d-scatter-plot', 'figure'),
    [Input('world-states-store', 'data'), Input('coord-type-dropdown', 'value')],
    [State('evolution-params-store', 'data')]
)
def update_3d_scatter_plot_gm42(states_data_json, coord_type, evo_params_json):
    # (Using corrected version from previous response)
    if not states_data_json:
        return go.Figure(layout=go.Layout(title="数据加载中...",
                                          scene=dict(xaxis=dict(range=[0, 10]), yaxis=dict(range=[0, 10]),
                                                     zaxis=dict(range=[0, 10]), aspectmode='cube')))
    traces = []
    plot_weights = default_evolution_params_gm42['plot_weights']
    if evo_params_json and 'plot_weights' in evo_params_json and \
            isinstance(evo_params_json['plot_weights'], dict) and \
            all(k in evo_params_json['plot_weights'] for k in ['b', 'y', 'h']):
        plot_weights = evo_params_json['plot_weights']
    state_items = list(states_data_json.items())
    for i, (state_id, state_dict) in enumerate(state_items):
        if not isinstance(state_dict, dict): continue
        try:
            state_obj = WorldState.from_dict(state_dict)
            current_coords = state_obj.get_coords_for_plot(
                coord_type, plot_weights.get('b'), plot_weights.get('y'), plot_weights.get('h'))
        except Exception as e:
            continue
        hover_text_parts = [f"<b>{state_obj.get_display_name()}</b>"]
        for key_dim in DIM_KEYS:
            label_short = DIMENSION_LABELS_MAP_ZH.get(key_dim, key_dim).split(': ')[-1].split(' ')[0]
            hover_text_parts.append(f"{label_short}: {getattr(state_obj, key_dim, 'N/A'):.1f}")
        coord_type_label = '简化' if coord_type == 'simplified' else '综合'
        hover_text_parts.append(
            f"--- Plot ({coord_type_label}) ---<br>X:{current_coords[0]:.2f}, Y:{current_coords[1]:.2f}, Z:{current_coords[2]:.2f}")
        marker_color_value = i
        color_is_numeric_for_scale = False
        if len(current_coords) == 3 and isinstance(current_coords[2], (int, float)):
            marker_color_value = current_coords[2]
            color_is_numeric_for_scale = True
        show_this_colorbar_for_this_trace = (i == 0 and color_is_numeric_for_scale)
        traces.append(go.Scatter3d(
            x=[current_coords[0]], y=[current_coords[1]], z=[current_coords[2]], mode='markers+text',
            text=[state_obj.name_zh], textfont=dict(size=10, color="#1f77b4"), textposition='top center',
            marker=dict(size=11, opacity=0.9, color=marker_color_value, colorscale='Plasma',
                        showscale=show_this_colorbar_for_this_trace,
                        colorbar=dict(title=f"Z:{AXIS_LABELS_ZH.get(coord_type, {}).get('z', 'Z')}", thickness=15,
                                      x=1.05) if show_this_colorbar_for_this_trace else None),
            hoverinfo='text', hovertext=["<br>".join(hover_text_parts)], name=state_obj.get_display_name(),
            legendgroup=state_obj.name_en))
        if state_obj.history and len(state_obj.history) > 1:
            valid_history = [h for h in state_obj.history if isinstance(h, (list, tuple)) and len(h) == 3]
            if len(valid_history) > 1:
                hist_x, hist_y, hist_z = zip(*valid_history)
                traces.append(go.Scatter3d(
                    x=list(hist_x), y=list(hist_y), z=list(hist_z), mode='lines',
                    line=dict(width=2.5, color=f'rgba(120,120,120,0.35)'), hoverinfo='skip',
                    name=f"{state_obj.name_zh} 历史", showlegend=False, legendgroup=state_obj.name_en))
    current_axis_labels_dict = AXIS_LABELS_ZH.get(coord_type, {'x': 'X', 'y': 'Y', 'z': 'Z'})
    fig = go.Figure(data=traces)
    fig.update_layout(
        margin=dict(l=0, r=0, b=0, t=50),
        title=dict(text=f"三世界坐标 ({'Simp.' if coord_type == 'simplified' else 'Comp.'}) - 高级演化", x=0.5, y=0.97,
                   font=dict(size=16)),
        scene=dict(xaxis_title=current_axis_labels_dict.get('x', 'X轴'),
                   yaxis_title=current_axis_labels_dict.get('y', 'Y轴'),
                   zaxis_title=current_axis_labels_dict.get('z', 'Z轴'),
                   xaxis=dict(range=[0, 10], autorange=False, nticks=6, backgroundcolor="rgb(230,230,230)",
                              gridcolor="white", zerolinecolor="white"),
                   yaxis=dict(range=[0, 10], autorange=False, nticks=6, backgroundcolor="rgb(230,230,230)",
                              gridcolor="white", zerolinecolor="white"),
                   zaxis=dict(range=[0, 10], autorange=False, nticks=6, backgroundcolor="rgb(230,230,230)",
                              gridcolor="white", zerolinecolor="white"),
                   aspectmode='cube', camera=dict(eye=dict(x=1.7, y=1.7, z=0.6))),
        legend=dict(orientation="v", yanchor="top", y=0.95, xanchor="left", x=0.01, bgcolor='rgba(250,250,250,0.75)',
                    bordercolor='#ccc', borderwidth=1))
    return fig


if __name__ == '__main__':
    app.run(debug=True, port=8057)  # New port for GM4.2
