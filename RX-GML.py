# --- GM4.5.2 (Corrected References, Risk Logic, Param Access) ---
import dash
from dash import dcc, html, Input, Output, State, callback_context, no_update
from dash.exceptions import PreventUpdate
import plotly.graph_objects as go
import numpy as np
import random
import json
import time
from collections import defaultdict

# --- 1. 全局常量和辅助函数 ---
DIM_KEYS = ['b1_resource', 'b2_limitation', 'y1_clarity', 'y2_drive', 'y3_aspiration',
            'h1_possibilities', 'h2_innovation', 'h3_risk_appetite',
            's1_trustworthiness', 's2_reputation']

DIMENSION_LABELS_MAP_ZH = {
    'b1_resource': "本然 B1: 资源/条件", 'b2_limitation': "本然 B2: 限制/障碍",
    'y1_clarity': "应然 Y1: 价值清晰度", 'y2_drive': "应然 Y2: 驱动力", 'y3_aspiration': "应然 Y3: 理想高度",
    'h1_possibilities': "或然 H1: 可能性广度", 'h2_innovation': "或然 H2: 创新能力",
    'h3_risk_appetite': "或然 H3: 风险承受",
    's1_trustworthiness': "社交 S1: 可信度", 's2_reputation': "社交 S2: 声望"
}

AXIS_LABELS_ZH = {
    'simplified': {'x': '本然 B1 (资源)', 'y': '应然 Y2 (驱动)', 'z': '或然 H1 (可能)'},
    'composite': {'x': '本然 (综合)', 'y': '应然 (综合)', 'z': '或然 (综合)'}
}
MAX_LOG_LINES = 150


def sigmoid(x, k=1, x0=5):
    if isinstance(x, (np.ndarray, list)): x = np.array(x, dtype=float)
    x_clipped = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-k * (x_clipped - x0)))


def scale_value(value, old_min=0, old_max=10, new_min=0, new_max=1):
    if old_max == old_min: return new_min
    return new_min + (value - old_min) * (new_max - new_min) / (old_max - old_min)


# --- 2. WorldState 类 ---
class WorldState:
    def __init__(self, name_zh, name_en, b1_res, b2_lim, y1_cla, y2_dri, y3_asp,
                 h1_pos, h2_inn, h3_ris, s1_tru=5.0, s2_rep=3.0):
        self.name_zh = name_zh;
        self.name_en = name_en
        dim_values = {
            'b1_resource': b1_res, 'b2_limitation': b2_lim, 'y1_clarity': y1_cla,
            'y2_drive': y2_dri, 'y3_aspiration': y3_asp, 'h1_possibilities': h1_pos,
            'h2_innovation': h2_inn, 'h3_risk_appetite': h3_ris,
            's1_trustworthiness': s1_tru, 's2_reputation': s2_rep
        }
        for key in DIM_KEYS:
            setattr(self, key, np.clip(float(dim_values.get(key, 0)), 0, 10))
        self.history = []
        self.neighbors = []
        self.trust_levels = defaultdict(lambda: 5.0)
        self.active_effects_log = []
        self.last_risk_outcome_factor = 0.0  # Stores the outcome of the PREVIOUS step's risk project

    def get_display_name(self):
        return f"{self.name_zh} ({self.name_en})"

    def get_coords_for_plot(self, coord_type='simplified', wb=(0.6, 0.4), wy=(0.4, 0.4, 0.2), wh=(0.4, 0.4, 0.2)):
        if coord_type == 'simplified':
            return (self.b1_resource, self.y2_drive, self.h1_possibilities)
        elif coord_type == 'composite':
            b_comp = wb[0] * self.b1_resource - wb[1] * self.b2_limitation
            y_comp = wy[0] * self.y1_clarity + wy[1] * self.y2_drive + wy[2] * self.y3_aspiration
            h_comp = wh[0] * self.h1_possibilities + wh[1] * self.h2_innovation + wh[2] * self.h3_risk_appetite
            return (np.clip(b_comp, 0, 10), np.clip(y_comp, 0, 10), np.clip(h_comp, 0, 10))
        return (self.b1_resource, self.y1_clarity, self.h1_possibilities)

    def _apply_boundary_effect(self, current_value, delta_value, min_val=0, max_val=10, boundary_threshold=0.5):
        if current_value <= min_val + boundary_threshold and delta_value < 0:
            return delta_value * scale_value(current_value, min_val, min_val + boundary_threshold, 0, 1)
        elif current_value >= max_val - boundary_threshold and delta_value > 0:
            return delta_value * scale_value(current_value, max_val - boundary_threshold, max_val, 1, 0)
        return delta_value

    # --- Delta Calculation Methods (Using current_risk_outcome_this_step) ---
    def _calculate_delta_b1(self, k, params, avg_b1_others, neighbor_effect_b1, risk_project_b1_change_this_step):
        effect_h2 = k.get('b1_from_h2', 0) * sigmoid(self.h2_innovation, k=0.7, x0=5)
        effect_y2 = k.get('b1_from_y2', 0) * scale_value(self.y2_drive, 0, 10, 0.3, 1)
        loss_b2 = k.get('b1_loss_b2', 0) * (self.b2_limitation / 10) ** 1.8
        cost_h2_activity = k.get('b1_cost_h2_activity', 0) * self.h2_innovation
        cost_y2_sustain = k.get('b1_cost_y2_sustain', 0) * self.y2_drive
        social_pressure_b1 = 0
        if avg_b1_others is not None:
            social_pressure_b1 = k.get('b1_social_pressure', 0) * np.clip(avg_b1_others - self.b1_resource, -3, 3)
        reputation_bonus_b1 = k.get('b1_from_s2_reputation', 0) * self.s2_reputation
        delta = (effect_h2 + effect_y2 - loss_b2 - params.get('base_consumption',
                                                              0.01) - cost_h2_activity - cost_y2_sustain +
                 risk_project_b1_change_this_step + social_pressure_b1 + neighbor_effect_b1 + reputation_bonus_b1)
        return self._apply_boundary_effect(self.b1_resource, delta)

    def _calculate_delta_b2(self, k, params, neighbor_effect_b2, risk_project_b2_change_this_step):
        reduction_y2 = k.get('b2_reduce_y2', 0) * sigmoid(self.y2_drive, k=0.8, x0=3)
        reduction_h2 = k.get('b2_reduce_h2', 0) * sigmoid(self.h2_innovation, k=0.8, x0=3)
        random_event_b2 = 0
        if random.random() < params.get('b2_random_event_chance', 0.04):
            random_event_b2 = random.uniform(0, 0.8) * k.get('b2_random_factor', 0.15)
        delta = -(reduction_y2 + reduction_h2) + random_event_b2 + risk_project_b2_change_this_step + neighbor_effect_b2
        return self._apply_boundary_effect(self.b2_limitation, delta)

    def _calculate_delta_y1(self, k, params, neighbor_effect_y1, current_risk_outcome_this_step):
        effect_experience_validation = 0
        if current_risk_outcome_this_step > params.get('risk_success_return_threshold_for_y1', 0.03):
            effect_experience_validation = k.get('y1_from_success_validation', 0) * sigmoid(
                current_risk_outcome_this_step, k=5, x0=0.1)
        elif current_risk_outcome_this_step < -params.get('risk_failure_loss_threshold_for_y1', 0.03):
            effect_experience_validation = k.get('y1_from_failure_doubt', 0) * sigmoid(current_risk_outcome_this_step,
                                                                                       k=-5, x0=-0.1)
        loss_b2 = k.get('y1_loss_b2', 0) * (self.b2_limitation / 7) ** 2.0
        reality_measure = (self.b1_resource + self.h1_possibilities) / 2
        aspiration_reality_gap = self.y3_aspiration - reality_measure
        loss_from_gap = 0
        if aspiration_reality_gap > params.get('y1_gap_threshold_for_loss', 3.0):
            loss_from_gap = k.get('y1_loss_aspiration_gap', 0) * (
                        aspiration_reality_gap - params.get('y1_gap_threshold_for_loss', 3.0))
        delta = effect_experience_validation - loss_b2 - loss_from_gap + neighbor_effect_y1
        return self._apply_boundary_effect(self.y1_clarity, delta)

    def _calculate_delta_y2(self, k, params, neighbor_effect_y2, current_risk_outcome_this_step):
        effect_y1 = k.get('y2_from_y1', 0) * sigmoid(self.y1_clarity, k=0.8, x0=3.5)
        aspiration_gap_y2 = self.y3_aspiration - self.y2_drive
        effect_y3 = k.get('y2_from_y3', 0) * sigmoid(aspiration_gap_y2, k=0.4, x0=0.5) * (
                    1 - sigmoid(self.y2_drive, k=1, x0=8))
        effect_risk_outcome = 0
        if current_risk_outcome_this_step > params.get('risk_success_return_threshold_for_y2', 0.02):
            effect_risk_outcome = k.get('y2_from_risk_success_激励', 0) * sigmoid(current_risk_outcome_this_step, k=6,
                                                                                  x0=0.05)
        elif current_risk_outcome_this_step < -params.get('risk_failure_loss_threshold_for_y2', 0.02):
            effect_risk_outcome = k.get('y2_from_risk_failure_打击', 0) * sigmoid(current_risk_outcome_this_step, k=-6,
                                                                                  x0=-0.05)
        loss_b2 = k.get('y2_loss_b2', 0) * (self.b2_limitation / 9) ** 1.5
        loss_low_y1 = k.get('y2_loss_low_y1', 0) * (1 - sigmoid(self.y1_clarity, k=1, x0=1.5))
        sustain_cost_low_b1 = 0
        if self.b1_resource < params.get('y2_b1_sustain_threshold', 2.0):
            sustain_cost_low_b1 = k.get('y2_sustain_cost_low_b1', 0) * (
                        params.get('y2_b1_sustain_threshold', 2.0) - self.b1_resource)
        delta = effect_y1 + effect_y3 + effect_risk_outcome - loss_b2 - loss_low_y1 - sustain_cost_low_b1 + neighbor_effect_y2
        return self._apply_boundary_effect(self.y2_drive, delta)

    def _calculate_delta_y3(self, k, params, neighbor_effect_y3, avg_b1_others, avg_h1_others):
        adjustment_from_y2 = k.get('y3_adjust_y2', 0) * (self.y2_drive - self.y3_aspiration) * 0.025
        boost_y1 = k.get('y3_boost_y1', 0) * sigmoid(self.y1_clarity - 6.0, k=0.9, x0=0)
        social_norm_b1_factor = 0
        if avg_b1_others is not None:
            social_norm_b1_factor = k.get('y3_social_norm_b1', 0) * (
                        avg_b1_others + params.get('y3_social_b1_offset', 1.0) - self.y3_aspiration)
        social_norm_h1_factor = 0
        if avg_h1_others is not None:
            social_norm_h1_factor = k.get('y3_social_norm_h1', 0) * (
                        avg_h1_others + params.get('y3_social_h1_offset', 0.5) - self.y3_aspiration)
        self_h1_factor = k.get('y3_self_h1_factor', 0) * (self.h1_possibilities - self.y3_aspiration)
        reality_crush = 0
        if self.b1_resource < self.y3_aspiration - params.get('y3_reality_gap_threshold', 4.0):
            reality_crush = k.get('y3_loss_reality_gap', 0) * \
                            (self.y3_aspiration - self.b1_resource - params.get('y3_reality_gap_threshold', 4.0))
        delta = adjustment_from_y2 + boost_y1 + social_norm_b1_factor + social_norm_h1_factor + self_h1_factor - reality_crush + neighbor_effect_y3
        return self._apply_boundary_effect(self.y3_aspiration, delta)

    def _calculate_delta_h1(self, k, params, neighbor_effect_h1):
        effect_b1 = k.get('h1_from_b1', 0) * sigmoid(self.b1_resource, k=0.6, x0=3.5)
        effect_h2 = k.get('h1_from_h2', 0) * sigmoid(self.h2_innovation, k=0.7, x0=3.0)
        loss_b2 = k.get('h1_loss_b2', 0) * (self.b2_limitation / 6) ** 2.0
        loss_low_y2_y1 = k.get('h1_loss_low_y_factor', 0) * \
                         ((1 - scale_value(self.y2_drive, 0, 3.5, 0, 1)) + (
                                     1 - scale_value(self.y1_clarity, 0, 3.5, 0, 1))) / 2
        delta = effect_b1 + effect_h2 - loss_b2 - loss_low_y2_y1 + neighbor_effect_h1
        return self._apply_boundary_effect(self.h1_possibilities, delta)

    def _calculate_delta_h2(self, k, params, neighbor_effect_h2):
        synergy_y1_y2 = self.y1_clarity * self.y2_drive / 100
        effect_y2_h3 = (k.get('h2_from_y2', 0) * self.y2_drive + k.get('h2_from_h3', 0) * self.h3_risk_appetite) * \
                       (0.4 + 0.6 * sigmoid(synergy_y1_y2, k=0.1, x0=(5 * 6 / 100)))
        practice_factor = (scale_value(self.y2_drive, 0, 10, 0.1, 1) + scale_value(self.h3_risk_appetite, 0, 10, 0.1,
                                                                                   1)) / 2
        decay = k.get('h2_decay_no_practice', 0) * (1.1 - practice_factor) * (self.h2_innovation / 9)
        delta = effect_y2_h3 - decay + neighbor_effect_h2
        return self._apply_boundary_effect(self.h2_innovation, delta)

    def _calculate_delta_h3(self, k, params, neighbor_effect_h3, current_risk_outcome_this_step):
        effect_risk_outcome = 0
        if current_risk_outcome_this_step > params.get('risk_success_return_threshold_for_h3', 0.03):
            effect_risk_outcome = k.get('h3_from_risk_success_回报', 0) * sigmoid(current_risk_outcome_this_step, k=8,
                                                                                  x0=0.05)
        elif current_risk_outcome_this_step < -params.get('risk_failure_loss_threshold_for_h3', 0.03):
            effect_risk_outcome = k.get('h3_from_risk_failure_惩罚', 0) * sigmoid(current_risk_outcome_this_step, k=-8,
                                                                                  x0=-0.05)
        effect_y1 = k.get('h3_from_y1', 0) * sigmoid(self.y1_clarity, k=0.7, x0=6.0)
        effect_y2 = k.get('h3_from_y2', 0) * sigmoid(self.y2_drive, k=0.7, x0=6.0)
        loss_b2 = k.get('h3_loss_b2', 0) * (self.b2_limitation / 10) ** 1.3
        delta = effect_risk_outcome + effect_y1 + effect_y2 - loss_b2 + neighbor_effect_h3
        return self._apply_boundary_effect(self.h3_risk_appetite, delta)

    def _calculate_delta_s1_trustworthiness(self, k, params, neighbor_feedback_trust, current_risk_outcome_this_step):
        delta = 0
        consistency_factor = sigmoid(self.y1_clarity - 5, k=0.8, x0=0) * \
                             sigmoid((self.b1_resource + self.h2_innovation) / 2 - (self.y3_aspiration - 2), k=0.6,
                                     x0=0)
        delta += k.get('s1_from_consistency', 0) * consistency_factor
        if current_risk_outcome_this_step < params.get('s1_risk_failure_penalty_thresh',
                                                       -0.3):  # Use current step's outcome
            delta -= k.get('s1_penalty_risk_failure', 0) * abs(current_risk_outcome_this_step)
        delta += neighbor_feedback_trust
        delta -= k.get('s1_decay', 0) * (self.s1_trustworthiness / 10)
        return self._apply_boundary_effect(self.s1_trustworthiness, delta)

    def _calculate_delta_s2_reputation(self, k, params, neighbor_feedback_rep, current_risk_outcome_this_step):
        delta = 0
        achievement_factor = (scale_value(self.b1_resource, 3, 10, 0, 1) + \
                              scale_value(self.h2_innovation, 4, 10, 0, 1)) / 2
        delta += k.get('s2_from_achievement', 0) * achievement_factor
        value_appeal_factor = sigmoid(self.y1_clarity - 6, k=0.7, x0=0) * \
                              sigmoid(self.y3_aspiration - 6, k=0.7, x0=0)
        delta += k.get('s2_from_value_appeal', 0) * value_appeal_factor
        if current_risk_outcome_this_step > params.get('s2_risk_success_bonus_thresh',
                                                       0.15):  # Use current step's outcome
            delta += k.get('s2_bonus_risk_success', 0) * current_risk_outcome_this_step
        delta += neighbor_feedback_rep
        delta -= k.get('s2_decay', 0) * (self.s2_reputation / 10)
        return self._apply_boundary_effect(self.s2_reputation, delta)

    def _calculate_neighbor_effects(self, params, all_states_objects_dict):
        k_social = params.get('coefficients', {}).get('social_interactions', {})  # Safe access
        effects = {key: 0.0 for key in DIM_KEYS}
        if not self.neighbors: return effects
        num_valid_neighbors = 0
        sum_neighbor_y1, sum_neighbor_y2, sum_neighbor_y3 = 0, 0, 0
        sum_neighbor_s1, sum_neighbor_s2 = 0, 0

        for neighbor_name in self.neighbors:
            if neighbor_name in all_states_objects_dict and neighbor_name != self.name_en:
                neighbor_obj = all_states_objects_dict[neighbor_name]
                num_valid_neighbors += 1

                trust_target = neighbor_obj.s1_trustworthiness
                trust_delta = k_social.get('trust_formation_rate', 0) * (
                            trust_target - self.trust_levels[neighbor_name])
                trust_delta *= scale_value(self.s1_trustworthiness, 0, 10, 0.8, 1.2)
                self.trust_levels[neighbor_name] = np.clip(self.trust_levels[neighbor_name] + trust_delta, 0, 10)
                current_trust_in_neighbor = self.trust_levels[neighbor_name]

                sum_neighbor_y1 += neighbor_obj.y1_clarity;
                sum_neighbor_y2 += neighbor_obj.y2_drive
                sum_neighbor_y3 += neighbor_obj.y3_aspiration;
                sum_neighbor_s1 += neighbor_obj.s1_trustworthiness
                sum_neighbor_s2 += neighbor_obj.s2_reputation

                b1_diff = neighbor_obj.b1_resource - self.b1_resource
                coop_factor_trust = scale_value(current_trust_in_neighbor, 0, 10, 0.5, 1.5)
                if b1_diff > k_social.get('b1_comp_diff_thresh', 1.0):
                    effects['b1_resource'] -= k_social.get('b1_comp_loss_factor', 0) * b1_diff * (
                                1.1 - coop_factor_trust * 0.5)
                elif abs(b1_diff) < k_social.get('b1_coop_diff_thresh', 0.5) and self.b1_resource > 2.0:
                    effects['b1_resource'] += k_social.get('b1_coop_gain_factor', 0) * \
                                              min(self.b1_resource, neighbor_obj.b1_resource) * coop_factor_trust

                if neighbor_obj.h2_innovation > self.h2_innovation:
                    info_gain_potential = (neighbor_obj.h2_innovation - self.h2_innovation) * \
                                          k_social.get('h2_info_share_factor', 0)
                    trust_in_source_factor = scale_value(current_trust_in_neighbor, 0, 10, 0.3, 1.0)
                    source_reputation_factor = scale_value(neighbor_obj.s2_reputation, 0, 10, 0.5, 1.2)
                    # Ensure this effect is additive and reasonable
                    effects['h2_innovation'] = effects.get('h2_innovation',
                                                           0) + info_gain_potential * trust_in_source_factor * source_reputation_factor

        if num_valid_neighbors > 0:
            avg_neighbor_y1 = sum_neighbor_y1 / num_valid_neighbors;
            avg_neighbor_y2 = sum_neighbor_y2 / num_valid_neighbors
            avg_neighbor_y3 = sum_neighbor_y3 / num_valid_neighbors;
            avg_neighbor_s1 = sum_neighbor_s1 / num_valid_neighbors
            avg_neighbor_s2 = sum_neighbor_s2 / num_valid_neighbors

            confidence_factor_self = scale_value(self.s2_reputation, 0, 10, 0.5, 1.0)
            effects['y1_clarity'] = effects.get('y1_clarity', 0) + k_social.get('y1_alignment_factor', 0) * (
                        avg_neighbor_y1 - self.y1_clarity) * \
                                    (1 - sigmoid(self.y1_clarity, k=1.2, x0=7.5)) * (1.1 - confidence_factor_self)
            effects['y2_drive'] = effects.get('y2_drive', 0) + k_social.get('y2_contagion_factor', 0) * (
                        avg_neighbor_y2 - self.y2_drive) * \
                                  (1 - sigmoid(self.y2_drive, k=1.2, x0=7.5)) * (1.1 - confidence_factor_self)
            effects['y3_aspiration'] = effects.get('y3_aspiration', 0) + k_social.get('y3_alignment_factor', 0) * (
                        avg_neighbor_y3 - self.y3_aspiration) * \
                                       (1 - sigmoid(self.y3_aspiration, k=1, x0=8.0)) * (1.1 - confidence_factor_self)
            effects['s1_trustworthiness'] = effects.get('s1_trustworthiness', 0) + k_social.get('s1_social_norm_factor',
                                                                                                0) * (
                                                        avg_neighbor_s1 - self.s1_trustworthiness)
            effects['s2_reputation'] = effects.get('s2_reputation', 0) + k_social.get('s2_social_pressure_factor',
                                                                                      0) * (
                                                   avg_neighbor_s2 - self.s2_reputation)
        return effects

    def evolve(self, params, all_states_objects_dict, active_event_effects_on_self, global_env_factors):
        k = params.get('coefficients', {})
        noise_level = params.get('noise_level', 0.05)
        lr = params.get('learning_rate', 0.1)

        b1_acquisition_modifier = global_env_factors.get('b1_acq_mod', 1.0)
        b2_base_increase = global_env_factors.get('b2_base_inc', 0.0)

        # --- 1. 风险项目评估与执行 ---
        current_risk_outcome_this_step = 0.0
        risk_project_b1_change = 0.0
        risk_project_b2_change = 0.0

        # Ensure params for risk calculation are present
        h3_risk_attempt_thresh = params.get('h3_risk_attempt_threshold', 4.0)
        y2_risk_attempt_thresh = params.get('y2_risk_attempt_threshold', 3.0)

        risk_attempt_probability = sigmoid(self.h3_risk_appetite - h3_risk_attempt_thresh, k=0.8, x0=0) * \
                                   sigmoid(self.y2_drive - y2_risk_attempt_thresh, k=0.7, x0=0) * \
                                   scale_value(self.s2_reputation, 0, 10, 1.1, 0.8)

        if random.random() < risk_attempt_probability:
            b1_invest_ratio_base = scale_value(self.h3_risk_appetite, 0, 10,
                                               params.get('risk_invest_ratio_min', 0.03),
                                               params.get('risk_invest_ratio_max', 0.12))
            b1_invest_ratio_modifier = (0.6 + 0.2 * sigmoid(self.y1_clarity, k=1, x0=5) + \
                                        0.2 * sigmoid(self.s1_trustworthiness, k=1, x0=6))
            b1_invested = self.b1_resource * np.clip(b1_invest_ratio_base * b1_invest_ratio_modifier, 0, 0.30)

            potential_R_factor = k.get('risk_potential_R_base', 0) + \
                                 k.get('risk_potential_R_h2', 0) * self.h2_innovation + \
                                 k.get('risk_potential_R_y1', 0) * self.y1_clarity + \
                                 k.get('risk_potential_R_s2', 0) * self.s2_reputation

            inherent_risk_L_factor = k.get('risk_inherent_L_base', 0) + \
                                     k.get('risk_inherent_L_h3', 0) * self.h3_risk_appetite + \
                                     k.get('risk_inherent_L_b2', 0) * self.b2_limitation - \
                                     k.get('risk_reduction_s1', 0) * self.s1_trustworthiness

            mean_outcome = potential_R_factor - inherent_risk_L_factor * 0.6
            std_dev_outcome = max(0.01, inherent_risk_L_factor * 0.7)
            actual_return_factor_on_investment = random.normalvariate(mean_outcome, std_dev_outcome)
            current_risk_outcome_this_step = actual_return_factor_on_investment

            risk_project_b1_change = b1_invested * actual_return_factor_on_investment * b1_acquisition_modifier
            if current_risk_outcome_this_step < params.get('risk_failure_threshold_for_b2', -0.1):
                risk_project_b2_change = k.get('b2_from_major_risk_failure', 0) * abs(
                    current_risk_outcome_this_step) * (b1_invested / max(0.1,
                                                                         self.b1_resource if self.b1_resource > 0 else 0.1))
            self.active_effects_log.append(
                f"风险项目: 投 {b1_invested:.2f}, 回报因子 {current_risk_outcome_this_step:.2f}, B1变 {risk_project_b1_change:.2f}, B2变 {risk_project_b2_change:.2f}")

        # --- 2. 计算其他全局平均值 ---
        avg_b1_others, avg_h1_others = None, None
        all_b1s = [obj.b1_resource for name, obj in all_states_objects_dict.items() if name != self.name_en]
        if all_b1s: avg_b1_others = np.mean(all_b1s)
        all_h1s = [obj.h1_possibilities for name, obj in all_states_objects_dict.items() if name != self.name_en]
        if all_h1s: avg_h1_others = np.mean(all_h1s)

        # --- 3. 计算邻居效应 ---
        neighbor_effects = self._calculate_neighbor_effects(params, all_states_objects_dict)

        # --- 4. 计算各维度Deltas (传递本轮的风险结果) ---
        deltas = {
            'b1_resource': self._calculate_delta_b1(k, params, avg_b1_others, neighbor_effects.get('b1_resource', 0),
                                                    risk_project_b1_change),
            'b2_limitation': self._calculate_delta_b2(k, params, neighbor_effects.get('b2_limitation', 0),
                                                      risk_project_b2_change) + b2_base_increase,
            'y1_clarity': self._calculate_delta_y1(k, params, neighbor_effects.get('y1_clarity', 0),
                                                   current_risk_outcome_this_step),
            'y2_drive': self._calculate_delta_y2(k, params, neighbor_effects.get('y2_drive', 0),
                                                 current_risk_outcome_this_step),
            'y3_aspiration': self._calculate_delta_y3(k, params, neighbor_effects.get('y3_aspiration', 0),
                                                      avg_b1_others, avg_h1_others),
            'h1_possibilities': self._calculate_delta_h1(k, params, neighbor_effects.get('h1_possibilities', 0)),
            'h2_innovation': self._calculate_delta_h2(k, params, neighbor_effects.get('h2_innovation', 0)),
            'h3_risk_appetite': self._calculate_delta_h3(k, params, neighbor_effects.get('h3_risk_appetite', 0),
                                                         current_risk_outcome_this_step),
            's1_trustworthiness': self._calculate_delta_s1_trustworthiness(k, params,
                                                                           neighbor_effects.get('s1_trustworthiness',
                                                                                                0),
                                                                           current_risk_outcome_this_step),
            's2_reputation': self._calculate_delta_s2_reputation(k, params, neighbor_effects.get('s2_reputation', 0),
                                                                 current_risk_outcome_this_step),
        }

        # --- 5. 应用Deltas, 噪声, 事件效果, 和裁剪 ---
        # Store logs from this state's evolution for this step
        current_step_evolution_logs = self.active_effects_log[:]  # Copy current risk logs
        self.active_effects_log.clear()  # Clear for event effects

        for key in DIM_KEYS:
            current_val = getattr(self, key)
            total_delta_from_sources = deltas.get(key, 0)
            noise_val = random.uniform(-noise_level, noise_level)
            effective_delta = total_delta_from_sources
            if key == 'b1_resource' and effective_delta > 0:
                effective_delta *= b1_acquisition_modifier
            effective_delta += noise_val
            new_val_before_event = current_val + effective_delta * lr
            final_val_after_event = new_val_before_event

            if key in active_event_effects_on_self:
                for event_eff in active_event_effects_on_self.get(key, []):  # Use .get for safety
                    eff_val = event_eff.get('val', 0)
                    resilience_factor = scale_value(self.y1_clarity, 0, 10, 0.7, 1.3)
                    actual_eff_val = eff_val / resilience_factor if eff_val < 0 else eff_val * resilience_factor
                    # Log event effect application by EventManager, not duplicate here
                    # self.active_effects_log.append(f"事件 '{event_eff.get('name','未命名')}' -> {key} {event_eff.get('type','未知类型')} {actual_eff_val:.2f} (原:{eff_val:.2f})")
                    if event_eff.get('type') == 'add_abs':
                        final_val_after_event += actual_eff_val
                    elif event_eff.get('type') == 'set_abs':
                        final_val_after_event = actual_eff_val
                    elif event_eff.get('type') == 'multiply_abs':
                        final_val_after_event *= actual_eff_val

            setattr(self, key, np.clip(final_val_after_event, 0, 10))

        # After all dimensions are updated, THEN set the last_risk_outcome_factor for the *next* step's S1/S2 or memory.
        self.last_risk_outcome_factor = current_risk_outcome_this_step
        # Combine evolution logs (like risk project) with event application logs
        # EventManager will provide its own log messages for triggering.
        # WorldState.active_effects_log here will now mainly contain details if an event was *applied* to it.
        # It might be better for EventManager to generate all user-facing event logs.
        # For now, `run_evolution_step` combines them.
        # Let's ensure WorldState's log is primarily for *its own internal* decisions like risk.
        self.active_effects_log = current_step_evolution_logs  # Restore risk logs for this step for run_evolution_step to collect.

    def record_history(self, coord_type, wb, wy, wh, max_history=50):
        coords = self.get_coords_for_plot(coord_type, wb, wy, wh)
        if len(self.history) >= max_history: self.history.pop(0)
        self.history.append(coords)

    def clear_history(self):
        self.history = []

    def to_dict(self):
        data = {key: getattr(self, key) for key in DIM_KEYS}
        data.update({'name_zh': self.name_zh, 'name_en': self.name_en,
                     'history': self.history, 'neighbors': self.neighbors,
                     'trust_levels': dict(self.trust_levels),
                     'last_risk_outcome_factor': self.last_risk_outcome_factor})
        return data

    @classmethod
    def from_dict(cls, data):
        obj = cls(
            data.get('name_zh', '未知状态'), data.get('name_en', f'Unknown_{random.randint(1000, 9999)}'),
            data.get('b1_resource', 0), data.get('b2_limitation', 0), data.get('y1_clarity', 0),
            data.get('y2_drive', 0), data.get('y3_aspiration', 0), data.get('h1_possibilities', 0),
            data.get('h2_innovation', 0), data.get('h3_risk_appetite', 0),
            data.get('s1_trustworthiness', 5.0), data.get('s2_reputation', 3.0)
        )
        obj.history = data.get('history', [])
        obj.neighbors = data.get('neighbors', [])
        obj.trust_levels = defaultdict(lambda: 5.0, data.get('trust_levels', {}))
        obj.last_risk_outcome_factor = data.get('last_risk_outcome_factor', 0)
        return obj

    def __repr__(self):
        return f"<WorldState: {self.get_display_name()}>"


# --- EventManager and Event classes (Mostly from GM4.5.1) ---
class Event:
    def __init__(self, name, trigger_type, trigger_params, target_selector, effects, duration=1, one_time=False):
        self.name = name;
        self.trigger_type = trigger_type;
        self.trigger_params = trigger_params
        self.target_selector = target_selector;
        self.effects = effects;
        self.duration = duration
        self.one_time = one_time;
        self.triggered_this_step = False

    def check_trigger(self, all_states_objects_dict, global_metrics, global_env_factors):
        if self.trigger_type == 'probabilistic':
            base_prob = self.trigger_params.get('prob', 0.01)
            env_prob_mod_key = self.trigger_params.get('env_prob_mod_key')
            env_prob_modifier = 1.0
            if env_prob_mod_key and global_env_factors:
                env_prob_modifier = global_env_factors.get(env_prob_mod_key, 1.0)
            return random.random() < (base_prob * env_prob_modifier)
        elif self.trigger_type == 'conditional_global':
            metric_name = self.trigger_params.get('dim')
            source_type = self.trigger_params.get('source', 'metrics')
            source_dict = global_metrics if source_type == 'metrics' else global_env_factors
            if not metric_name: return False
            metric_val = source_dict.get(metric_name)
            if metric_val is None: return False
            op = self.trigger_params.get('op');
            thresh_val = self.trigger_params.get('val')
            if not op or thresh_val is None: return False
            if op == '<': return metric_val < thresh_val
            if op == '>': return metric_val > thresh_val
            if op == '==': return metric_val == thresh_val
        return False

    def select_targets(self, all_states_objects_dict):
        targets = []
        if self.target_selector == 'all':
            targets = list(all_states_objects_dict.values())
        elif isinstance(self.target_selector, dict) and 'type' in self.target_selector:
            st = self.target_selector['type']
            if st == 'random_n':
                n = self.target_selector.get('n', 1);
                pop = list(all_states_objects_dict.values())
                if pop: targets = random.sample(pop, min(n, len(pop)))
            elif st == 'conditional_individual':
                dim, op, val, max_t = (self.target_selector.get(k) for k in ['dim', 'op', 'val', 'max_targets'])
                if not dim or not op or val is None: return []
                max_t = float(max_t) if max_t is not None else float('inf')
                eligible = [s for s in all_states_objects_dict.values() if s and hasattr(s, dim) and \
                            (getattr(s, dim, None) is not None and (
                                    (op == '>' and getattr(s, dim) > val) or \
                                    (op == '<' and getattr(s, dim) < val) or \
                                    (op == '==' and getattr(s, dim) == val)))]
                if eligible: targets = random.sample(eligible, min(int(max_t), len(eligible)))
        return targets

    def get_effects_for_target(self):
        return [{'name': self.name, 'dim': e.get('dim'), 'type': e.get('type'), 'duration': self.duration,
                 'val': e.get('val', 0) + (
                     random.uniform(-e.get('rand_range', 0), e.get('rand_range', 0)) * abs(e.get('val', 0)) if e.get(
                         'rand_range', 0) > 0 else 0)
                 } for e in self.effects]


class EventManager:
    def __init__(self, event_definitions_template_arg):
        self.event_definitions_template = event_definitions_template_arg
        self.events = [Event(**ed) for ed in self.event_definitions_template]
        self.active_timed_effects = defaultdict(list)

    def reset_events(self):
        self.events = [Event(**ed) for ed in self.event_definitions_template]
        self.active_timed_effects.clear()

    def process_step(self, all_states_objects_dict, global_metrics, global_env_factors):
        effects_to_apply_this_step = defaultdict(lambda: defaultdict(list))
        log_messages = []
        events_to_remove_indices = []
        for i, event_obj in enumerate(self.events):
            event_obj.triggered_this_step = False
            if event_obj.check_trigger(all_states_objects_dict, global_metrics, global_env_factors):
                event_obj.triggered_this_step = True
                log_messages.append(f"事件 '{event_obj.name}' 已触发.")
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
                if event_obj.one_time and event_obj.triggered_this_step: events_to_remove_indices.append(i)
        for index_to_remove in sorted(events_to_remove_indices, reverse=True): del self.events[index_to_remove]
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
                        log_messages.append(f"定时效果 '{effect_data.get('name', '未命名')}' 在 {state_name} 上已到期.")
        self.active_timed_effects = new_active_timed_effects
        return effects_to_apply_this_step, log_messages


# --- 4. Initialization (GM4.5.2 - using _gm452 suffix) ---
initial_states_templates_gm452 = [
    {'name_zh': "孤僻者", 'name_en': "Loner", 'b1_res': 3, 'b2_lim': 6, 'y1_cla': 4, 'y2_dri': 3, 'y3_asp': 5,
     'h1_pos': 2, 'h2_inn': 2, 'h3_ris': 2, 's1_tru': 4, 's2_rep': 2},
    {'name_zh': "合作领袖", 'name_en': "CollaborativeLeader", 'b1_res': 6, 'b2_lim': 3, 'y1_cla': 8, 'y2_dri': 7,
     'y3_asp': 8, 'h1_pos': 7, 'h2_inn': 7, 'h3_ris': 6, 's1_tru': 8, 's2_rep': 7},
    {'name_zh': "投机者", 'name_en': "Opportunist", 'b1_res': 5, 'b2_lim': 4, 'y1_cla': 5, 'y2_dri': 6, 'y3_asp': 7,
     'h1_pos': 6, 'h2_inn': 5, 'h3_ris': 8, 's1_tru': 3, 's2_rep': 4},
    {'name_zh': "勤勉工匠", 'name_en': "DiligentArtisan", 'b1_res': 4, 'b2_lim': 4, 'y1_cla': 7, 'y2_dri': 5,
     'y3_asp': 6, 'h1_pos': 4, 'h2_inn': 6, 'h3_ris': 3, 's1_tru': 7, 's2_rep': 5},
    {'name_zh': "保守长者", 'name_en': "ConservativeElder", 'b1_res': 7, 'b2_lim': 2, 'y1_cla': 6, 'y2_dri': 3,
     'y3_asp': 5, 'h1_pos': 3, 'h2_inn': 2, 'h3_ris': 1, 's1_tru': 6, 's2_rep': 6},
]
initial_states_obj_list_gm452 = [WorldState(**s_data) for s_data in initial_states_templates_gm452]
neighbor_config_gm452 = {
    "CollaborativeLeader": ["Opportunist", "DiligentArtisan", "ConservativeElder"],
    "Opportunist": ["CollaborativeLeader", "Loner"],
    "DiligentArtisan": ["CollaborativeLeader", "ConservativeElder"], "Loner": ["Opportunist"],
    "ConservativeElder": ["CollaborativeLeader", "DiligentArtisan"]
}
for state_obj in initial_states_obj_list_gm452: state_obj.neighbors = neighbor_config_gm452.get(state_obj.name_en, [])
initial_world_states_store_data_gm452 = {s.name_en: s.to_dict() for s in initial_states_obj_list_gm452}

default_evolution_params_gm452 = {  # Copied from GM4.5, use this as base for current version
    'learning_rate': 0.05, 'noise_level': 0.015, 'base_consumption': 0.03,
    'h3_risk_attempt_threshold': 4.0, 'y2_risk_attempt_threshold': 3.0, 'risk_invest_ratio_min': 0.025,
    'risk_invest_ratio_max': 0.13, 'risk_success_return_threshold_for_y1': 0.035,
    'risk_failure_loss_threshold_for_y1': 0.035, 'risk_success_return_threshold_for_y2': 0.025,
    'risk_failure_loss_threshold_for_y2': 0.025, 'risk_success_return_threshold_for_h3': 0.035,
    'risk_failure_loss_threshold_for_h3': 0.035, 'risk_failure_threshold_for_b2': -0.08,
    'y2_b1_sustain_threshold': 2.2, 'y1_gap_threshold_for_loss': 3.0, 'y3_reality_gap_threshold': 4.0,
    'y3_social_b1_offset': 1.0, 'y3_social_h1_offset': 0.6, 'h3_active_threshold_for_cost': 3.5,
    's1_risk_failure_penalty_thresh': -0.2, 's2_risk_success_bonus_thresh': 0.15,
    'coefficients': {
        's1_from_consistency': 0.025, 's1_penalty_risk_failure': 0.035, 's1_decay': 0.008,
        's2_from_achievement': 0.03, 's2_from_value_appeal': 0.015, 's2_bonus_risk_success': 0.035, 's2_decay': 0.01,
        'b1_from_s2_reputation': 0.003, 'risk_potential_R_s2': 0.005, 'risk_reduction_s1': 0.005,
        'risk_potential_R_base': 0.04, 'risk_potential_R_h2': 0.02, 'risk_potential_R_y1': 0.01,
        'risk_inherent_L_base': 0.18, 'risk_inherent_L_h3': 0.02, 'risk_inherent_L_b2': 0.018,
        'b2_from_major_risk_failure': 0.1, 'y1_from_success_validation': 0.18, 'y1_from_failure_doubt': -0.20,
        'y2_from_risk_success_激励': 0.16, 'y2_from_risk_failure_打击': -0.18, 'y2_sustain_cost_low_b1': 0.028,
        'y3_social_norm_b1': 0.003, 'y3_social_norm_h1': 0.0025, 'y3_self_h1_factor': 0.004,
        'h3_from_risk_success_回报': 0.18, 'h3_from_risk_failure_惩罚': -0.22,
        'b1_from_h2': 0.08, 'b1_from_y2': 0.07, 'b1_loss_b2': 0.125, 'b1_cost_h2_activity': 0.012,
        'b1_cost_y2_sustain': 0.007, 'b1_social_pressure': 0.0015, 'b2_reduce_y2': 0.05, 'b2_reduce_h2': 0.04,
        'b2_random_factor': 0.08, 'y1_loss_b2': 0.115, 'y1_loss_aspiration_gap': 0.02,
        'y2_from_y1': 0.06, 'y2_from_y3': 0.02, 'y2_from_b1_growth': 0.0, 'y2_loss_b2': 0.095, 'y2_loss_low_y1': 0.23,
        'y3_adjust_y2': 0.05, 'y3_boost_y1': 0.007, 'y3_loss_reality_gap': 0.035,
        'h1_from_b1': 0.08, 'h1_from_h2': 0.10, 'h1_loss_b2': 0.135, 'h1_loss_low_y_factor': 0.02,
        'h2_from_y2': 0.09, 'h2_from_h3': 0.06, 'h2_decay_no_practice': 0.07,
        'h3_from_y1': 0.04, 'h3_from_y2': 0.07, 'h3_loss_b2': 0.115,
        'social_interactions': {
            'trust_formation_rate': 0.06, 'y1_alignment_factor': 0.005, 'y2_contagion_factor': 0.01,
            'y3_alignment_factor': 0.004, 's1_social_norm_factor': 0.004, 's2_social_pressure_factor': 0.003,
            'h2_info_share_factor': 0.008, 'b1_comp_diff_thresh': 1.0, 'b1_comp_loss_factor': 0.018,
            'b1_coop_diff_thresh': 0.6, 'b1_coop_gain_factor': 0.0008,
        }
    },
    'plot_weights': {'b': (0.6, 0.4), 'y': (0.35, 0.35, 0.3), 'h': (0.35, 0.35, 0.3), 's': (0.5, 0.5)}
}
global_environment_factors_gm452 = {  # Use specific name for this version
    'b1_acq_mod': 1.0, 'b2_base_inc': 0.0, 'event_crisis_likelihood_mod': 1.0, 'econ_cycle_event_mod': 1.0
}
event_definitions_gm452 = [  # Use specific name for this version
    {'name': "经济周期波动", 'trigger_type': "probabilistic",
     'trigger_params': {'prob': 0.03, 'env_prob_mod_key': 'econ_cycle_event_mod'},
     'target_selector': "all",
     'effects': [{'dim': 'b1_resource', 'type': 'multiply_abs', 'val': random.choice([0.9, 1.1]), 'rand_range': 0.05}]},
    {'name': "行业革新", 'trigger_type': "probabilistic", 'trigger_params': {'prob': 0.01},
     'target_selector': {'type': 'random_n', 'n': 1},
     'effects': [{'dim': 'h2_innovation', 'type': 'add_abs', 'val': 2.0, 'rand_range': 0.5},
                 {'dim': 'h1_possibilities', 'type': 'add_abs', 'val': 1.5}]},
    {'name': "信任危机 (全局)", 'trigger_type': "conditional_global",
     'trigger_params': {'dim': 'avg_s1_trust', 'op': '<', 'val': 3.5, 'source': 'metrics'},
     'target_selector': "all",
     'effects': [{'dim': 's1_trustworthiness', 'type': 'add_abs', 'val': -0.5, 'rand_range': 0.1}], 'duration': 3,
     'one_time': True},
]
event_manager = EventManager(event_definitions_gm452)

# --- 5. Dash App Layout (using _gm452 suffixed initial data) ---
app = dash.Dash(__name__, suppress_callback_exceptions=True)
app.title = "三世界理论 - v4.5.2 (最终修正版)"
app.layout = html.Div([
    html.H1(app.title, style={'textAlign': 'center', 'color': '#2c3e50'}),
    dcc.Store(id='world-states-store', data=initial_world_states_store_data_gm452),
    dcc.Store(id='selected-point-id-store', data=None),
    dcc.Store(id='evolution-params-store', data=default_evolution_params_gm452),
    dcc.Store(id='global-env-factors-store', data=global_environment_factors_gm452),
    dcc.Store(id='event-log-store', data=[]),
    dcc.Interval(id='evolution-interval', interval=1000, n_intervals=0, disabled=True),
    html.Div([
        html.Div([  # Left Panel
            html.H3("控制面板", style={'textAlign': 'center', 'borderBottom': '1px solid #ccc', 'paddingBottom': '10px',
                                       'marginBottom': '15px'}),
            html.Label("坐标系类型:", style={'fontWeight': 'bold'}),
            dcc.Dropdown(id='coord-type-dropdown', options=[{'label': '简化坐标 (B1, Y2, H1)', 'value': 'simplified'},
                                                            {'label': '综合坐标 (B_c, Y_c, H_c)',
                                                             'value': 'composite'}], value='simplified',
                         clearable=False, style={'marginBottom': '15px'}),
            html.Label("选择状态点:", style={'fontWeight': 'bold'}),
            dcc.Dropdown(id='point-selector-dropdown', clearable=False, style={'marginBottom': '15px'}),
            html.Div(id='edit-panel-div', style={'padding': '10px', 'border': '1px solid #ddd', 'borderRadius': '5px',
                                                 'marginBottom': '15px', 'backgroundColor': '#f9f9f9'}),
            html.Div(id='neighbor-info-div', style={'fontSize': 'small', 'marginBottom': '15px', 'padding': '5px',
                                                    'border': '1px dashed #ccc'}),
            html.Div(id='trust-info-div', style={'fontSize': 'small', 'marginBottom': '15px', 'padding': '5px',
                                                 'border': '1px dashed #bbf'}),
            html.H4("全局环境因子", style={'marginTop': '15px', 'borderTop': '1px solid #ccc', 'paddingTop': '10px'}),
            html.Label("B1获取效率修正:", style={'fontSize': 'small'}),
            dcc.Slider(id='env-b1-acq-slider', min=0.5, max=1.5, step=0.05,
                       value=global_environment_factors_gm452['b1_acq_mod'],
                       marks={i / 10: str(i / 10) for i in range(5, 16, 2)}, tooltip={"placement": "bottom"}),
            html.Label("B2基础增量:", style={'fontSize': 'small'}),
            dcc.Slider(id='env-b2-inc-slider', min=-0.1, max=0.2, step=0.01,
                       value=global_environment_factors_gm452['b2_base_inc'],
                       marks={i / 100: str(i / 100) for i in range(-10, 21, 5)}, tooltip={"placement": "bottom"}),
            html.H4("动态演化控制", style={'marginTop': '20px', 'borderTop': '1px solid #ccc', 'paddingTop': '15px'}),
            html.Div([html.Button('开始/暂停演化', id='toggle-evolution-button', n_clicks=0, className='button-primary',
                                  style={'marginRight': '10px'}),
                      html.Button('演化一步', id='step-evolution-button', n_clicks=0, style={'marginRight': '10px'}),
                      html.Button('重置所有状态', id='reset-states-button', n_clicks=0, className='button-danger'), ],
                     style={'marginBottom': '10px'}),
            html.Div([html.Label("演化速度 (ms/步): ", style={'display': 'inline-block', 'marginRight': '5px'}),
                      dcc.Input(id='evolution-interval-input', type='number', value=1000, min=100, step=100,
                                style={'width': '80px'})], style={'marginTop': '10px'}),
            html.Label("学习率 (lr):", style={'fontWeight': 'bold', 'marginTop': '10px'}),
            dcc.Slider(id='lr-slider', min=0.01, max=0.1, step=0.005,
                       value=default_evolution_params_gm452['learning_rate'],
                       marks={i / 100: f"{i / 100:.2f}" for i in range(1, 11, 1)},
                       tooltip={"placement": "bottom", "always_visible": True}),
            html.Label("噪声水平:", style={'fontWeight': 'bold', 'marginTop': '10px'}),
            dcc.Slider(id='noise-slider', min=0, max=0.03, step=0.001,
                       value=default_evolution_params_gm452['noise_level'],
                       marks={i / 1000: f"{i / 1000:.3f}" for i in range(0, 31, 5)},
                       tooltip={"placement": "bottom", "always_visible": True}),
            html.Div(id='n-intervals-display', style={'marginTop': '15px', 'fontSize': 'small', 'color': 'gray'})
        ], style={'width': '30%', 'float': 'left', 'padding': '20px', 'boxSizing': 'border-box',
                  'backgroundColor': '#f0f4f8', 'borderRight': '1px solid #ccc', 'maxHeight': '90vh',
                  'overflowY': 'auto'}),
        html.Div([html.Div(dcc.Graph(id='main-3d-scatter-plot', style={'height': '70vh'})),
                  html.Div([html.H5("事件与状态日志:", style={'marginTop': '10px', 'marginBottom': '5px'}),
                            dcc.Textarea(id='event-log-textarea', value="", readOnly=True,
                                         style={'width': '100%', 'height': '12vh', 'fontSize': 'small',
                                                'border': '1px solid #ddd', 'backgroundColor': '#fafafa'})])
                  ], style={'width': '70%', 'float': 'right', 'padding': '10px', 'boxSizing': 'border-box'})]),
    html.Div(style={'clear': 'both'}),
    html.Footer(f"三世界理论模型 - v4.5.2 (最终引用修正) - {time.strftime('%Y-%m-%d %H:%M:%S')}",
                style={'textAlign': 'center', 'marginTop': '20px', 'padding': '10px', 'fontSize': 'x-small',
                       'color': '#888'})
], style={'fontFamily': "'Segoe UI', Tahoma, Geneva, Verdana, sans-serif", 'maxWidth': '1800px', 'margin': 'auto',
          'backgroundColor': '#e9ecef'})


# --- 6. Callbacks (using _gm452 suffix) ---
@app.callback(Output('trust-info-div', 'children'),
              [Input('point-selector-dropdown', 'value'), Input('world-states-store', 'data')])
def display_trust_info_gm452(selected_id, states_data_json):
    if not selected_id or not states_data_json or selected_id not in states_data_json: return "选择状态点查看信任信息。"
    state_data = states_data_json[selected_id];
    trust_levels_dict = state_data.get('trust_levels', {})
    if not trust_levels_dict: return f"{state_data.get('name_zh', 'N/A')} 尚无信任记录。"
    trust_info_str = f"{state_data.get('name_zh', 'N/A')} 的信任级别: "
    trust_details = [f"{n_id}: {trust_val:.1f}" for n_id, trust_val in trust_levels_dict.items()]
    return trust_info_str + ", ".join(trust_details) if trust_details else trust_info_str + "无具体信任对象。"


@app.callback(Output('global-env-factors-store', 'data'),
              [Input('env-b1-acq-slider', 'value'), Input('env-b2-inc-slider', 'value')],
              [State('global-env-factors-store', 'data')])
def update_global_env_factors_gm452(b1_acq_mod, b2_base_inc, current_env_data):  # Renamed
    if b1_acq_mod is None or b2_base_inc is None: raise PreventUpdate
    new_env_data = current_env_data.copy() if current_env_data else {}
    new_env_data['b1_acq_mod'] = float(b1_acq_mod);
    new_env_data['b2_base_inc'] = float(b2_base_inc)
    return new_env_data


@app.callback([Output('point-selector-dropdown', 'options'), Output('point-selector-dropdown', 'value')],
              [Input('world-states-store', 'data')], [State('selected-point-id-store', 'data')])
def update_point_selector_gm452(states_data_json, selected_point_id):  # Renamed
    if not states_data_json: return [], None
    options = [{'label': f"{s_data['name_zh']} ({s_data['name_en']})", 'value': s_data['name_en']}
               for s_id, s_data in states_data_json.items() if
               isinstance(s_data, dict) and 'name_en' in s_data and 'name_zh' in s_data]
    valid_ids = [opt['value'] for opt in options]
    current_value = selected_point_id if selected_point_id in valid_ids else (options[0]['value'] if options else None)
    return options, current_value


@app.callback(Output('edit-panel-div', 'children'), Input('point-selector-dropdown', 'value'),
              State('world-states-store', 'data'))
def update_edit_panel_gm452(selected_id, states_data_json):  # Renamed
    if not selected_id or not states_data_json or selected_id not in states_data_json: return html.P(
        "请选择一个状态点进行编辑。", style={'color': 'orange'})
    state_data = states_data_json[selected_id]
    if not isinstance(state_data, dict): return html.P(f"加载状态点 '{selected_id}' 数据格式错误。",
                                                       style={'color': 'red'})
    try:
        state_obj = WorldState.from_dict(state_data)
    except Exception as e:
        return html.P(f"加载状态点 '{selected_id}' 数据时出错: {e}", style={'color': 'red'})
    panel_children = [
        html.H4(f"编辑: {state_obj.get_display_name()}", style={'marginTop': '0', 'marginBottom': '10px'})]
    for key_dim in DIM_KEYS:
        label = DIMENSION_LABELS_MAP_ZH.get(key_dim, key_dim.replace('_', ' ').title())
        val = getattr(state_obj, key_dim, 0)
        panel_children.extend([html.Label(label, style={'fontWeight': 'normal', 'fontSize': 'small', 'display': 'block',
                                                        'marginBottom': '2px'}),
                               dcc.Slider(id={'type': 'dim-slider', 'index': key_dim}, min=0, max=10, step=0.1,
                                          value=val, marks={i: str(i) for i in range(0, 11, 2)},
                                          tooltip={"placement": "bottom", "always_visible": False},
                                          className='dim-slider-style'),
                               html.Div(style={'marginBottom': '5px'})])
    return panel_children


@app.callback(Output('world-states-store', 'data', allow_duplicate=True),
              Input({'type': 'dim-slider', 'index': dash.ALL}, 'value'),
              State({'type': 'dim-slider', 'index': dash.ALL}, 'id'), State('point-selector-dropdown', 'value'),
              State('world-states-store', 'data'), prevent_initial_call=True)
def update_state_from_sliders_gm452(slider_values, slider_ids_obj_list, selected_id, states_data_json):  # Renamed
    ctx = callback_context
    if not ctx.triggered or not selected_id or not states_data_json or selected_id not in states_data_json: return no_update
    triggered_input = ctx.triggered[0];
    slider_key, slider_value = None, None
    if isinstance(ctx.triggered_id, dict) and 'index' in ctx.triggered_id:
        slider_key = ctx.triggered_id['index']; slider_value = triggered_input['value']
    elif slider_ids_obj_list and slider_values:
        prop_id_str = triggered_input['prop_id']
        for i, id_obj in enumerate(slider_ids_obj_list):
            if isinstance(id_obj, dict) and json.dumps(id_obj, sort_keys=True) in prop_id_str: slider_key = id_obj.get(
                'index'); slider_value = slider_values[i]; break
    if not slider_key or slider_value is None or slider_key not in DIM_KEYS: return no_update
    updated_states = states_data_json.copy();
    point_to_update = updated_states[selected_id].copy()
    point_to_update[slider_key] = float(slider_value);
    updated_states[selected_id] = point_to_update
    return updated_states


@app.callback(Output('selected-point-id-store', 'data'), Input('point-selector-dropdown', 'value'))
def update_selected_point_id_store_val_gm452(selected_id): return selected_id  # Renamed


@app.callback([Output('evolution-interval', 'disabled'), Output('toggle-evolution-button', 'children')],
              [Input('toggle-evolution-button', 'n_clicks')], [State('evolution-interval', 'disabled')])
def toggle_evolution_gm452(n_clicks, disabled_state):  # Renamed
    if n_clicks == 0: return True, '开始演化'
    is_now_disabled = not disabled_state
    return is_now_disabled, '暂停演化' if not is_now_disabled else '开始演化'


@app.callback(Output('evolution-interval', 'interval'), Input('evolution-interval-input', 'value'))
def update_evolution_interval_time_gm452(value): return int(value) if value and int(value) >= 100 else 1000  # Renamed


@app.callback(Output('evolution-params-store', 'data'), [Input('lr-slider', 'value'), Input('noise-slider', 'value')],
              [State('evolution-params-store', 'data')])
def update_evolution_hyperparams_gm452(lr, noise, params_json):  # Renamed
    if lr is None or noise is None: raise PreventUpdate
    new_params = json.loads(json.dumps(params_json));
    new_params['learning_rate'] = float(lr);
    new_params['noise_level'] = float(noise)
    return new_params


@app.callback(
    [Output('world-states-store', 'data'), Output('n-intervals-display', 'children'),
     Output('event-log-store', 'data')],
    [Input('evolution-interval', 'n_intervals'), Input('step-evolution-button', 'n_clicks')],
    [State('world-states-store', 'data'), State('evolution-params-store', 'data'),
     State('evolution-interval', 'disabled'),
     State('coord-type-dropdown', 'value'), State('event-log-store', 'data'), State('global-env-factors-store', 'data')]
)
def run_evolution_step_advanced_gm452(n_auto, n_manual, states_json, evo_json, interval_disabled, coord_type, log_list,
                                      env_json):  # Renamed
    ctx = callback_context;
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None
    if not triggered_id or (triggered_id == 'evolution-interval' and interval_disabled) or not all(
        [states_json, evo_json, env_json]): raise PreventUpdate

    obj_dict = {name: WorldState.from_dict(data) for name, data in states_json.items()}
    metrics = {
        'avg_b1_resource': np.mean([s.b1_resource for s in obj_dict.values() if s]) if obj_dict else 0,
        'avg_y1_clarity': np.mean([s.y1_clarity for s in obj_dict.values() if s]) if obj_dict else 0,
        'avg_h1_possibilities': np.mean([s.h1_possibilities for s in obj_dict.values() if s]) if obj_dict else 0,
        'avg_s1_trust': np.mean([s.s1_trustworthiness for s in obj_dict.values() if s]) if obj_dict else 0,
        'avg_s2_reputation': np.mean([s.s2_reputation for s in obj_dict.values() if s]) if obj_dict else 0,
    }
    effects_by_event, trigger_msgs = event_manager.process_step(obj_dict, metrics, env_json)
    updated_json_out = {};
    step_internal_logs = []

    for name, obj in obj_dict.items():
        active_effs = effects_by_event.get(name, {})
        try:
            obj.evolve(evo_json, obj_dict, active_effs, env_json)
            weights = evo_json.get('plot_weights', default_evolution_params_gm452['plot_weights'])
            obj.record_history(coord_type, weights['b'], weights['y'], weights['h'])
            updated_json_out[name] = obj.to_dict()
            if obj.active_effects_log: step_internal_logs.extend(
                [f"状态 '{obj.name_zh}': {log}" for log in obj.active_effects_log])
        except Exception as e:
            print(f"演化状态 {name} 出错: {e}, 数据: {states_json.get(name)}")
            updated_json_out[name] = states_json[name]

    if not isinstance(log_list, list): log_list = []
    new_entries = []
    if trigger_msgs: new_entries.extend(trigger_msgs)
    if step_internal_logs: new_entries.extend(step_internal_logs)
    if new_entries:
        timestamp = f"--- 步骤 {n_auto if triggered_id == 'evolution-interval' else '(手动)'} ({time.strftime('%H:%M:%S')}) ---"
        combined_logs = log_list + [timestamp] + new_entries
    else:
        combined_logs = log_list
    final_log_store = combined_logs[-MAX_LOG_LINES:]
    step_info = f"自动迭代: {n_auto}" if triggered_id == 'evolution-interval' else f"手动步进 (总: {n_manual})"
    return updated_json_out, step_info, final_log_store


@app.callback(Output('event-log-textarea', 'value'), Input('event-log-store', 'data'))
def update_event_log_display_gm452(log_data_list):  # Renamed
    if isinstance(log_data_list, list): return "\n".join(log_data_list)
    return "事件日志为空或格式错误."


@app.callback(
    [Output('world-states-store', 'data', allow_duplicate=True),
     Output('evolution-interval', 'n_intervals', allow_duplicate=True),
     Output('event-log-store', 'data', allow_duplicate=True)],
    [Input('reset-states-button', 'n_clicks')], prevent_initial_call=True
)
def reset_all_states_gm452(n_clicks):  # Renamed
    if n_clicks is None or n_clicks == 0: raise PreventUpdate
    fresh_data = {}
    temp_list = [WorldState(**s_dict) for s_dict in initial_states_templates_gm452]  # Use correct template
    for obj in temp_list:
        obj.neighbors = neighbor_config_gm452.get(obj.name_en, [])  # Use correct neighbor config
        fresh_data[obj.name_en] = obj.to_dict()
    event_manager.reset_events()  # event_manager must be defined
    return fresh_data, 0, ["状态和事件已重置."]


@app.callback(
    Output('main-3d-scatter-plot', 'figure'),
    [Input('world-states-store', 'data'), Input('coord-type-dropdown', 'value')],
    [State('evolution-params-store', 'data')]
)
def update_3d_scatter_plot_gm452(states_data_json, coord_type, evo_params_json):  # Renamed and using corrected logic
    # (Using corrected version from GM4.5.1)
    if not states_data_json:
        return go.Figure(layout=go.Layout(title="数据加载中...",
                                          scene=dict(xaxis=dict(range=[0, 10]), yaxis=dict(range=[0, 10]),
                                                     zaxis=dict(range=[0, 10]), aspectmode='cube')))
    traces = []
    # Ensure evo_params_json and its plot_weights are valid, use default if not.
    plot_weights = default_evolution_params_gm452['plot_weights']  # Use current version's default
    if evo_params_json and 'plot_weights' in evo_params_json and \
            isinstance(evo_params_json['plot_weights'], dict) and \
            all(k in evo_params_json['plot_weights'] for k in
                ['b', 'y', 'h']):  # Could add 's' here if composite plot uses it
        plot_weights = evo_params_json['plot_weights']

    state_items = list(states_data_json.items())
    for i, (state_id, state_dict) in enumerate(state_items):
        if not isinstance(state_dict, dict): continue
        try:
            state_obj = WorldState.from_dict(state_dict)
            current_coords = state_obj.get_coords_for_plot(
                coord_type,
                plot_weights.get('b'),
                plot_weights.get('y'),
                plot_weights.get('h')
                # If composite plot can use S dimensions, pass plot_weights.get('s') here
            )
        except Exception as e:
            print(f"Error in plot for {state_id}: {e}")  # Print error for debugging
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
            marker=dict(size=11, opacity=0.9, color=marker_color_value, colorscale='Viridis',
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
    app.run(debug=True, port=8061)  # New port for GM4.5.2
