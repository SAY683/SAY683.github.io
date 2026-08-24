import dash
from dash import dcc, html, Input, Output, State, callback_context, no_update, ALL
from dash.exceptions import PreventUpdate
import plotly.graph_objects as go
import numpy as np
import random
import json
import time
from collections import defaultdict, deque
import traceback
import uuid

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
MAX_LOG_LINES = 250


def sigmoid(x, k=1, x0=5):
    if isinstance(x, (np.ndarray, list)): x = np.array(x, dtype=float)
    x_clipped = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-k * (x_clipped - x0)))


def scale_value(value, old_min=0, old_max=10, new_min=0, new_max=1):
    if old_max == old_min: return new_min
    return new_min + (value - old_min) * (new_max - new_min) / (old_max - old_min)


SIMULATION_LOG = deque(maxlen=MAX_LOG_LINES * 2)


def log_message(level, message, source="System"):
    global SIMULATION_LOG
    timestamp = time.strftime('%H:%M:%S')
    log_entry = f"[{timestamp}][{level.upper()}][{source}] {message}"
    SIMULATION_LOG.append(log_entry)
    if level.upper() in ["ERROR", "CRITICAL"]: print(log_entry)


# --- Community Project Class ---
class CommunityProject:
    def __init__(self, project_id, name, required_b1, required_h2, duration,
                 creator_en, target_participants=2, reward_type='b2_reduction', reward_value=0.5):
        self.project_id = project_id;
        self.name = name
        self.required_b1_total = required_b1;
        self.required_h2_avg = required_h2
        self.duration_total = duration;
        self.creator_en = creator_en
        self.target_participants = target_participants
        self.reward_type = reward_type;
        self.reward_value = reward_value
        self.participants = {creator_en};
        self.contributed_b1 = 0
        self.current_duration = 0;
        self.status = "pending"

    def add_participant(self, state_en, contribution_b1):
        if len(self.participants) < self.target_participants and self.status == "pending":
            self.participants.add(state_en);
            self.contributed_b1 += contribution_b1
            log_message("INFO", f"{state_en} 加入项目 '{self.name}', 贡献B1: {contribution_b1:.2f}", "CommunityProject")
            avg_h2_of_participants = 5.0  # Placeholder - this needs real calculation if used
            # Real H2 check would require passing all_states_objects_dict here to get H2 of participants
            if len(self.participants) == self.target_participants and \
                    self.contributed_b1 >= self.required_b1_total:  # and avg_h2_of_participants >= self.required_h2_avg:
                self.status = "active"
                log_message("INFO", f"项目 '{self.name}' 已激活!", "CommunityProject")
            return True
        return False

    def progress_project(self, all_states_objects_dict):
        if self.status != "active": return
        self.current_duration += 1
        if self.current_duration >= self.duration_total:
            self.status = "completed"
            log_message("INFO", f"项目 '{self.name}' 已完成!", "CommunityProject")
            self.distribute_rewards(all_states_objects_dict)

    def distribute_rewards(self, all_states_objects_dict):
        if self.status != "completed": return
        for p_name in self.participants:
            if p_name in all_states_objects_dict and all_states_objects_dict[p_name]:
                p_obj = all_states_objects_dict[p_name]
                reward_applied = False
                if self.reward_type == 'b1_gain':
                    p_obj.b1_resource = np.clip(p_obj.b1_resource + self.reward_value, 0, 10);
                    reward_applied = True
                elif self.reward_type == 'b2_reduction':
                    p_obj.b2_limitation = np.clip(p_obj.b2_limitation - self.reward_value, 0, 10);
                    reward_applied = True
                elif self.reward_type == 'h1_boost':
                    p_obj.h1_possibilities = np.clip(p_obj.h1_possibilities + self.reward_value, 0, 10);
                    reward_applied = True
                if reward_applied:
                    p_obj.active_effects_log.append(
                        f"项目'{self.name}'奖励: {self.reward_type} {self.reward_value:.2f}")
                    log_message("INFO", f"项目'{self.name}'奖励已分配给 {p_name}", "CommunityProject")

    def to_dict(self):
        return {'project_id': self.project_id, 'name': self.name, 'required_b1_total': self.required_b1_total,
                'required_h2_avg': self.required_h2_avg, 'duration_total': self.duration_total,
                'creator_en': self.creator_en, 'target_participants': self.target_participants,
                'reward_type': self.reward_type, 'reward_value': self.reward_value,
                'participants': list(self.participants), 'contributed_b1': self.contributed_b1,
                'current_duration': self.current_duration, 'status': self.status}

    @classmethod
    def from_dict(cls, data):
        proj = cls(data['project_id'], data['name'], data['required_b1_total'], data['required_h2_avg'],
                   data['duration_total'], data['creator_en'], data.get('target_participants', 2),
                   data['reward_type'], data['reward_value'])
        proj.participants = set(data.get('participants', []));
        proj.contributed_b1 = data.get('contributed_b1', 0)
        proj.current_duration = data.get('current_duration', 0);
        proj.status = data.get('status', "pending")
        return proj


# --- WorldState Class ---
class WorldState:
    def __init__(self, name_zh, name_en, b1_res, b2_lim, y1_cla, y2_dri, y3_asp,
                 h1_pos, h2_inn, h3_ris, s1_tru=5.0, s2_rep=3.0):
        self.name_zh = name_zh;
        self.name_en = name_en
        dim_values = {'b1_resource': b1_res, 'b2_limitation': b2_lim, 'y1_clarity': y1_cla, 'y2_drive': y2_dri,
                      'y3_aspiration': y3_asp, 'h1_possibilities': h1_pos, 'h2_innovation': h2_inn,
                      'h3_risk_appetite': h3_ris, 's1_trustworthiness': s1_tru, 's2_reputation': s2_rep}
        for key in DIM_KEYS: setattr(self, key, np.clip(float(dim_values.get(key, 0)), 0, 10))
        self.history = [];
        self.neighbors = [];
        self.trust_levels = defaultdict(lambda: 5.0)
        self.active_effects_log = [];
        self.last_risk_outcome_factor = 0.0
        self.alliance_partners = set();
        self.rivals = set();
        self.social_interaction_cooldowns = defaultdict(int)
        self.perceived_b1_resource = self.b1_resource;
        self.perceived_h1_possibilities = self.h1_possibilities
        self.perception_accuracy = scale_value(self.y1_clarity, 0, 10, 0.5, 1.0);
        self.mood = 0
        self.community_norm_conformity = random.uniform(0.3, 0.7);
        self.contributed_projects = {}

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
            return delta_value * scale_value(current_value, min_val, min_val + boundary_threshold, 0.1, 1)
        elif current_value >= max_val - boundary_threshold and delta_value > 0:
            return delta_value * scale_value(current_value, max_val - boundary_threshold, max_val, 1, 0.1)
        return delta_value

    def update_perception(self, params):
        k_cog = params.get('coefficients', {}).get('cognitive_model', {})
        base_acc = scale_value(self.y1_clarity, 0, 10, k_cog.get('y1_to_acc_min', 0.4), k_cog.get('y1_to_acc_max', 1.0))
        h2_bonus = scale_value(self.h2_innovation, 0, 10, 0, k_cog.get('h2_to_acc_bonus', 0.1))
        self.perception_accuracy = np.clip(base_acc + h2_bonus, 0.1, 1.0)
        mood_bias = 0
        if self.mood > 0:
            mood_bias = k_cog.get('mood_pos_bias_factor', 0.1) * self.mood
        elif self.mood < 0:
            mood_bias = k_cog.get('mood_neg_bias_factor', -0.1) * abs(self.mood)
        err_scale = params.get('perception_error_scale', 5.0)
        b1_err = (1 - self.perception_accuracy) * err_scale;
        self.perceived_b1_resource = np.clip(
            self.b1_resource + random.uniform(-b1_err, b1_err) + (self.b1_resource * mood_bias), 0, 10)
        h1_err = (1 - self.perception_accuracy) * params.get('perception_error_scale_h1', 4.0);
        self.perceived_h1_possibilities = np.clip(
            self.h1_possibilities + random.uniform(-h1_err, h1_err) + (self.h1_possibilities * mood_bias), 0, 10)

    def update_mood(self, params, risk_outcome_this_step):
        k_cog = params.get('coefficients', {}).get('cognitive_model', {})
        change = 0
        if risk_outcome_this_step > params.get('mood_risk_success_thresh', 0.1):
            change += k_cog.get('mood_from_risk_success', 0.5) * scale_value(risk_outcome_this_step, 0.1, 1, 0, 1)
        elif risk_outcome_this_step < params.get('mood_risk_failure_thresh', -0.1):
            change += k_cog.get('mood_from_risk_failure', -0.5) * scale_value(abs(risk_outcome_this_step), 0.1, 1, 0, 1)
        if self.b1_resource < params.get('mood_b1_low_thresh', 2.0):
            change -= k_cog.get('mood_from_low_b1', 0.1)
        elif self.b1_resource > params.get('mood_b1_high_thresh', 7.0):
            change += k_cog.get('mood_from_high_b1', 0.05)
        if len(self.alliance_partners) > len(self.rivals) and len(self.alliance_partners) > 0:
            change += k_cog.get('mood_from_alliances', 0.05)
        elif len(self.rivals) > len(self.alliance_partners) and len(self.rivals) > 0:
            change -= k_cog.get('mood_from_rivals', 0.05)
        self.mood += change * k_cog.get('mood_change_rate', 0.3)
        self.mood = np.clip(self.mood, k_cog.get('mood_min', -1.0), k_cog.get('mood_max', 1.0))
        self.mood *= k_cog.get('mood_decay_factor', 0.95)

    # --- Delta Calculation Methods (GM4.5.6 - includes Stage 2, 5 modifications) ---
    # (Copied from GM4.5.6 provided previously, assuming they are correct for this consolidation)
    def _calculate_delta_b1(self, k_dim, params, avg_b1_others, neighbor_effect_b1, risk_project_b1_change_this_step):
        effect_h2 = k_dim.get('from_h2', 0) * sigmoid(self.h2_innovation, k=0.7, x0=5);
        effect_y2 = k_dim.get('from_y2', 0) * scale_value(self.y2_drive, 0, 10, 0.3, 1)
        loss_b2 = k_dim.get('loss_b2', 0) * (self.b2_limitation / 10) ** 1.8;
        cost_h2_activity = k_dim.get('cost_h2_activity', 0) * self.h2_innovation * (1 + self.h2_innovation / 20)
        cost_y2_sustain = k_dim.get('cost_y2_sustain', 0) * self.y2_drive * (1 + self.y2_drive / 20);
        reputation_bonus_b1 = k_dim.get('from_s2_reputation', 0) * self.s2_reputation
        social_pressure_b1 = 0;
        if avg_b1_others is not None: social_pressure_b1 = k_dim.get('social_pressure', 0) * np.clip(
            avg_b1_others - self.b1_resource, -3, 3)
        base_consumption = params.get('b1_maintenance_base', 0.03);
        non_linear_maintenance = params.get('b1_maintenance_factor', 0.005) * (
                self.b1_resource / params.get('b1_maintenance_scale_ref', 10.0)) ** params.get(
            'b1_maintenance_exponent', 2.0)
        total_consumption = base_consumption + non_linear_maintenance;
        h2_to_b1_direct_gain = 0
        if self.h2_innovation > params.get('h2_to_b1_direct_thresh', 7.0) and self.y2_drive > params.get(
                'y2_for_h2_to_b1_direct_thresh', 6.0) and random.random() < k_dim.get('h2_to_b1_direct_prob', 0.01):
            h2_to_b1_direct_gain = k_dim.get('h2_to_b1_direct_factor', 0.05) * self.h2_innovation;
            self.active_effects_log.append(f"H2成果B1: +{h2_to_b1_direct_gain:.2f}")
        delta = (
                effect_h2 + effect_y2 - loss_b2 - total_consumption - cost_h2_activity - cost_y2_sustain + risk_project_b1_change_this_step + social_pressure_b1 + neighbor_effect_b1 + reputation_bonus_b1 + h2_to_b1_direct_gain)
        return self._apply_boundary_effect(self.b1_resource, delta)

    def _calculate_delta_b2(self, k_dim, params, neighbor_effect_b2, risk_project_b2_change_this_step):
        reduction_y2 = k_dim.get('reduce_y2', 0) * sigmoid(self.y2_drive, k=0.8, x0=3);
        reduction_h2 = k_dim.get('reduce_h2', 0) * sigmoid(self.h2_innovation, k=0.8, x0=3);
        random_event_b2 = 0
        if random.random() < params.get('b2_random_event_chance', 0.04): random_event_b2 = random.uniform(0,
                                                                                                          0.8) * k_dim.get(
            'random_factor', 0.15)
        over_extension_factor = 0
        if self.y2_drive > params.get('b2_y2_overextension_thresh', 9.0): over_extension_factor += (
                                                                                                           self.y2_drive - params.get(
                                                                                                       'b2_y2_overextension_thresh',
                                                                                                       9.0)) * k_dim.get(
            'from_y2_overextension', 0.015)
        if self.h2_innovation > params.get('b2_h2_overextension_thresh', 9.0): over_extension_factor += (
                                                                                                                self.h2_innovation - params.get(
                                                                                                            'b2_h2_overextension_thresh',
                                                                                                            9.0)) * k_dim.get(
            'from_h2_overextension', 0.01)
        h2_to_b2_reduction = 0
        if self.h2_innovation > params.get('h2_to_b2_reduction_thresh', 7.5) and self.y1_clarity > params.get(
                'y1_for_h2_to_b2_reduction_thresh', 6.5) and random.random() < k_dim.get('h2_to_b2_reduction_prob',
                                                                                         0.008):
            h2_to_b2_reduction = k_dim.get('h2_to_b2_reduction_factor', 0.06) * self.h2_innovation;
            self.active_effects_log.append(f"H2成果降低B2: -{h2_to_b2_reduction:.2f}")
        delta = -(
                reduction_y2 + reduction_h2) - h2_to_b2_reduction + random_event_b2 + over_extension_factor + risk_project_b2_change_this_step + neighbor_effect_b2
        return self._apply_boundary_effect(self.b2_limitation, delta)

    def _calculate_delta_y1(self, k_dim, params, neighbor_effect_y1, current_risk_outcome_this_step):
        effect_experience_validation = 0
        if current_risk_outcome_this_step > params.get('risk_success_return_threshold_for_y1', 0.03):
            effect_experience_validation = k_dim.get('from_success_validation', 0) * sigmoid(
                current_risk_outcome_this_step, k=5, x0=0.1)
        elif current_risk_outcome_this_step < -params.get('risk_failure_loss_threshold_for_y1', 0.03):
            effect_experience_validation = k_dim.get('from_failure_doubt', 0) * sigmoid(current_risk_outcome_this_step,
                                                                                        k=-5, x0=-0.1)
        loss_b2 = k_dim.get('loss_b2', 0) * (self.b2_limitation / 7) ** 2.0;
        perceived_reality_measure = (self.perceived_b1_resource + self.perceived_h1_possibilities) / 2
        aspiration_reality_gap = self.y3_aspiration - perceived_reality_measure;
        loss_from_gap = 0
        if aspiration_reality_gap > params.get('y1_gap_threshold_for_loss', 3.0): loss_from_gap = k_dim.get(
            'loss_aspiration_gap', 0) * (aspiration_reality_gap - params.get('y1_gap_threshold_for_loss', 3.0))
        y1_maintenance_cost = 0
        if self.y1_clarity > params.get('y1_high_maintenance_thresh', 8.0):
            grounding_factor = (scale_value(self.b1_resource, 0, 5, 0.1, 1) + scale_value(self.h1_possibilities, 0, 5,
                                                                                          0.1, 1)) / 2
            y1_maintenance_cost = k_dim.get('high_y1_decay_factor', 0.005) * (
                    self.y1_clarity - params.get('y1_high_maintenance_thresh', 8.0)) / (grounding_factor + 0.1)
        b_state_y1_adjustment = 0
        if self.b1_resource < params.get('b1_for_y1_erosion_thresh', 2.0) and self.b2_limitation > params.get(
                'b2_for_y1_erosion_thresh', 7.0):
            b_state_y1_adjustment -= k_dim.get('b_state_y1_erosion_factor', 0.01) * (
                    1 - scale_value(self.y1_clarity, 0, 5, 0, 1))
        elif self.b1_resource > params.get('b1_for_y1_affirm_thresh', 7.0) and self.b2_limitation < params.get(
                'b2_for_y1_affirm_thresh', 3.0) and self.y2_drive > params.get('y2_for_y1_affirm_thresh', 5.0):
            b_state_y1_adjustment += k_dim.get('b_state_y1_affirm_factor', 0.005) * (
                    1 - scale_value(self.y1_clarity, 5, 10, 0, 1))
        delta = effect_experience_validation - loss_b2 - loss_from_gap - y1_maintenance_cost + b_state_y1_adjustment + neighbor_effect_y1
        return self._apply_boundary_effect(self.y1_clarity, delta)

    def _calculate_delta_y2(self, k_dim, params, neighbor_effect_y2, current_risk_outcome_this_step):
        effect_y1 = k_dim.get('from_y1', 0) * sigmoid(self.y1_clarity, k=0.8, x0=3.5);
        aspiration_gap_y2 = self.y3_aspiration - self.y2_drive
        effect_y3 = k_dim.get('from_y3', 0) * sigmoid(aspiration_gap_y2, k=0.4, x0=0.5) * (
                1 - sigmoid(self.y2_drive, k=params.get('y2_saturation_k', 1.0),
                            x0=params.get('y2_saturation_x0', 8.0)))
        effect_risk_outcome = 0
        if current_risk_outcome_this_step > params.get('risk_success_return_threshold_for_y2', 0.02):
            effect_risk_outcome = k_dim.get('from_risk_success_激励', 0) * sigmoid(current_risk_outcome_this_step, k=6,
                                                                                   x0=0.05)
        elif current_risk_outcome_this_step < -params.get('risk_failure_loss_threshold_for_y2', 0.02):
            effect_risk_outcome = k_dim.get('from_risk_failure_打击', 0) * sigmoid(current_risk_outcome_this_step, k=-6,
                                                                                   x0=-0.05)
        loss_b2 = k_dim.get('loss_b2', 0) * (self.b2_limitation / 9) ** 1.5;
        loss_low_y1 = k_dim.get('loss_low_y1', 0) * (1 - sigmoid(self.y1_clarity, k=1, x0=1.5));
        sustain_cost_low_b1 = 0
        if self.b1_resource < params.get('y2_b1_sustain_threshold', 2.0): sustain_cost_low_b1 = k_dim.get(
            'sustain_cost_low_b1', 0) * (params.get('y2_b1_sustain_threshold', 2.0) - self.b1_resource)
        y2_burnout_factor = 0
        if self.y2_drive > params.get('y2_burnout_thresh', 7.5):
            y2_burnout_factor = k_dim.get('burnout_factor_base', 0.012) * (
                    self.y2_drive - params.get('y2_burnout_thresh', 7.5)) * (
                                        1.5 - scale_value(self.b1_resource, 0, 7, 0.3, 1.0)) * (
                                        1.5 - scale_value(self.y1_clarity, 0, 7, 0.3, 1.0))
            y2_burnout_factor = max(0, y2_burnout_factor)
        b_state_y2_adjustment = 0
        if self.b1_resource < params.get('b1_for_y2_sap_thresh', 1.5) or self.b2_limitation > params.get(
                'b2_for_y2_sap_thresh', 7.5):
            b_state_y2_adjustment -= k_dim.get('b_state_y2_sap_factor', 0.012) * (
                    1 - scale_value(self.y2_drive, 0, 4, 0, 1))
        elif self.b1_resource > params.get('b1_for_y2_boost_thresh', 6.5) and self.y1_clarity > params.get(
                'y1_for_y2_boost_thresh', 6.0):
            b_state_y2_adjustment += k_dim.get('b_state_y2_boost_factor', 0.006) * (
                    1 - scale_value(self.y2_drive, 6, 10, 0, 1))
        delta = effect_y1 + effect_y3 + effect_risk_outcome - loss_b2 - loss_low_y1 - sustain_cost_low_b1 - y2_burnout_factor + b_state_y2_adjustment + neighbor_effect_y2
        return self._apply_boundary_effect(self.y2_drive, delta)

    def _calculate_delta_y3(self, k_dim, params, neighbor_effect_y3, avg_b1_others, avg_h1_others):
        adjustment_from_y2 = k_dim.get('adjust_y2', 0) * (self.y2_drive - self.y3_aspiration) * 0.025;
        boost_y1 = k_dim.get('boost_y1', 0) * sigmoid(self.y1_clarity - 6.0, k=0.9, x0=0)
        social_norm_b1_factor = 0;
        if avg_b1_others is not None: social_norm_b1_factor = k_dim.get('social_norm_b1', 0) * (
                avg_b1_others + params.get('y3_social_b1_offset', 1.0) - self.y3_aspiration)
        social_norm_h1_factor = 0;
        if avg_h1_others is not None: social_norm_h1_factor = k_dim.get('social_norm_h1', 0) * (
                avg_h1_others + params.get('y3_social_h1_offset', 0.5) - self.y3_aspiration)
        self_h1_factor = k_dim.get('self_h1_factor', 0) * (self.perceived_h1_possibilities - self.y3_aspiration) * (
                1 - sigmoid(self.y3_aspiration, k=params.get('y3_aspiration_h1_influence_damp_k', 1.5),
                            x0=params.get('y3_aspiration_h1_influence_damp_x0', 8.5)))
        reality_crush = 0
        if self.perceived_b1_resource < self.y3_aspiration - params.get('y3_reality_gap_threshold',
                                                                        4.0): reality_crush = k_dim.get(
            'loss_reality_gap', 0) * (self.y3_aspiration - self.perceived_b1_resource - params.get(
            'y3_reality_gap_threshold', 4.0))
        y3_complacency_drag = 0
        if self.y3_aspiration > params.get('y3_complacency_thresh', 8.5):
            if (self.y3_aspiration - self.perceived_b1_resource > params.get('y3_complacency_b1_gap', 4.0)) or (
                    self.y3_aspiration - self.perceived_h1_possibilities > params.get('y3_complacency_h1_gap', 4.0)):
                y3_complacency_drag = k_dim.get('complacency_drag_factor', 0.006) * (
                        self.y3_aspiration - params.get('y3_complacency_thresh', 8.5))
        b_success_y3_lift = 0
        if self.b1_resource > params.get('b1_for_y3_lift_thresh', 7.5) and self.b2_limitation < params.get(
                'b2_for_y3_lift_thresh', 2.5) and self.y2_drive > params.get('y2_for_y3_lift_thresh', 6.5):
            b_success_y3_lift = k_dim.get('b_success_y3_lift_factor', 0.004) * (
                    params.get('y3_target_after_b_success', 9.0) - self.y3_aspiration)
        delta = adjustment_from_y2 + boost_y1 + social_norm_b1_factor + social_norm_h1_factor + self_h1_factor - reality_crush - y3_complacency_drag + b_success_y3_lift + neighbor_effect_y3
        return self._apply_boundary_effect(self.y3_aspiration, delta)

    def _calculate_delta_h1(self, k_dim, params, neighbor_effect_h1):
        effect_b1 = k_dim.get('from_b1', 0) * sigmoid(self.b1_resource, k=0.6, x0=3.5) * (
                1.1 - sigmoid(self.h1_possibilities, k=params.get('h1_b1_influence_damp_k', 1),
                              x0=params.get('h1_b1_influence_damp_x0', 8)))
        effect_h2 = k_dim.get('from_h2', 0) * sigmoid(self.h2_innovation, k=0.7, x0=3.0);
        loss_b2 = k_dim.get('loss_b2', 0) * (self.b2_limitation / 6) ** 2.0
        loss_low_y2_y1 = k_dim.get('loss_low_y_factor', 0) * ((1 - scale_value(self.y2_drive, 0, 3.5, 0, 1)) + (
                1 - scale_value(self.y1_clarity, 0, 3.5, 0, 1))) / 2;
        h1_focus_cost = 0
        if self.h1_possibilities > params.get('h1_focus_cost_thresh', 8.5):
            h1_focus_cost = k_dim.get('focus_cost_factor', 0.008) * (
                    self.h1_possibilities - params.get('h1_focus_cost_thresh', 8.5)) * (
                                    1.2 - scale_value(self.y1_clarity, 0, 7, 0.2, 1.0));
            h1_focus_cost = max(0, h1_focus_cost)
        b2_direct_h1_suppression = 0
        if self.b2_limitation > params.get('b2_h1_suppression_thresh', 6.0): b2_direct_h1_suppression = k_dim.get(
            'b2_h1_suppression_factor', 0.01) * (self.b2_limitation - params.get('b2_h1_suppression_thresh',
                                                                                 6.0)) * scale_value(
            self.h1_possibilities, 0, 10, 0.3, 1)
        delta = effect_b1 + effect_h2 - loss_b2 - loss_low_y2_y1 - h1_focus_cost - b2_direct_h1_suppression + neighbor_effect_h1
        return self._apply_boundary_effect(self.h1_possibilities, delta)

    def _calculate_delta_h2(self, k_dim, params, neighbor_effect_h2):
        synergy_y1_y2 = self.y1_clarity * self.y2_drive / 100
        effect_y2_h3 = (k_dim.get('from_y2', 0) * self.y2_drive + k_dim.get('from_h3', 0) * self.h3_risk_appetite) * (
                0.4 + 0.6 * sigmoid(synergy_y1_y2, k=0.1, x0=(5 * 6 / 100))) * (
                               1.1 - sigmoid(self.h2_innovation, k=params.get('h2_innovation_saturation_k', 1),
                                             x0=params.get('h2_innovation_saturation_x0', 8.5)))
        practice_factor = (scale_value(self.y2_drive, 0, 10, 0.1, 1) + scale_value(self.h3_risk_appetite, 0, 10, 0.1,
                                                                                   1)) / 2
        decay = k_dim.get('decay_no_practice', 0) * (1.1 - practice_factor) * (
                self.h2_innovation / params.get('h2_decay_scale_ref', 9.0));
        h2_complexity_cost = 0
        if self.h2_innovation > params.get('h2_complexity_thresh', 8.0):
            integration_capacity = (scale_value(self.b1_resource, 0, 6, 0.2, 1.0) + scale_value(self.y1_clarity, 0, 6,
                                                                                                0.2, 1.0)) / 2
            h2_complexity_cost = k_dim.get('complexity_cost_factor', 0.007) * (
                    self.h2_innovation - params.get('h2_complexity_thresh', 8.0)) / (integration_capacity + 0.1)
        y3_h2_drive = 0
        if self.y3_aspiration > params.get('y3_for_h2_drive_thresh', 7.0) and self.y1_clarity > params.get(
                'y1_for_h2_drive_thresh', 6.0) and self.b1_resource > params.get('b1_for_h2_drive_thresh', 3.0):
            y3_h2_drive = k_dim.get('y3_h2_drive_factor', 0.005) * (
                    self.y3_aspiration - params.get('y3_for_h2_drive_thresh', 7.0)) * (
                                  1 - scale_value(self.h2_innovation, 7, 10, 0, 1))
        delta = effect_y2_h3 + y3_h2_drive - decay - h2_complexity_cost + neighbor_effect_h2
        return self._apply_boundary_effect(self.h2_innovation, delta)

    def _calculate_delta_h3(self, k_dim, params, neighbor_effect_h3, current_risk_outcome_this_step):
        effect_risk_outcome = 0;
        risk_outcome_h3_dampening = (
                1 - sigmoid(abs(self.h3_risk_appetite - 5), k=params.get('h3_risk_feedback_damp_k', 0.3),
                            x0=params.get('h3_risk_feedback_damp_x0', 4)))
        if current_risk_outcome_this_step > params.get('risk_success_return_threshold_for_h3', 0.03):
            effect_risk_outcome = k_dim.get('from_risk_success_回报', 0) * sigmoid(current_risk_outcome_this_step, k=8,
                                                                                   x0=0.05) * risk_outcome_h3_dampening
        elif current_risk_outcome_this_step < -params.get('risk_failure_loss_threshold_for_h3', 0.03):
            effect_risk_outcome = k_dim.get('from_risk_failure_惩罚', 0) * sigmoid(current_risk_outcome_this_step, k=-8,
                                                                                   x0=-0.05) * risk_outcome_h3_dampening
        effect_y1 = k_dim.get('from_y1', 0) * sigmoid(self.y1_clarity, k=0.7, x0=6.0);
        effect_y2 = k_dim.get('from_y2', 0) * sigmoid(self.y2_drive, k=0.7, x0=6.0);
        loss_b2 = k_dim.get('loss_b2', 0) * (self.b2_limitation / 10) ** 1.3;
        stability_caution = 0
        if self.b1_resource > params.get('h3_b1_stability_thresh', 8.0) and self.b2_limitation < params.get(
                'h3_b2_stability_thresh', 2.0): stability_caution = k_dim.get('stability_caution_factor', 0.006) * (
                self.h3_risk_appetite - params.get('h3_stable_target_risk', 3.0))
        desperation_risk_push = 0
        if self.b1_resource < params.get('h3_b1_desperation_thresh', 2.0) and self.y2_drive > params.get(
                'h3_y2_desperation_thresh', 6.0): desperation_risk_push = k_dim.get('desperation_risk_factor',
                                                                                    0.004) * (
                                                                                  params.get('h3_desperate_target_risk',
                                                                                             7.0) - self.h3_risk_appetite)
        delta = effect_risk_outcome + effect_y1 + effect_y2 - loss_b2 - stability_caution + desperation_risk_push + neighbor_effect_h3
        return self._apply_boundary_effect(self.h3_risk_appetite, delta)

    def _calculate_delta_s1_trustworthiness(self, k_dim, params, neighbor_feedback_s1, current_risk_outcome_this_step):
        delta = 0;
        consistency_factor = sigmoid(self.y1_clarity - 5, k=0.8, x0=0) * sigmoid(
            (self.b1_resource + self.h2_innovation) / 2 - (self.y3_aspiration - 2), k=0.6, x0=0)
        delta += k_dim.get('from_consistency', 0) * consistency_factor * (
                1.1 - sigmoid(self.s1_trustworthiness, k=1, x0=8.5))
        if current_risk_outcome_this_step < params.get('s1_risk_failure_penalty_thresh', -0.3): delta -= k_dim.get(
            'penalty_risk_failure', 0) * abs(current_risk_outcome_this_step)
        delta += neighbor_feedback_s1;
        delta -= k_dim.get('decay', 0) * (self.s1_trustworthiness / params.get('s_decay_scale_ref', 10.0))
        return self._apply_boundary_effect(self.s1_trustworthiness, delta)

    def _calculate_delta_s2_reputation(self, k_dim, params, neighbor_feedback_s2, current_risk_outcome_this_step):
        delta = 0;
        achievement_factor = (scale_value(self.b1_resource, 3, 10, 0, 1) + scale_value(self.h2_innovation, 4, 10, 0,
                                                                                       1)) / 2
        delta += k_dim.get('from_achievement', 0) * achievement_factor * (
                1.1 - sigmoid(self.s2_reputation, k=1, x0=8.5))
        value_appeal_factor = sigmoid(self.y1_clarity - 6, k=0.7, x0=0) * sigmoid(self.y3_aspiration - 6, k=0.7, x0=0)
        delta += k_dim.get('from_value_appeal', 0) * value_appeal_factor * (
                1.1 - sigmoid(self.s2_reputation, k=1, x0=8.5))
        if current_risk_outcome_this_step > params.get('s2_risk_success_bonus_thresh', 0.15): delta += k_dim.get(
            'bonus_risk_success', 0) * current_risk_outcome_this_step
        delta += neighbor_feedback_s2;
        delta -= k_dim.get('decay', 0) * (self.s2_reputation / params.get('s_decay_scale_ref', 10.0))
        return self._apply_boundary_effect(self.s2_reputation, delta)

    def _calculate_neighbor_effects(self, params, all_states_objects_dict):
        k_social = params.get('coefficients', {}).get('social_interactions', {});
        effects = {key: 0.0 for key in DIM_KEYS};
        num_valid_neighbors = 0
        if not self.neighbors: return effects
        sum_neighbor_y1, sum_neighbor_y2, sum_neighbor_y3, sum_neighbor_s1, sum_neighbor_s2 = 0, 0, 0, 0, 0
        for neighbor_name in self.neighbors:
            if neighbor_name in all_states_objects_dict and neighbor_name != self.name_en:
                neighbor_obj = all_states_objects_dict[neighbor_name];
                if neighbor_obj is None: continue
                num_valid_neighbors += 1;
                trust_target = neighbor_obj.s1_trustworthiness
                trust_delta = k_social.get('trust_formation_rate', 0) * (
                        trust_target - self.trust_levels[neighbor_name]) * scale_value(self.s1_trustworthiness, 0,
                                                                                       10, 0.8, 1.2)
                self.trust_levels[neighbor_name] = np.clip(self.trust_levels[neighbor_name] + trust_delta, 0, 10);
                current_trust_in_neighbor = self.trust_levels[neighbor_name]
                sum_neighbor_y1 += neighbor_obj.y1_clarity;
                sum_neighbor_y2 += neighbor_obj.y2_drive;
                sum_neighbor_y3 += neighbor_obj.y3_aspiration;
                sum_neighbor_s1 += neighbor_obj.s1_trustworthiness;
                sum_neighbor_s2 += neighbor_obj.s2_reputation
                coop_factor_trust = scale_value(current_trust_in_neighbor, 0, 10,
                                                k_social.get('trust_effect_min_coop', 0.5),
                                                k_social.get('trust_effect_max_coop', 1.5))
                b1_diff = neighbor_obj.b1_resource - self.b1_resource;
                b1_comp_loss_mod = 1.0;
                b1_coop_gain_mod = 1.0
                if neighbor_name in self.rivals:
                    b1_comp_loss_mod = k_social.get('rival_comp_loss_multiplier', 1.5);
                    b1_coop_gain_mod = k_social.get(
                        'rival_coop_gain_multiplier', 0.3)
                elif neighbor_name in self.alliance_partners:
                    b1_comp_loss_mod = k_social.get('alliance_comp_loss_multiplier',
                                                    0.5);
                    b1_coop_gain_mod = k_social.get(
                        'alliance_coop_gain_multiplier', 1.5)
                if b1_diff > k_social.get('b1_comp_diff_thresh', 1.0):
                    effects['b1_resource'] -= k_social.get('b1_comp_loss_factor', 0) * b1_diff * (
                            k_social.get('base_comp_factor', 1.1) - coop_factor_trust * 0.5) * b1_comp_loss_mod
                elif abs(b1_diff) < k_social.get('b1_coop_diff_thresh', 0.6) and self.b1_resource > k_social.get(
                        'b1_coop_min_self_res', 2.0):
                    effects['b1_resource'] += k_social.get('b1_coop_gain_factor', 0) * min(self.b1_resource,
                                                                                           neighbor_obj.b1_resource) * coop_factor_trust * b1_coop_gain_mod
                h2_info_share_mod = 1.0
                if neighbor_name in self.rivals:
                    h2_info_share_mod = k_social.get('rival_h2_share_multiplier', 0.1)
                elif neighbor_name in self.alliance_partners:
                    h2_info_share_mod = k_social.get('alliance_h2_share_multiplier', 1.8)
                if neighbor_obj.h2_innovation > self.h2_innovation + k_social.get('h2_info_share_min_diff', 0.5):
                    info_gain_potential = (neighbor_obj.h2_innovation - self.h2_innovation) * k_social.get(
                        'h2_info_share_factor', 0)
                    trust_in_source_factor = scale_value(current_trust_in_neighbor, 0, 10, 0.3, 1.0);
                    source_reputation_factor = scale_value(neighbor_obj.s2_reputation, 0, 10, 0.5, 1.2);
                    self_openness_factor = scale_value(self.y1_clarity, 0, 10, 0.7, 1.0)
                    effects['h2_innovation'] = effects.get('h2_innovation',
                                                           0) + info_gain_potential * trust_in_source_factor * source_reputation_factor * self_openness_factor * h2_info_share_mod
                if neighbor_name in self.rivals and current_trust_in_neighbor < k_social.get('rival_harm_trust_thresh',
                                                                                             3.0):
                    sabotage_strength = (k_social.get('rival_harm_trust_thresh',
                                                      3.0) - current_trust_in_neighbor) * scale_value(
                        neighbor_obj.y2_drive, 0, 10, 0.5, 1.2)
                    effects['b2_limitation'] = effects.get('b2_limitation', 0) + k_social.get(
                        'rival_sabotage_b2_factor', 0.005) * sabotage_strength
                    effects['s2_reputation'] = effects.get('s2_reputation', 0) - k_social.get('rival_smear_s2_factor',
                                                                                              0.003) * sabotage_strength
        if num_valid_neighbors > 0:
            avg_neighbor_y1 = sum_neighbor_y1 / num_valid_neighbors;
            avg_neighbor_y2 = sum_neighbor_y2 / num_valid_neighbors;
            avg_neighbor_y3 = sum_neighbor_y3 / num_valid_neighbors;
            avg_neighbor_s1 = sum_neighbor_s1 / num_valid_neighbors;
            avg_neighbor_s2 = sum_neighbor_s2 / num_valid_neighbors
            confidence_factor_self = scale_value(self.s2_reputation, 0, 10, 0.5, 1.0);
            y_align_mod = 1.0
            num_allies = len(self.alliance_partners.intersection(self.neighbors));
            num_rivals = len(self.rivals.intersection(self.neighbors))
            if num_allies / num_valid_neighbors > k_social.get('alliance_majority_for_y_align_boost', 0.6):
                y_align_mod = k_social.get('alliance_y_align_multiplier', 1.3)
            elif num_rivals / num_valid_neighbors > k_social.get('rival_majority_for_y_align_reduction', 0.4):
                y_align_mod = k_social.get('rival_y_align_multiplier', 0.7)
            effects['y1_clarity'] = effects.get('y1_clarity', 0) + k_social.get('y1_alignment_factor', 0) * (
                    avg_neighbor_y1 - self.y1_clarity) * (1 - sigmoid(self.y1_clarity, k=1.2,
                                                                      x0=params.get('y_social_align_self_thresh',
                                                                                    7.5))) * (
                                            1.1 - confidence_factor_self) * y_align_mod
            effects['y2_drive'] = effects.get('y2_drive', 0) + k_social.get('y2_contagion_factor', 0) * (
                    avg_neighbor_y2 - self.y2_drive) * (1 - sigmoid(self.y2_drive, k=1.2,
                                                                    x0=params.get('y_social_align_self_thresh',
                                                                                  7.5))) * (
                                          1.1 - confidence_factor_self) * y_align_mod
            effects['y3_aspiration'] = effects.get('y3_aspiration', 0) + k_social.get('y3_alignment_factor', 0) * (
                    avg_neighbor_y3 - self.y3_aspiration) * (1 - sigmoid(self.y3_aspiration, k=1, x0=params.get(
                'y3_social_align_self_thresh', 8.0))) * (1.1 - confidence_factor_self) * y_align_mod
            effects['s1_trustworthiness'] = effects.get('s1_trustworthiness', 0) + k_social.get('s1_social_norm_factor',
                                                                                                0) * (
                                                    avg_neighbor_s1 - self.s1_trustworthiness)
            effects['s2_reputation'] = effects.get('s2_reputation', 0) + k_social.get('s2_social_pressure_factor',
                                                                                      0) * (
                                               avg_neighbor_s2 - self.s2_reputation)
            avg_neighbor_h3 = np.mean([s.h3_risk_appetite for s_name, s in all_states_objects_dict.items() if
                                       s_name in self.neighbors and s is not None]) if num_valid_neighbors > 0 else self.h3_risk_appetite
            h3_norm_pressure = (avg_neighbor_h3 - self.h3_risk_appetite) * k_social.get('h3_norm_pressure_factor',
                                                                                        0.002) * self.community_norm_conformity * (
                                       1 - scale_value(self.y1_clarity, 0, 10, 0, 0.8))
            effects['h3_risk_appetite'] = effects.get('h3_risk_appetite', 0) + h3_norm_pressure
        return effects

        ### MODIFIED: manage_social_relations with stricter checks and cleanup ###

    def manage_social_relations(self, params, all_states_objects_dict, current_step):
        k_social = params.get('coefficients', {}).get('social_interactions', {})
        social_action_interval = params.get('social_action_interval', 5)

        if current_step % social_action_interval != 0:
            return

        # Decay cooldowns first
        for key in list(self.social_interaction_cooldowns.keys()):
            self.social_interaction_cooldowns[key] -= social_action_interval
            if self.social_interaction_cooldowns[key] <= 0:
                del self.social_interaction_cooldowns[key]

        for neighbor_name in list(
                self.neighbors):  # Iterate over a copy if self.neighbors might change (not in this logic)
            if neighbor_name not in all_states_objects_dict or neighbor_name == self.name_en:
                continue
            neighbor_obj = all_states_objects_dict[neighbor_name]
            if not neighbor_obj:  # Should not happen if dict is clean
                continue

            my_trust_in_neighbor = self.trust_levels.get(neighbor_name, 5.0)  # My trust in them
            neighbor_trust_in_me = neighbor_obj.trust_levels.get(self.name_en, 5.0)  # Their trust in me

            cooldown_key_alliance = f"form_alliance_{neighbor_name}"
            cooldown_key_rivalry = f"form_rivalry_{neighbor_name}"
            cooldown_key_break_alliance = f"break_alliance_{neighbor_name}"  # Optional: cooldown for re-evaluating break
            cooldown_key_end_rivalry = f"end_rivalry_{neighbor_name}"  # Optional: cooldown for re-evaluating end

            # --- I. Attempt to form new relations OR break existing ones ---

            # A. Check for forming an ALLIANCE
            # Conditions: Not already allied, not already rival, not on alliance cooldown, AND meets alliance criteria
            if neighbor_name not in self.alliance_partners and \
                    neighbor_name not in self.rivals and \
                    self.social_interaction_cooldowns.get(cooldown_key_alliance, 0) <= 0:

                y1_similarity = 10 - abs(self.y1_clarity - neighbor_obj.y1_clarity)
                y3_similarity = 10 - abs(self.y3_aspiration - neighbor_obj.y3_aspiration)

                alliance_propensity = 0
                if my_trust_in_neighbor > k_social.get('alliance_form_my_trust_thresh', 7.2) and \
                        neighbor_trust_in_me > k_social.get('alliance_form_their_trust_thresh', 7.2):
                    alliance_propensity += k_social.get('alliance_trust_factor', 0.35)
                if y1_similarity > k_social.get('alliance_form_y1_sim_thresh', 7.5):
                    alliance_propensity += k_social.get('alliance_y1_sim_factor', 0.3)
                if y3_similarity > k_social.get('alliance_form_y3_sim_thresh', 7.0):
                    alliance_propensity += k_social.get('alliance_y3_sim_factor', 0.25)

                if random.random() < alliance_propensity * k_social.get('alliance_form_base_prob', 0.12):
                    # --- Stricter Check: Ensure neighbor ALSO doesn't see me as a rival ---
                    if self.name_en not in neighbor_obj.rivals:
                        self.alliance_partners.add(neighbor_name)
                        neighbor_obj.alliance_partners.add(self.name_en)  # Reciprocal

                        # If by some chance they were rivals before, clear that (shouldn't happen with above checks but defensive)
                        self.rivals.discard(neighbor_name)
                        neighbor_obj.rivals.discard(self.name_en)

                        log_msg = f"与 {neighbor_obj.name_zh} 结为联盟!"
                        self.active_effects_log.append(log_msg)
                        neighbor_obj.active_effects_log.append(f"与 {self.name_zh} 结为联盟!")
                        log_message("INFO", f"{self.name_zh} 与 {neighbor_obj.name_zh} 结盟", "SocialLogic")

                        self.social_interaction_cooldowns[cooldown_key_alliance] = params.get('alliance_cooldown',
                                                                                              20)
                        neighbor_obj.social_interaction_cooldowns[f"form_alliance_{self.name_en}"] = params.get(
                            'alliance_cooldown', 20)
                    else:
                        log_message("INFO",
                                    f"{self.name_en} to {neighbor_name}: Alliance blocked, target views self as rival.",
                                    "SocialLogic")


            # B. Check for forming a RIVALRY
            # Conditions: Not already allied, not already rival, not on rivalry cooldown, AND meets rivalry criteria
            # Using elif to ensure this block is only considered if alliance formation didn't happen with this neighbor in this step
            elif neighbor_name not in self.alliance_partners and \
                    neighbor_name not in self.rivals and \
                    self.social_interaction_cooldowns.get(cooldown_key_rivalry, 0) <= 0:

                b1_competition_diff = abs(self.b1_resource - neighbor_obj.b1_resource)
                is_driven_competition = (self.y2_drive > k_social.get('rival_form_self_y2_thresh', 6.5) or \
                                         neighbor_obj.y2_drive > k_social.get('rival_form_other_y2_thresh', 6.5))

                rivalry_propensity = 0
                if my_trust_in_neighbor < k_social.get('rival_form_my_trust_thresh', 2.8):
                    rivalry_propensity += k_social.get('rival_trust_factor', 0.45)
                if b1_competition_diff > k_social.get('rival_form_b1_diff_thresh', 2.5) and is_driven_competition:
                    rivalry_propensity += k_social.get('rival_b1_comp_factor', 0.35)

                if random.random() < rivalry_propensity * k_social.get('rival_form_base_prob', 0.1):
                    # --- Stricter Check: Ensure neighbor ALSO doesn't see me as an ally ---
                    if self.name_en not in neighbor_obj.alliance_partners:
                        self.rivals.add(neighbor_name)
                        neighbor_obj.rivals.add(self.name_en)  # Reciprocal

                        # If by some chance they were allies before, clear that
                        self.alliance_partners.discard(neighbor_name)
                        neighbor_obj.alliance_partners.discard(self.name_en)

                        log_msg = f"与 {neighbor_obj.name_zh} 成为对手!"
                        self.active_effects_log.append(log_msg)
                        neighbor_obj.active_effects_log.append(f"与 {self.name_zh} 成为对手!")
                        log_message("INFO", f"{self.name_zh} 与 {neighbor_obj.name_zh} 成为对手", "SocialLogic")

                        self.social_interaction_cooldowns[cooldown_key_rivalry] = params.get('rivalry_cooldown', 25)
                        neighbor_obj.social_interaction_cooldowns[f"form_rivalry_{self.name_en}"] = params.get(
                            'rivalry_cooldown', 25)
                    else:
                        log_message("INFO",
                                    f"{self.name_en} to {neighbor_name}: Rivalry blocked, target views self as ally.",
                                    "SocialLogic")


            # C. Check for BREAKING an existing ALLIANCE
            # Conditions: Currently allied, not on break alliance cooldown (optional), AND meets break conditions
            elif neighbor_name in self.alliance_partners and \
                    self.social_interaction_cooldowns.get(cooldown_key_break_alliance, 0) <= 0:  # Optional cooldown

                y1_divergence = abs(self.y1_clarity - neighbor_obj.y1_clarity)

                break_alliance = False
                if my_trust_in_neighbor < k_social.get('alliance_break_trust_thresh', 3.5):
                    break_alliance = True
                    reason = "low trust"
                elif y1_divergence > k_social.get('alliance_break_y1_diff_thresh', 6.0) and \
                        random.random() < k_social.get('alliance_break_prob_y_diverge', 0.04):
                    break_alliance = True
                    reason = "value divergence"

                if break_alliance:
                    self.alliance_partners.discard(neighbor_name)
                    neighbor_obj.alliance_partners.discard(self.name_en)  # 双向解除

                    # Explicitly ensure they are not rivals either after breaking alliance,
                    # unless conditions for rivalry are met *in a subsequent step*.
                    self.rivals.discard(neighbor_name)  # Safety cleanup
                    neighbor_obj.rivals.discard(self.name_en)  # Safety cleanup

                    log_msg = f"与 {neighbor_obj.name_zh} 解除联盟 (原因: {reason})."
                    self.active_effects_log.append(log_msg)
                    # neighbor_obj.active_effects_log.append(f"与 {self.name_zh} 的联盟已解除 (原因: {reason}).") # Avoid duplicate logs if B also processes A
                    log_message("INFO", f"{self.name_zh} 与 {neighbor_obj.name_zh} 解除联盟 (原因: {reason})",
                                "SocialLogic")

                    # self.social_interaction_cooldowns[cooldown_key_break_alliance] = params.get('alliance_break_cooldown', 10) # Optional
                    # No immediate attempt to form rivalry here; let next step's logic handle it if conditions are met.


            # D. Check for ENDING an existing RIVALRY
            # Conditions: Currently rivals, not on end rivalry cooldown (optional), AND meets end conditions
            elif neighbor_name in self.rivals and \
                    self.social_interaction_cooldowns.get(cooldown_key_end_rivalry, 0) <= 0:  # Optional cooldown

                end_rivalry = False
                if my_trust_in_neighbor > k_social.get('rival_end_trust_thresh', 6.5) and \
                        random.random() < k_social.get('rival_end_prob_trust_improve', 0.1):
                    end_rivalry = True
                    reason = "improved trust"
                # Could add other conditions, e.g., common powerful enemy, mutual benefit opportunity

                if end_rivalry:
                    self.rivals.discard(neighbor_name)
                    neighbor_obj.rivals.discard(self.name_en)  # 双向解除

                    # Explicitly ensure they are not allies either after ending rivalry,
                    # unless conditions for alliance are met *in a subsequent step*.
                    self.alliance_partners.discard(neighbor_name)  # Safety cleanup
                    neighbor_obj.alliance_partners.discard(self.name_en)  # Safety cleanup

                    log_msg = f"与 {neighbor_obj.name_zh} 结束敌对 (原因: {reason})."
                    self.active_effects_log.append(log_msg)
                    # neighbor_obj.active_effects_log.append(f"与 {self.name_zh} 的敌对已结束 (原因: {reason}).")
                    log_message("INFO", f"{self.name_zh} 与 {neighbor_obj.name_zh} 结束敌对 (原因: {reason})",
                                "SocialLogic")

                    # self.social_interaction_cooldowns[cooldown_key_end_rivalry] = params.get('rivalry_end_cooldown', 15) # Optional
                    # No immediate attempt to form alliance here.
        # End of loop through neighbors

    ### END MODIFIED ###

    def decide_community_actions(self, params, all_states_objects_dict, community_projects):
        k_comm = params.get('coefficients', {}).get('community_actions', {})
        if self.y2_drive > k_comm.get('initiate_project_y2_thresh', 7.0) and self.s2_reputation > k_comm.get(
                'initiate_project_s2_thresh', 5.0) and random.random() < k_comm.get('initiate_project_prob', 0.01):
            project_id = str(uuid.uuid4())[:8];
            proj_name = f"公共设施改善-{project_id}"
            req_b1 = self.b1_resource * k_comm.get('project_b1_cost_ratio_self', 0.3);
            req_b1_total = req_b1 * 3
            new_project = CommunityProject(project_id, proj_name, req_b1_total,
                                           required_h2=k_comm.get('project_req_h2', 4.0),
                                           duration=k_comm.get('project_duration', 10), creator_en=self.name_en,
                                           target_participants=k_comm.get('project_target_participants', 3),
                                           reward_type='b2_reduction',
                                           reward_value=k_comm.get('project_b2_reward_val', 0.8))
            if self.b1_resource >= req_b1:
                self.b1_resource -= req_b1;
                new_project.contributed_b1 += req_b1;
                new_project.participants = {self.name_en}
                community_projects[project_id] = new_project;
                self.active_effects_log.append(f"发起社区项目'{proj_name}', 贡献B1: {req_b1:.2f}")
                log_message("INFO", f"{self.name_zh} 发起社区项目'{proj_name}'", "CommunityAction")
            else:
                self.active_effects_log.append(f"想发起项目'{proj_name}'但B1不足")
        for proj_id, project in community_projects.items():
            if project.status == "pending" and self.name_en not in project.participants and len(
                    project.participants) < project.target_participants:
                trust_in_creator = self.trust_levels.get(project.creator_en, 3.0)
                perceived_value_vs_cost_ratio = (project.reward_value / (
                        project.required_b1_total / project.target_participants + 0.1)) * scale_value(
                    self.y1_clarity, 0, 10, 0.5, 1.2)
                can_afford_contribution = self.b1_resource > (
                        project.required_b1_total / project.target_participants) * 1.2;
                join_propensity = 0
                if trust_in_creator > k_comm.get('join_project_trust_thresh', 5.0): join_propensity += k_comm.get(
                    'join_trust_factor', 0.3)
                if perceived_value_vs_cost_ratio > k_comm.get('join_project_value_ratio_thresh',
                                                              1.5): join_propensity += k_comm.get('join_value_factor',
                                                                                                  0.4)
                if can_afford_contribution: join_propensity += k_comm.get('join_afford_factor', 0.2)
                if random.random() < join_propensity * k_comm.get('join_project_base_prob', 0.1):
                    my_contribution = project.required_b1_total / project.target_participants
                    if self.b1_resource >= my_contribution:
                        self.b1_resource -= my_contribution
                        if project.add_participant(self.name_en, my_contribution):
                            self.contributed_projects[proj_id] = my_contribution
                        else:
                            self.b1_resource += my_contribution
                    break

    def evolve(self, params, all_states_objects_dict, active_event_effects_on_self, global_env_factors, current_step,
               community_projects):
        self.update_perception(params)
        coeffs = params.get('coefficients', {});
        noise_level = params.get('noise_level', 0.05);
        lr = params.get('learning_rate', 0.1)
        b1_acq_mod = global_env_factors.get('b1_acq_mod', 1.0);
        b2_base_inc = global_env_factors.get('b2_base_inc', 0.0)  # Ensure 'b2_base_inc' is correct key
        current_risk_outcome_this_step = 0.0;
        risk_project_b1_change = 0.0;
        risk_project_b2_change = 0.0
        self.manage_social_relations(params, all_states_objects_dict, current_step)
        self.decide_community_actions(params, all_states_objects_dict, community_projects)
        h3_risk_attempt_thresh = params.get('h3_risk_attempt_threshold', 4.0);
        y2_risk_attempt_thresh = params.get('y2_risk_attempt_threshold', 3.0)
        perceived_can_invest = self.perceived_b1_resource > params.get('risk_min_perceived_b1_to_attempt', 1.0)
        risk_attempt_probability = sigmoid(self.h3_risk_appetite - h3_risk_attempt_thresh, k=0.8, x0=0) * sigmoid(
            self.y2_drive - y2_risk_attempt_thresh, k=0.7, x0=0) * scale_value(self.s2_reputation, 0, 10, 1.1, 0.8) * (
                                           0.5 + 0.5 * scale_value(self.mood, -1, 1, 0.8, 1.2))
        if random.random() < risk_attempt_probability and perceived_can_invest:
            b1_invest_ratio_base = scale_value(self.h3_risk_appetite, 0, 10, params.get('risk_invest_ratio_min', 0.03),
                                               params.get('risk_invest_ratio_max', 0.12))
            b1_invest_ratio_modifier = (
                    0.6 + 0.2 * sigmoid(self.y1_clarity, k=1, x0=5) + 0.2 * sigmoid(self.s1_trustworthiness, k=1,
                                                                                    x0=6))
            potential_investment_amount = self.perceived_b1_resource * np.clip(
                b1_invest_ratio_base * b1_invest_ratio_modifier, 0, params.get('risk_max_invest_cap_ratio', 0.30))
            b1_invested = min(potential_investment_amount, self.b1_resource)
            if b1_invested > 0.01:
                k_risk = coeffs.get('risk_project', {});
                potential_R_factor = k_risk.get('potential_R_base', 0) + k_risk.get('potential_R_h2',
                                                                                    0) * self.h2_innovation + k_risk.get(
                    'potential_R_y1', 0) * self.y1_clarity + k_risk.get('potential_R_s2', 0) * self.s2_reputation
                inherent_risk_L_factor = k_risk.get('inherent_L_base', 0) + k_risk.get('inherent_L_h3',
                                                                                       0) * self.h3_risk_appetite + k_risk.get(
                    'inherent_L_b2', 0) * self.b2_limitation - k_risk.get('reduction_s1', 0) * self.s1_trustworthiness
                mean_outcome = potential_R_factor - inherent_risk_L_factor * k_risk.get('L_to_mean_factor', 0.6);
                std_dev_outcome = max(k_risk.get('min_std_dev', 0.01),
                                      inherent_risk_L_factor * k_risk.get('L_to_std_factor', 0.7))
                actual_return_factor_on_investment = random.normalvariate(mean_outcome, std_dev_outcome);
                current_risk_outcome_this_step = actual_return_factor_on_investment
                b1_investment_scale_factor = sigmoid(
                    self.b1_resource * params.get('risk_invest_diminishing_threshold_factor', 0.1) - b1_invested,
                    k=-params.get('risk_invest_diminishing_k', 0.1), x0=0)
                effective_return_factor = actual_return_factor_on_investment * (0.5 + 0.5 * b1_investment_scale_factor);
                risk_project_b1_change = b1_invested * effective_return_factor * b1_acq_mod
                if current_risk_outcome_this_step < params.get('risk_failure_threshold_for_b2',
                                                               -0.1): risk_project_b2_change = k_risk.get(
                    'b2_from_major_failure', 0) * abs(current_risk_outcome_this_step) * (b1_invested / max(0.1,
                                                                                                           self.b1_resource if self.b1_resource > 0 else 0.1))
                self.active_effects_log.append(
                    f"风险项目: 计划投(基于感知) {potential_investment_amount:.2f}, 实际投 {b1_invested:.2f}, 回报因子 {current_risk_outcome_this_step:.2f} (有效 {effective_return_factor:.2f}), B1变 {risk_project_b1_change:.2f}, B2变 {risk_project_b2_change:.2f}")
            else:
                self.active_effects_log.append(f"风险项目: 感知B1不足或实际B1不足以支持投资。")
        else:
            if not perceived_can_invest: self.active_effects_log.append(
                f"风险项目: 因感知B1({self.perceived_b1_resource:.2f})不足而未尝试。")
        self.update_mood(params, current_risk_outcome_this_step)
        avg_b1_others, avg_h1_others = None, None
        all_b1s = [obj.b1_resource for name_other, obj in all_states_objects_dict.items() if
                   name_other != self.name_en and obj is not None]
        if all_b1s: avg_b1_others = np.mean(all_b1s);
        all_h1s = [obj.h1_possibilities for name_other, obj in all_states_objects_dict.items() if
                   name_other != self.name_en and obj is not None]
        if all_h1s: avg_h1_others = np.mean(all_h1s)
        neighbor_effects = self._calculate_neighbor_effects(params, all_states_objects_dict)
        deltas = {'b1_resource': self._calculate_delta_b1(coeffs.get('b1_resource', {}), params, avg_b1_others,
                                                          neighbor_effects.get('b1_resource', 0),
                                                          risk_project_b1_change),
                  'b2_limitation': self._calculate_delta_b2(coeffs.get('b2_limitation', {}), params,
                                                            neighbor_effects.get('b2_limitation', 0),
                                                            risk_project_b2_change) + b2_base_inc,  # Use b2_base_inc
                  'y1_clarity': self._calculate_delta_y1(coeffs.get('y1_clarity', {}), params,
                                                         neighbor_effects.get('y1_clarity', 0),
                                                         current_risk_outcome_this_step),
                  'y2_drive': self._calculate_delta_y2(coeffs.get('y2_drive', {}), params,
                                                       neighbor_effects.get('y2_drive', 0),
                                                       current_risk_outcome_this_step),
                  'y3_aspiration': self._calculate_delta_y3(coeffs.get('y3_aspiration', {}), params,
                                                            neighbor_effects.get('y3_aspiration', 0), avg_b1_others,
                                                            avg_h1_others),
                  'h1_possibilities': self._calculate_delta_h1(coeffs.get('h1_possibilities', {}), params,
                                                               neighbor_effects.get('h1_possibilities', 0)),
                  'h2_innovation': self._calculate_delta_h2(coeffs.get('h2_innovation', {}), params,
                                                            neighbor_effects.get('h2_innovation', 0)),
                  'h3_risk_appetite': self._calculate_delta_h3(coeffs.get('h3_risk_appetite', {}), params,
                                                               neighbor_effects.get('h3_risk_appetite', 0),
                                                               current_risk_outcome_this_step),
                  's1_trustworthiness': self._calculate_delta_s1_trustworthiness(coeffs.get('s1_trustworthiness', {}),
                                                                                 params, neighbor_effects.get(
                          's1_trustworthiness', 0), current_risk_outcome_this_step),
                  's2_reputation': self._calculate_delta_s2_reputation(coeffs.get('s2_reputation', {}), params,
                                                                       neighbor_effects.get('s2_reputation', 0),
                                                                       current_risk_outcome_this_step), }
        internal_logs_this_step = self.active_effects_log[:]
        self.active_effects_log.clear()
        for key in DIM_KEYS:
            current_val = getattr(self, key);
            total_delta_from_sources = deltas.get(key, 0);
            noise_val = random.uniform(-noise_level, noise_level)
            effective_delta = total_delta_from_sources
            if key == 'b1_resource' and effective_delta > 0: effective_delta *= b1_acq_mod
            effective_delta += noise_val;
            new_val_before_event = current_val + effective_delta * lr;
            final_val_after_event = new_val_before_event
            if key in active_event_effects_on_self:
                for event_eff in active_event_effects_on_self.get(key, []):
                    eff_val = event_eff.get('val', 0);
                    eff_type = event_eff.get('type')
                    resilience_score = (self.y1_clarity / 10) * params.get('event_resilience_y1_factor', 0.6) + (
                            self.b1_resource / 10) * params.get('event_resilience_b1_factor', 0.4)
                    resilience_score = np.clip(resilience_score, 0.1, 1.0);
                    actual_eff_val = eff_val
                    if eff_val < 0:
                        actual_eff_val = eff_val / (
                                resilience_score * params.get('event_neg_resilience_mult', 1.5) + 0.5)
                    else:
                        actual_eff_val = eff_val * (
                                resilience_score * params.get('event_pos_resilience_mult', 0.8) + 0.6)
                    if eff_type == 'add_abs':
                        final_val_after_event += actual_eff_val
                    elif eff_type == 'set_abs':
                        final_val_after_event = actual_eff_val
                    elif eff_type == 'multiply_abs':
                        if actual_eff_val < 1.0 and eff_val < 1.0:
                            actual_eff_val = 1.0 - (1.0 - actual_eff_val) / (resilience_score + 0.1)
                        elif actual_eff_val > 1.0 and eff_val > 1.0:
                            actual_eff_val = 1.0 + (actual_eff_val - 1.0) * (resilience_score + 0.1)
                        final_val_after_event *= actual_eff_val
            final_val_after_event = np.clip(final_val_after_event, 0, 10)
            if key == 'b1_resource' and final_val_after_event > params.get('b1_practical_max', 9.7):
                overflow = final_val_after_event - params.get('b1_practical_max', 9.7);
                final_val_after_event = params.get('b1_practical_max', 9.7)
                s2_coeffs_for_overflow = coeffs.get('s2_reputation', {})
                self.s2_reputation += overflow * s2_coeffs_for_overflow.get('from_b1_overflow', 0.05);
                self.s2_reputation = np.clip(self.s2_reputation, 0, 10)
                if overflow > 1e-3: self.active_effects_log.append(f"B1溢出 {overflow:.2f} -> S2增加")
            setattr(self, key, final_val_after_event)
        self.last_risk_outcome_factor = current_risk_outcome_this_step
        self.active_effects_log.extend(internal_logs_this_step)

    def record_history(self, coord_type, wb, wy, wh, max_history=50):
        coords = self.get_coords_for_plot(coord_type, wb, wy, wh);
        if len(self.history) >= max_history: self.history.pop(0)
        self.history.append(coords)

    def clear_history(self):
        self.history = []

    def to_dict(self):
        data = {key: getattr(self, key) for key in DIM_KEYS}
        data.update(
            {'name_zh': self.name_zh, 'name_en': self.name_en, 'history': self.history, 'neighbors': self.neighbors,
             'trust_levels': dict(self.trust_levels), 'last_risk_outcome_factor': self.last_risk_outcome_factor,
             'alliance_partners': list(self.alliance_partners), 'rivals': list(self.rivals),
             'perceived_b1_resource': self.perceived_b1_resource,
             'perceived_h1_possibilities': self.perceived_h1_possibilities,
             'perception_accuracy': self.perception_accuracy, 'mood': self.mood,
             'community_norm_conformity': self.community_norm_conformity,
             'contributed_projects': self.contributed_projects, })
        return data

    @classmethod
    def from_dict(cls, data):
        dim_data = {key: data.get(key, 0 if 'limitation' not in key else 5) for key in DIM_KEYS}
        obj = cls(data.get('name_zh', '未知状态'), data.get('name_en', f'Unknown_{random.randint(1000, 9999)}'),
                  dim_data['b1_resource'], dim_data['b2_limitation'], dim_data['y1_clarity'], dim_data['y2_drive'],
                  dim_data['y3_aspiration'], dim_data['h1_possibilities'], dim_data['h2_innovation'],
                  dim_data['h3_risk_appetite'],
                  dim_data.get('s1_trustworthiness', 5.0), dim_data.get('s2_reputation', 3.0))
        obj.history = data.get('history', []);
        obj.neighbors = data.get('neighbors', [])
        obj.trust_levels = defaultdict(lambda: 5.0, data.get('trust_levels', {}))
        obj.last_risk_outcome_factor = data.get('last_risk_outcome_factor', 0)
        obj.alliance_partners = set(data.get('alliance_partners', []));
        obj.rivals = set(data.get('rivals', []))
        obj.perceived_b1_resource = data.get('perceived_b1_resource', obj.b1_resource)
        obj.perceived_h1_possibilities = data.get('perceived_h1_possibilities', obj.h1_possibilities)
        obj.perception_accuracy = data.get('perception_accuracy', scale_value(obj.y1_clarity, 0, 10, 0.5, 1.0))
        obj.mood = data.get('mood', 0)
        obj.community_norm_conformity = data.get('community_norm_conformity', random.uniform(0.3, 0.7))
        obj.contributed_projects = data.get('contributed_projects', {})
        return obj

    def __repr__(self):
        return f"<WorldState: {self.get_display_name()}>"


# --- EventManager and Event classes (GM4.5.6) ---
class Event:
    def __init__(self, name, trigger_type, trigger_params, target_selector, effects, duration=1, one_time=False,
                 chain_event_name=None, chain_event_delay=0, chain_event_prob=0):
        self.name = name;
        self.trigger_type = trigger_type;
        self.trigger_params = trigger_params
        self.target_selector = target_selector;
        self.effects = effects;
        self.duration = duration
        self.one_time = one_time;
        self.triggered_this_step = False
        self.chain_event_name = chain_event_name;
        self.chain_event_delay = chain_event_delay;
        self.chain_event_prob = chain_event_prob

    def check_trigger(self, all_states_objects_dict, global_metrics, global_env_factors, current_step=0):
        # --- Probabilistic Trigger ---
        if self.trigger_type == 'probabilistic':
            base_prob = self.trigger_params.get('prob', 0.01)
            env_prob_mod_key = self.trigger_params.get('env_prob_mod_key')
            env_prob_modifier = 1.0
            if env_prob_mod_key and global_env_factors:
                env_prob_modifier = global_env_factors.get(env_prob_mod_key, 1.0)
            return random.random() < (base_prob * env_prob_modifier)

        # --- Conditional Global Trigger ---
        elif self.trigger_type == 'conditional_global':
            metric_name = self.trigger_params.get('dim')
            source_type = self.trigger_params.get('source', 'metrics')  # 'metrics', 'env_factors', or 'metrics_derived'

            if source_type == 'metrics':
                source_dict = global_metrics
            elif source_type == 'env_factors':
                source_dict = global_env_factors
            elif source_type == 'metrics_derived':  # For metrics calculated on-the-fly like diversity
                source_dict = global_metrics  # Assume they are added to global_metrics
            else:
                log_message("WARN",
                            f"Event '{self.name}': Unknown source_type '{source_type}' for conditional_global trigger.",
                            "EventCheck")
                return False

            if not metric_name:
                log_message("WARN", f"Event '{self.name}': Missing 'dim' in trigger_params for conditional_global.",
                            "EventCheck")
                return False

            metric_val = source_dict.get(metric_name)
            if metric_val is None:
                # log_message("DEBUG", f"Event '{self.name}': Metric '{metric_name}' not found in source_dict for conditional_global.", "EventCheck")
                return False  # Metric not available yet or misspelled

            op = self.trigger_params.get('op')
            thresh_val = self.trigger_params.get('val')

            if not op or thresh_val is None:
                log_message("WARN",
                            f"Event '{self.name}': Missing 'op' or 'val' in trigger_params for conditional_global.",
                            "EventCheck")
                return False

            if op == '<': return metric_val < thresh_val
            if op == '>': return metric_val > thresh_val
            if op == '==': return abs(metric_val - thresh_val) < 1e-6  # Float equality
            log_message("WARN", f"Event '{self.name}': Unknown operator '{op}' for conditional_global.", "EventCheck")
            return False

        # --- Timed Trigger ---
        elif self.trigger_type == 'timed':
            return current_step == self.trigger_params.get('step', -1)

        ### ADDED: Clearer implementation for 'conditional_individual' trigger type ###
        elif self.trigger_type == 'conditional_individual':
            dim_to_check = self.trigger_params.get('dim')
            op_to_check = self.trigger_params.get('op')
            val_to_check = self.trigger_params.get('val')

            if not all([dim_to_check, op_to_check, val_to_check is not None]):
                log_message("WARN",
                            f"Event '{self.name}': Missing 'dim', 'op', or 'val' in trigger_params for conditional_individual.",
                            "EventCheck")
                return False

            # Iterate through all valid WorldState objects
            for state_name, state_obj in all_states_objects_dict.items():
                if state_obj is None:  # Skip if object failed to load
                    continue

                if hasattr(state_obj, dim_to_check):
                    s_val = getattr(state_obj, dim_to_check)
                    condition_met = False
                    if s_val is not None:
                        if op_to_check == '>' and s_val > val_to_check:
                            condition_met = True
                        elif op_to_check == '<' and s_val < val_to_check:
                            condition_met = True
                        elif op_to_check == '==' and abs(s_val - val_to_check) < 1e-6:  # Float equality
                            condition_met = True
                        # Add other operators if needed (e.g., '>=', '<=')

                    if condition_met:
                        # log_message("DEBUG", f"Event '{self.name}' triggered by individual '{state_obj.name_en}' on dim '{dim_to_check}'.", "EventCheck")
                        return True  # Event triggers if *any one* individual meets the condition
                else:
                    # This might happen if a dim_to_check is, for example, a complex derived property not directly on state_obj
                    # log_message("DEBUG", f"Event '{self.name}': State '{state_obj.name_en}' does not have attribute '{dim_to_check}'.", "EventCheck")
                    pass

            return False  # No individual met the condition
        ### END ADDED ###

        # --- None Trigger (for chained events or externally triggered events) ---
        elif self.trigger_type == 'none':
            return False  # These are typically triggered by other mechanisms (e.g., chain_event in EventManager)

        else:
            log_message("WARN", f"Event '{self.name}': Unknown trigger_type '{self.trigger_type}'.", "EventCheck")
            return False

    # Inside class Event:
    def select_targets(self, all_states_objects_dict):
        targets = []
        pop_valid = [s for s in all_states_objects_dict.values() if s is not None]
        if not pop_valid:
            return []

        # --- 强制 target_selector 为字典格式，并获取 type ---
        if not isinstance(self.target_selector, dict) or 'type' not in self.target_selector:
            # 处理旧的 "all" 字符串格式作为特殊情况，或者将其也转为字典 {'type':'all'}
            if self.target_selector == "all":
                target_selector_dict = {'type': 'all'}
            else:
                log_message("ERROR",
                            f"Event '{self.name}': target_selector format is invalid. Expected a dict with a 'type' key, got: {self.target_selector}",
                            "EventSelect")
                return []  # 无效格式，直接返回空
        else:
            target_selector_dict = self.target_selector

        target_selector_type = target_selector_dict.get('type')
        # --- End of new parsing logic ---

        if target_selector_type == 'all':
            targets = pop_valid
        elif target_selector_type == 'random_n':
            n = target_selector_dict.get('n', 1)
            if pop_valid:
                targets = random.sample(pop_valid, min(n, len(pop_valid)))
            else:
                targets = []

        elif target_selector_type == 'conditional_individual':
            dim = target_selector_dict.get('dim')
            op = target_selector_dict.get('op')
            val = target_selector_dict.get('val')
            max_t = target_selector_dict.get('max_targets')

            if not dim or not op or val is None:
                log_message("WARN",
                            f"Event '{self.name}': Missing params in conditional_individual selector (dim, op, or val) for: {target_selector_dict}",
                            "EventSelect")
                return []  # 参数不全则返回空

            max_t = float(max_t) if max_t is not None else float('inf')
            eligible = []
            for s in pop_valid:
                if hasattr(s, dim):
                    s_val = getattr(s, dim, None)
                    if s_val is not None:
                        condition_met = False
                        if op == '>' and s_val > val:
                            condition_met = True
                        elif op == '<' and s_val < val:
                            condition_met = True
                        elif op == '==' and abs(s_val - val) < 1e-6:
                            condition_met = True
                        # Add other ops like '>=' or '<=' if needed
                        if condition_met:
                            eligible.append(s)
            if eligible:
                targets = random.sample(eligible, min(int(max_t), len(eligible)))
            else:
                targets = []

        elif target_selector_type == 'mood_based':
            op = target_selector_dict.get('op', '>')
            mood_val = target_selector_dict.get('val', 0.5)
            max_t = target_selector_dict.get('max_targets', float('inf'))
            eligible = []
            for s in pop_valid:  # Ensure s has 'mood' attribute
                if hasattr(s, 'mood'):
                    if (op == '>' and s.mood > mood_val) or \
                            (op == '<' and s.mood < mood_val):
                        eligible.append(s)
            if eligible:
                targets = random.sample(eligible, min(int(max_t), len(eligible)))
            else:
                targets = []

        elif target_selector_type == 'random_n_neighbors':
            if not pop_valid:  # Should have been caught at the start
                targets = []
            else:
                n_clusters = target_selector_dict.get('n_clusters', 1)
                cluster_size_avg = target_selector_dict.get('cluster_size_avg', 3)
                cluster_size_std = target_selector_dict.get('cluster_size_std', 1)
                selected_targets_set = set()

                temp_pop_valid_for_centers = list(pop_valid)  # Use a mutable copy for choosing centers

                for _ in range(n_clusters):
                    if not temp_pop_valid_for_centers: break
                    center_agent = random.choice(temp_pop_valid_for_centers)
                    temp_pop_valid_for_centers.remove(center_agent)  # Avoid picking same center twice

                    selected_targets_set.add(center_agent)

                    current_cluster_size = max(1, int(random.normalvariate(cluster_size_avg, cluster_size_std)))

                    # If cluster size is 1, only the center is targeted
                    if current_cluster_size == 1:
                        continue

                    potential_cluster_members = []
                    if hasattr(center_agent, 'neighbors'):
                        for neighbor_name in center_agent.neighbors:
                            neighbor_obj = all_states_objects_dict.get(neighbor_name)
                            if neighbor_obj and neighbor_obj not in selected_targets_set:
                                potential_cluster_members.append(neighbor_obj)

                    if potential_cluster_members:
                        num_to_add = min(len(potential_cluster_members), current_cluster_size - 1)
                        if num_to_add > 0:
                            selected_members = random.sample(potential_cluster_members, num_to_add)
                            for member in selected_members:
                                selected_targets_set.add(member)
                targets = list(selected_targets_set)
        else:
            log_message("ERROR",
                        f"Event '{self.name}': Unhandled target_selector type '{target_selector_type}'. Selector data: {target_selector_dict}",
                        "EventSelect")
            return []  # 如果类型未知，明确返回空列表

        return targets

    def get_effects_for_target(self):
        applied_effects = []
        for effect_template in self.effects:
            base_val = effect_template.get('val', 0);
            rand_range = effect_template.get('rand_range', 0);
            random_offset = 0
            if rand_range > 0 and base_val != 0:
                random_offset = random.uniform(-rand_range * abs(base_val), rand_range * abs(base_val))
            elif rand_range > 0 and base_val == 0:
                random_offset = random.uniform(-rand_range, rand_range)
            final_val = base_val + random_offset
            applied_effects.append(
                {'name': self.name, 'dim': effect_template.get('dim'), 'type': effect_template.get('type'),
                 'duration': self.duration, 'val': final_val})
        return applied_effects


class EventManager:
    def __init__(self, event_definitions_template_arg):
        self.event_definitions_template = event_definitions_template_arg
        self.events = [Event(**ed) for ed in self.event_definitions_template]
        self.active_timed_effects = defaultdict(list)
        self.pending_chained_events = []

    def reset_events(self):
        self.events = [Event(**ed) for ed in self.event_definitions_template];
        self.active_timed_effects.clear();
        self.pending_chained_events.clear()
        log_message("INFO", "EventManager reset.", "EventManager")

    def process_step(self, all_states_objects_dict, global_metrics, global_env_factors, current_step):
        effects_to_apply_this_step = defaultdict(lambda: defaultdict(list));
        log_messages_this_step = [];
        events_to_remove_indices = []
        for i, event_obj in enumerate(self.events):
            event_obj.triggered_this_step = False
            try:
                if event_obj.check_trigger(all_states_objects_dict, global_metrics, global_env_factors, current_step):
                    event_obj.triggered_this_step = True;
                    log_messages_this_step.append(f"事件 '{event_obj.name}' 已触发.")
                    targets = event_obj.select_targets(all_states_objects_dict)
                    for target_state in targets:
                        if not target_state or not hasattr(target_state, 'name_en'): continue
                        for effect_data in event_obj.get_effects_for_target():
                            effect_dim = effect_data.get('dim');
                            if not effect_dim: continue
                            effects_to_apply_this_step[target_state.name_en][effect_dim].append(effect_data)
                            log_messages_this_step.append(
                                f" -> 事件 '{event_obj.name}' 目标 {target_state.name_zh}, 维度 {effect_dim}, 值 {effect_data['val']:.2f}, 类型 {effect_data['type']}")
                            if effect_data['duration'] > 1: self.active_timed_effects[target_state.name_en].append(
                                {'effect_data': effect_data, 'remaining_duration': effect_data['duration'] - 1})
                        if event_obj.chain_event_name and random.random() < event_obj.chain_event_prob:
                            chain_target_step = current_step + event_obj.chain_event_delay + 1
                            self.pending_chained_events.append(
                                (chain_target_step, event_obj.chain_event_name, target_state.name_en))
                            log_messages_this_step.append(
                                f" -> 事件 '{event_obj.name}' 将链接事件 '{event_obj.chain_event_name}' 在步骤 {chain_target_step} 针对 {target_state.name_zh}")
                    if event_obj.one_time and event_obj.triggered_this_step: events_to_remove_indices.append(i)
            except Exception as e:
                log_message("ERROR", f"Error processing event '{event_obj.name}': {e}\n{traceback.format_exc()}",
                            "EventManager")
        for index_to_remove in sorted(events_to_remove_indices, reverse=True): del self.events[index_to_remove]
        remaining_pending_chains = []
        for chain_step, chain_name, chain_target_name in self.pending_chained_events:
            if current_step == chain_step:
                found_event_def = next((ed for ed in self.event_definitions_template if ed['name'] == chain_name), None)
                if found_event_def:
                    chained_event_obj = Event(**found_event_def);
                    target_obj = all_states_objects_dict.get(chain_target_name)
                    targets = [target_obj] if target_obj else chained_event_obj.select_targets(all_states_objects_dict)
                    log_messages_this_step.append(f"链接事件 '{chain_name}' 已激活 (原目标: {chain_target_name}).")
                    for target_state in targets:
                        if not target_state or not hasattr(target_state, 'name_en'): continue
                        for effect_data in chained_event_obj.get_effects_for_target():
                            effect_dim = effect_data.get('dim')
                            if not effect_dim: continue
                            effects_to_apply_this_step[target_state.name_en][effect_dim].append(effect_data)
                else:
                    log_messages_this_step.append(f"警告: 未找到链接事件定义 '{chain_name}'")
            elif chain_step > current_step:
                remaining_pending_chains.append((chain_step, chain_name, chain_target_name))
        self.pending_chained_events = remaining_pending_chains
        new_active_timed_effects = defaultdict(list)
        for state_name, timed_effects_list in self.active_timed_effects.items():
            for timed_effect in timed_effects_list:
                effect_data = timed_effect['effect_data'];
                effect_dim = effect_data.get('dim')
                if not effect_dim: continue
                effects_to_apply_this_step[state_name][effect_dim].append(effect_data)
                log_messages_this_step.append(
                    f" -> 定时效果 '{effect_data.get('name', '未命名')}' 目标 {state_name}, 维度 {effect_dim}, 值 {effect_data['val']:.2f} (剩余 {timed_effect['remaining_duration']})")
                timed_effect['remaining_duration'] -= 1
                if timed_effect['remaining_duration'] > 0:
                    new_active_timed_effects[state_name].append(timed_effect)
                else:
                    log_messages_this_step.append(
                        f"定时效果 '{effect_data.get('name', '未命名')}' 在 {state_name} 上已到期.")
        self.active_timed_effects = new_active_timed_effects
        return effects_to_apply_this_step, log_messages_this_step


# --- 4. Initialization (GM4.5.6) ---
initial_states_templates_gm456 = [
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
    ### ADDED: 新增特色角色 ###
    {'name_zh': "远见创新者", 'name_en': "VisionaryInnovator",
     'b1_res': 3.0, 'b2_lim': 4.0, 'y1_cla': 7.5, 'y2_dri': 8.5, 'y3_asp': 9.5,
     'h1_pos': 7.0, 'h2_inn': 8.5, 'h3_ris': 7.0, 's1_tru': 6.5, 's2_rep': 4.0},

    {'name_zh': "迷茫探索者", 'name_en': "WanderingExplorer",
     'b1_res': 4.5, 'b2_lim': 4.0, 'y1_cla': 3.0, 'y2_dri': 5.5, 'y3_asp': 3.5,
     'h1_pos': 6.0, 'h2_inn': 4.0, 'h3_ris': 6.0, 's1_tru': 5.0, 's2_rep': 4.5},

    {'name_zh': "务实建设者", 'name_en': "PragmaticBuilder",
     'b1_res': 6.5, 'b2_lim': 2.5, 'y1_cla': 8.0, 'y2_dri': 7.0, 'y3_asp': 6.0,
     'h1_pos': 5.5, 'h2_inn': 6.5, 'h3_ris': 4.0, 's1_tru': 7.5, 's2_rep': 6.5},

    {'name_zh': "愤世批评家", 'name_en': "CynicalCritic",
     'b1_res': 3.5, 'b2_lim': 6.0, 'y1_cla': 7.0, 'y2_dri': 5.0, 'y3_asp': 7.5,
     'h1_pos': 4.0, 'h2_inn': 5.5, 'h3_ris': 3.0, 's1_tru': 4.0, 's2_rep': 5.0},
    ### END ADDED ###
    ### ADDED: 进一步新增的角色 ###
    {'name_zh': "和平缔造者", 'name_en': "Peacemaker",
     'b1_res': 5.0, 'b2_lim': 3.0, 'y1_cla': 7.5, 'y2_dri': 6.5, 'y3_asp': 7.0,
     'h1_pos': 5.0, 'h2_inn': 4.0, 'h3_ris': 2.5, 's1_tru': 8.5, 's2_rep': 7.0},

    {'name_zh': "资源掌控者", 'name_en': "ResourceBaron",
     'b1_res': 8.5, 'b2_lim': 5.0, 'y1_cla': 7.0, 'y2_dri': 9.0, 'y3_asp': 7.5,
     'h1_pos': 5.0, 'h2_inn': 6.5, 'h3_ris': 7.5, 's1_tru': 3.0, 's2_rep': 6.0},

    {'name_zh': "传统守护者", 'name_en': "GuardianPreserver",  # 区别于保守长者，更强调“守护”特定事物
     'b1_res': 6.0, 'b2_lim': 2.0, 'y1_cla': 8.5, 'y2_dri': 5.5, 'y3_asp': 5.0,
     'h1_pos': 2.5, 'h2_inn': 1.5, 'h3_ris': 1.0, 's1_tru': 8.0, 's2_rep': 7.0},

    {'name_zh': "边缘艺术家", 'name_en': "MarginalArtist",
     'b1_res': 2.5, 'b2_lim': 5.5, 'y1_cla': 8.0, 'y2_dri': 6.0, 'y3_asp': 8.5,
     'h1_pos': 8.0, 'h2_inn': 7.0, 'h3_ris': 5.0, 's1_tru': 4.5, 's2_rep': 3.0},
    ### END ADDED ###
]

neighbor_config_gm456 = {
    # 原有角色
    "Loner": ["Opportunist", "WanderingExplorer", "CynicalCritic", "MarginalArtist"],
    "CollaborativeLeader": ["Opportunist", "DiligentArtisan", "ConservativeElder", "VisionaryInnovator",
                            "PragmaticBuilder", "Peacemaker"],
    "Opportunist": ["CollaborativeLeader", "Loner", "VisionaryInnovator", "WanderingExplorer", "CynicalCritic",
                    "ResourceBaron", "MarginalArtist"],
    "DiligentArtisan": ["CollaborativeLeader", "PragmaticBuilder", "ConservativeElder", "GuardianPreserver"],
    "ConservativeElder": ["CollaborativeLeader", "DiligentArtisan", "PragmaticBuilder", "GuardianPreserver",
                          "Peacemaker"],

    # 第一批新增角色
    "VisionaryInnovator": ["CollaborativeLeader", "Opportunist", "WanderingExplorer", "PragmaticBuilder",
                           "MarginalArtist"],
    "WanderingExplorer": ["Loner", "Opportunist", "VisionaryInnovator", "CynicalCritic", "MarginalArtist"],
    "PragmaticBuilder": ["CollaborativeLeader", "DiligentArtisan", "ConservativeElder", "VisionaryInnovator",
                         "Peacemaker", "ResourceBaron"],
    "CynicalCritic": ["Loner", "Opportunist", "WanderingExplorer", "MarginalArtist"],

    # 第二批新增角色
    "Peacemaker": ["CollaborativeLeader", "ConservativeElder", "PragmaticBuilder", "GuardianPreserver",
                   "WanderingExplorer"],  # 调解者需要广泛联系
    "ResourceBaron": ["Opportunist", "PragmaticBuilder", "CollaborativeLeader"],  # 资源掌控者与能带来利益或需要其资源的角色
    "GuardianPreserver": ["ConservativeElder", "DiligentArtisan", "Peacemaker"],  # 守护者与传统、稳定角色
    "MarginalArtist": ["Loner", "VisionaryInnovator", "WanderingExplorer", "CynicalCritic"]  # 边缘艺术家的小圈子
}

### MODIFIED: Enriched event_definitions_gm456 ###
event_definitions_gm456 = [
    # --- 自然灾害/环境变化事件 ---
    {'name': "区域性旱灾", 'trigger_type': "probabilistic",
     'trigger_params': {'prob': 0.01, 'env_prob_mod_key': 'environmental_stability_mod'},  # 假设有此环境因子
     'target_selector': {'type': 'random_n_neighbors', 'n_clusters': 1, 'cluster_size_avg': 3, 'cluster_size_std': 1},
     # 选择一个中心点及其N个邻居
     'effects': [{'dim': 'b1_resource', 'type': 'multiply_abs', 'val': 0.55, 'rand_range': 0.1},
                 {'dim': 'b2_limitation', 'type': 'add_abs', 'val': 1.8, 'rand_range': 0.2},
                 {'dim': 'mood', 'type': 'add_abs', 'val': -0.45, 'rand_range': 0.1}],
     'duration': 4,
     'chain_event_name': "灾后重建需求", 'chain_event_delay': 2, 'chain_event_prob': 0.75},

    {'name': "灾后重建需求", 'trigger_type': "none",  # 由旱灾链接
     'trigger_params': {},
     'target_selector': "all",  # 全局可感知，但主要由受灾者响应
     'effects': [{'dim': 'y3_aspiration', 'type': 'add_abs', 'val': 0.2, 'rand_range': 0.1}],  # 轻微提升理想（重建家园）
     'one_time': True  # 作为链式事件，只触发一次
     },  # Note: WorldState.decide_community_actions 应该检查此事件来提高发起重建项目的概率

    # --- 文化/意识形态冲突事件 ---
    {'name': "主流价值观挑战", 'trigger_type': "conditional_global",  # 例如，当Y1方差过大时 (高级) 或概率
     'trigger_params': {'prob': 0.008, 'dim': 'avg_y1_diversity', 'op': '>', 'val': 3.0, 'source': 'metrics_derived'},
     # 假设有此派生指标
     'target_selector': "all",
     'effects': [{'dim': 'y1_clarity', 'type': 'add_abs', 'val': -0.3, 'rand_range': 0.3},  # 普遍的价值困惑
                 {'dim': 's1_trustworthiness', 'type': 'multiply_abs', 'val': 0.9, 'rand_range': 0.05},  # 信任度下降
                 {'dim': 'mood', 'type': 'add_abs', 'val': -0.15, 'rand_range': 0.1}],
     'duration': 5, 'one_time': True},

    # --- 重大发现/启示事件 ---
    {'name': "古老智慧重现", 'trigger_type': "probabilistic", 'trigger_params': {'prob': 0.002},  # 非常罕见
     'target_selector': {'type': 'conditional_individual', 'dim': 'y1_clarity', 'op': '>', 'val': 8.0,
                         'max_targets': 1},  # 影响价值清晰度高的人
     'effects': [{'dim': 'y1_clarity', 'type': 'set_abs', 'val': 9.8},
                 {'dim': 'y3_aspiration', 'type': 'add_abs', 'val': 1.0, 'rand_range': 0.2},
                 {'dim': 'h1_possibilities', 'type': 'add_abs', 'val': 2.0, 'rand_range': 0.1},
                 {'dim': 's2_reputation', 'type': 'add_abs', 'val': 1.5, 'rand_range': 0.3}],
     'one_time': True},

    # --- 制度/规则变革事件 ---
    {'name': "创新激励政策", 'trigger_type': "conditional_global",
     'trigger_params': {'dim': 'avg_h2_innovation', 'op': '<', 'val': 3.0, 'source': 'metrics'},  # 全局创新不足时
     'target_selector': "all",
     'effects': [{'dim': 'h2_innovation', 'type': 'add_abs', 'val': 0.2, 'rand_range': 0.1},  # 轻微提升所有人的H2
                 # 可以在WorldState.evolve中让global_env_factors['h2_dev_cost_modifier']暂时降低
                 ],
     'duration': 6, 'one_time': True, 'chain_event_name': "创新泡沫风险", 'chain_event_delay': 4,
     'chain_event_prob': 0.3},

    {'name': "创新泡沫风险", 'trigger_type': "none", 'trigger_params': {}, 'target_selector': "all",
     'effects': [{'dim': 'h3_risk_appetite', 'type': 'add_abs', 'val': 0.5, 'rand_range': 0.2},  # 风险偏好不理性增加
                 {'dim': 'b2_limitation', 'type': 'add_abs', 'val': 0.3, 'rand_range': 0.1}],  # 潜在的系统性风险
     'duration': 3},
]

default_evolution_params_gm455_complete = {
    'learning_rate': 0.05, 'noise_level': 0.015,
    'h3_risk_attempt_threshold': 4.0, 'y2_risk_attempt_threshold': 3.0, 'risk_invest_ratio_min': 0.025,
    'risk_invest_ratio_max': 0.13, 'risk_max_invest_cap_ratio': 0.30,
    'risk_success_return_threshold_for_y1': 0.035, 'risk_failure_loss_threshold_for_y1': 0.035,
    'risk_success_return_threshold_for_y2': 0.025, 'risk_failure_loss_threshold_for_y2': 0.025,
    'risk_success_return_threshold_for_h3': 0.035, 'risk_failure_loss_threshold_for_h3': 0.035,
    'risk_failure_threshold_for_b2': -0.08,
    'y2_b1_sustain_threshold': 2.2, 'y1_gap_threshold_for_loss': 3.0, 'y3_reality_gap_threshold': 4.0,
    'y3_social_b1_offset': 1.0, 'y3_social_h1_offset': 0.6,
    's1_risk_failure_penalty_thresh': -0.2, 's2_risk_success_bonus_thresh': 0.15, 'y_social_align_self_thresh': 7.5,
    'y3_social_align_self_thresh': 8.0,
    'b1_maintenance_base': 0.03, 'b1_maintenance_factor': 0.005, 'b1_maintenance_exponent': 2.0,
    'b1_practical_max': 9.7,
    'risk_invest_diminishing_threshold_factor': 0.05, 'risk_invest_diminishing_k': 0.2,
    'event_resilience_y1_factor': 0.6, 'event_resilience_b1_factor': 0.4, 'event_neg_resilience_mult': 1.5,
    'event_pos_resilience_mult': 0.8,
    'max_history_points': 50, 'b1_maintenance_scale_ref': 10.0, 'y2_saturation_k': 1.0, 'y2_saturation_x0': 8.0,
    'y2_burnout_thresh': 7.5,
    'y1_high_maintenance_thresh': 8.0, 'y3_aspiration_h1_influence_damp_k': 1.5,
    'y3_aspiration_h1_influence_damp_x0': 8.5,
    'y3_complacency_thresh': 8.5, 'y3_complacency_b1_gap': 4.0, 'y3_complacency_h1_gap': 4.0,
    'h1_b1_influence_damp_k': 1.0, 'h1_b1_influence_damp_x0': 8.0,
    'h1_focus_cost_thresh': 8.5, 'h2_innovation_saturation_k': 1.0, 'h2_innovation_saturation_x0': 8.5,
    'h2_decay_scale_ref': 9.0, 'h2_complexity_thresh': 8.0,
    'h3_risk_feedback_damp_k': 0.3, 'h3_risk_feedback_damp_x0': 4.0, 'h3_b1_stability_thresh': 8.0,
    'h3_b2_stability_thresh': 2.0, 'h3_stable_target_risk': 3.0,
    'h3_b1_desperation_thresh': 2.0, 'h3_y2_desperation_thresh': 6.0, 'h3_desperate_target_risk': 7.0,
    's_decay_scale_ref': 10.0,
    'b2_y2_overextension_thresh': 9.0, 'b2_h2_overextension_thresh': 9.0,
    'social_action_interval': 5, 'alliance_cooldown': 20, 'rivalry_cooldown': 25,
    'h2_to_b1_direct_thresh': 7.0, 'y2_for_h2_to_b1_direct_thresh': 6.0, 'h2_to_b2_reduction_thresh': 7.5,
    'y1_for_h2_to_b2_reduction_thresh': 6.5,
    'b1_for_y1_erosion_thresh': 2.0, 'b2_for_y1_erosion_thresh': 7.0, 'b1_for_y1_affirm_thresh': 7.0,
    'b2_for_y1_affirm_thresh': 3.0, 'y2_for_y1_affirm_thresh': 5.0,
    'b1_for_y2_sap_thresh': 1.5, 'b2_for_y2_sap_thresh': 7.5, 'b1_for_y2_boost_thresh': 6.5,
    'y1_for_y2_boost_thresh': 6.0,
    'b1_for_y3_lift_thresh': 7.5, 'b2_for_y3_lift_thresh': 2.5, 'y2_for_y3_lift_thresh': 6.5,
    'y3_target_after_b_success': 9.0,
    'b2_h1_suppression_thresh': 6.0, 'y3_for_h2_drive_thresh': 7.0, 'y1_for_h2_drive_thresh': 6.0,
    'b1_for_h2_drive_thresh': 3.0,
    'coefficients': {
        'b1_resource': {'from_h2': 0.08, 'from_y2': 0.07, 'loss_b2': 0.125, 'cost_h2_activity': 0.012,
                        'cost_y2_sustain': 0.007, 'social_pressure': 0.0015, 'from_s2_reputation': 0.003,
                        'h2_to_b1_direct_prob': 0.01, 'h2_to_b1_direct_factor': 0.05, },
        'b2_limitation': {'reduce_y2': 0.05, 'reduce_h2': 0.04, 'random_factor': 0.08, 'from_y2_overextension': 0.015,
                          'from_h2_overextension': 0.01, 'h2_to_b2_reduction_prob': 0.008,
                          'h2_to_b2_reduction_factor': 0.06, },
        'y1_clarity': {'from_success_validation': 0.18, 'from_failure_doubt': -0.20, 'loss_b2': 0.115,
                       'loss_aspiration_gap': 0.02, 'high_y1_decay_factor': 0.005, 'b_state_y1_erosion_factor': 0.01,
                       'b_state_y1_affirm_factor': 0.005, },
        'y2_drive': {'from_y1': 0.06, 'from_y3': 0.02, 'from_risk_success_激励': 0.16, 'from_risk_failure_打击': -0.18,
                     'loss_b2': 0.095, 'loss_low_y1': 0.23, 'sustain_cost_low_b1': 0.028, 'burnout_factor_base': 0.012,
                     'b_state_y2_sap_factor': 0.012, 'b_state_y2_boost_factor': 0.006, },
        'y3_aspiration': {'adjust_y2': 0.05, 'boost_y1': 0.007, 'social_norm_b1': 0.003, 'social_norm_h1': 0.0025,
                          'self_h1_factor': 0.004, 'loss_reality_gap': 0.035, 'complacency_drag_factor': 0.006,
                          'b_success_y3_lift_factor': 0.004, },
        'h1_possibilities': {'from_b1': 0.08, 'from_h2': 0.10, 'loss_b2': 0.135, 'loss_low_y_factor': 0.02,
                             'focus_cost_factor': 0.008, 'b2_h1_suppression_factor': 0.01, },
        'h2_innovation': {'from_y2': 0.09, 'from_h3': 0.06, 'decay_no_practice': 0.07, 'complexity_cost_factor': 0.007,
                          'y3_h2_drive_factor': 0.005, },
        'h3_risk_appetite': {'from_risk_success_回报': 0.18, 'from_risk_failure_惩罚': -0.22, 'from_y1': 0.04,
                             'from_y2': 0.07, 'loss_b2': 0.115, 'stability_caution_factor': 0.006,
                             'desperation_risk_factor': 0.004, },
        's1_trustworthiness': {'from_consistency': 0.025, 'penalty_risk_failure': 0.035, 'decay': 0.008, },
        's2_reputation': {'from_achievement': 0.03, 'from_value_appeal': 0.015, 'bonus_risk_success': 0.035,
                          'decay': 0.01, 'from_b1_overflow': 0.05, },
        'risk_project': {'potential_R_base': 0.04, 'potential_R_h2': 0.02, 'potential_R_y1': 0.01,
                         'potential_R_s2': 0.005, 'inherent_L_base': 0.18, 'inherent_L_h3': 0.02,
                         'inherent_L_b2': 0.018, 'reduction_s1': 0.005, 'L_to_mean_factor': 0.6, 'L_to_std_factor': 0.7,
                         'min_std_dev': 0.01, 'b2_from_major_failure': 0.1, },
        'social_interactions': {'trust_formation_rate': 0.06, 'y1_alignment_factor': 0.005, 'y2_contagion_factor': 0.01,
                                'y3_alignment_factor': 0.004, 's1_social_norm_factor': 0.004,
                                's2_social_pressure_factor': 0.003, 'h2_info_share_factor': 0.008,
                                'h2_info_share_min_diff': 0.5, 'b1_comp_diff_thresh': 1.0, 'b1_comp_loss_factor': 0.018,
                                'b1_coop_diff_thresh': 0.6, 'b1_coop_gain_factor': 0.0008, 'b1_coop_min_self_res': 2.0,
                                'trust_effect_min_coop': 0.4, 'trust_effect_max_coop': 1.6, 'base_comp_factor': 1.1,
                                'rival_comp_loss_multiplier': 1.5, 'rival_coop_gain_multiplier': 0.3,
                                'alliance_comp_loss_multiplier': 0.6, 'alliance_coop_gain_multiplier': 1.4,
                                'rival_h2_share_multiplier': 0.1, 'alliance_h2_share_multiplier': 1.6,
                                'rival_harm_trust_thresh': 3.0, 'rival_sabotage_b2_factor': 0.006,
                                'rival_smear_s2_factor': 0.004, 'alliance_majority_for_y_align_boost': 0.6,
                                'alliance_y_align_multiplier': 1.2, 'rival_majority_for_y_align_reduction': 0.4,
                                'rival_y_align_multiplier': 0.8, 'alliance_form_my_trust_thresh': 7.2,
                                'alliance_form_their_trust_thresh': 7.2, 'alliance_trust_factor': 0.35,
                                'alliance_form_y1_sim_thresh': 7.5, 'alliance_y1_sim_factor': 0.3,
                                'alliance_form_y3_sim_thresh': 7.0, 'alliance_y3_sim_factor': 0.25,
                                'alliance_form_base_prob': 0.12, 'rival_form_my_trust_thresh': 2.8,
                                'rival_trust_factor': 0.45, 'rival_form_b1_diff_thresh': 2.5,
                                'rival_form_self_y2_thresh': 6.5, 'rival_form_other_y2_thresh': 6.5,
                                'rival_b1_comp_factor': 0.35, 'rival_form_base_prob': 0.1,
                                'alliance_break_trust_thresh': 3.5, 'alliance_break_y1_diff_thresh': 6.0,
                                'alliance_break_prob_y_diverge': 0.04, 'rival_end_trust_thresh': 6.5,
                                'rival_end_prob_trust_improve': 0.1,
                                'h3_norm_pressure_factor': 0.0025, }},
    'plot_weights': {'b': (0.6, 0.4), 'y': (0.35, 0.35, 0.3), 'h': (0.35, 0.35, 0.3)}}

default_evolution_params_gm456 = default_evolution_params_gm455_complete.copy()
default_evolution_params_gm456.update({
    'perception_error_scale': 4.0, 'mood_risk_success_thresh': 0.08, 'mood_risk_failure_thresh': -0.08,
    'mood_b1_low_thresh': 2.2, 'mood_b1_high_thresh': 6.8, 'risk_min_perceived_b1_to_attempt': 0.8,
    'show_perception_in_hover': True, 'show_mood_in_hover': True, })
coeffs_gm456 = json.loads(json.dumps(default_evolution_params_gm456['coefficients']))
coeffs_gm456['cognitive_model'] = {
    'y1_to_acc_min': 0.3, 'y1_to_acc_max': 0.95, 'h2_to_acc_bonus': 0.15, 'mood_pos_bias_factor': 0.08,
    'mood_neg_bias_factor': -0.08,
    'mood_from_risk_success': 0.6, 'mood_from_risk_failure': -0.6, 'mood_from_low_b1': 0.12, 'mood_from_high_b1': 0.06,
    'mood_from_alliances': 0.04, 'mood_from_rivals': 0.04, 'mood_change_rate': 0.35, 'mood_min': -1.0, 'mood_max': 1.0,
    'mood_decay_factor': 0.93, }
coeffs_gm456['community_actions'] = {
    'initiate_project_y2_thresh': 6.8, 'initiate_project_s2_thresh': 5.2, 'initiate_project_prob': 0.015,
    'project_b1_cost_ratio_self': 0.25, 'project_req_h2': 3.5, 'project_duration': 12, 'project_target_participants': 3,
    'project_b2_reward_val': 1.0, 'join_project_trust_thresh': 4.8, 'join_trust_factor': 0.25,
    'join_project_value_ratio_thresh': 1.3, 'join_value_factor': 0.35, 'join_afford_factor': 0.15,
    'join_project_base_prob': 0.12, }
default_evolution_params_gm456['coefficients'] = coeffs_gm456

global_environment_factors_gm456 = {
    'b1_acq_mod': 1.0, 'b2_base_inc': 0.0, 'event_crisis_likelihood_mod': 1.0, 'econ_cycle_event_mod': 1.0}

# This part is now corrected:
initial_states_obj_list_gm456 = [WorldState(**s_data) for s_data in initial_states_templates_gm456]
for state_obj in initial_states_obj_list_gm456:  # THIS LOOP IS NOW PRESENT
    state_obj.neighbors = neighbor_config_gm456.get(state_obj.name_en, [])
initial_world_states_store_data_gm456 = {s.name_en: s.to_dict() for s in initial_states_obj_list_gm456}

event_manager = EventManager(event_definitions_gm456)
COMMUNITY_PROJECTS_STORE = {}
# --- 5. Dash App Layout (GM4.5.6) ---
app = dash.Dash(__name__, suppress_callback_exceptions=True)
app.title = "三世界理论 - v4.5.6 (社群动态版)"
app.layout = html.Div([
    html.H1(app.title, style={'textAlign': 'center', 'color': '#2c3e50'}),
    dcc.Store(id='world-states-store', data=initial_world_states_store_data_gm456),
    dcc.Store(id='selected-point-id-store', data=None),
    dcc.Store(id='evolution-params-store', data=default_evolution_params_gm456),
    dcc.Store(id='global-env-factors-store', data=global_environment_factors_gm456),
    dcc.Store(id='simulation-log-store', data=list(SIMULATION_LOG)),
    dcc.Store(id='community-projects-store', data={pid: p.to_dict() for pid, p in COMMUNITY_PROJECTS_STORE.items()}),
    dcc.Interval(id='evolution-interval', interval=1000, n_intervals=0, disabled=True, max_intervals=-1),
    html.Div([
        html.Div([
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
            html.Div(id='neighbor-info-div', style={'fontSize': 'small', 'marginBottom': '10px', 'padding': '5px',
                                                    'border': '1px dashed #ccc'}),
            html.Div(id='trust-info-div', style={'fontSize': 'small', 'marginBottom': '10px', 'padding': '5px',
                                                 'border': '1px dashed #bbf'}),
            html.Div(id='social-relations-display-div',
                     style={'fontSize': 'small', 'marginBottom': '10px', 'padding': '8px', 'border': '1px solid #aaf',
                            'backgroundColor': '#f0f8ff'}),
            html.Div(id='cognitive-info-display-div',
                     style={'fontSize': 'small', 'marginBottom': '10px', 'padding': '8px', 'border': '1px solid #fab',
                            'backgroundColor': '#fff0f5'}),
            html.H4("全局环境因子", style={'marginTop': '15px', 'borderTop': '1px solid #ccc', 'paddingTop': '10px'}),
            html.Label("B1获取效率修正:", style={'fontSize': 'small'}),
            dcc.Slider(id='env-b1-acq-slider', min=0.5, max=1.5, step=0.05,
                       value=global_environment_factors_gm456['b1_acq_mod'],
                       marks={i / 10: str(i / 10) for i in range(5, 16, 2)}, tooltip={"placement": "bottom"}),
            html.Label("B2基础增量:", style={'fontSize': 'small'}),
            dcc.Slider(id='env-b2-inc-slider', min=-0.1, max=0.2, step=0.01,
                       value=global_environment_factors_gm456.get('b2_base_inc', 0.0),
                       marks={i / 100: str(i / 100) for i in range(-10, 21, 5)}, tooltip={"placement": "bottom"}),
            # Used .get() for safety
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
                       value=default_evolution_params_gm456['learning_rate'],
                       marks={i / 100: f"{i / 100:.2f}" for i in range(1, 11, 1)},
                       tooltip={"placement": "bottom", "always_visible": True}),
            html.Label("噪声水平:", style={'fontWeight': 'bold', 'marginTop': '10px'}),
            dcc.Slider(id='noise-slider', min=0, max=0.03, step=0.001,
                       value=default_evolution_params_gm456['noise_level'],
                       marks={i / 1000: f"{i / 1000:.3f}" for i in range(0, 31, 5)},
                       tooltip={"placement": "bottom", "always_visible": True}),
            html.Div(id='n-intervals-display', style={'marginTop': '15px', 'fontSize': 'small', 'color': 'gray'}),
            html.H4("参数调整 (示例)",
                    style={'marginTop': '15px', 'borderTop': '1px solid #ccc', 'paddingTop': '10px'}),
            html.Label("B1维护指数:", style={'fontSize': 'small'}),
            dcc.Slider(id='param-b1-maint-exp-slider', min=1.0, max=3.0, step=0.1,
                       value=default_evolution_params_gm456['b1_maintenance_exponent'],
                       marks={i / 10: str(i / 10) for i in range(10, 31, 5)}, tooltip={"placement": "bottom"}),
            html.Label("Y2倦怠基础因子:", style={'fontSize': 'small'}),
            dcc.Input(id='param-y2-burnout-factor-input', type='number',
                      value=default_evolution_params_gm456['coefficients']['y2_drive']['burnout_factor_base'],
                      step=0.001, style={'width': '80px'}),
        ], style={'width': '30%', 'float': 'left', 'padding': '20px', 'boxSizing': 'border-box',
                  'backgroundColor': '#f0f4f8', 'borderRight': '1px solid #ccc', 'maxHeight': '90vh',
                  'overflowY': 'auto'}),
        html.Div([
            html.Div(dcc.Graph(id='main-3d-scatter-plot', style={'height': '55vh'})),
            html.Div([
                html.H5("社区项目:", style={'marginTop': '5px', 'marginBottom': '5px'}),
                html.Div(id='community-projects-display-div',
                         style={'width': '100%', 'height': '10vh', 'fontSize': 'x-small', 'border': '1px solid #afa',
                                'backgroundColor': '#f0fff0', 'overflowY': 'auto', 'padding': '5px'})]),
            html.Div(
                [html.H5("模拟日志 (最新 %s 条):" % MAX_LOG_LINES, style={'marginTop': '5px', 'marginBottom': '5px'}),
                 dcc.Textarea(id='simulation-log-textarea', value="", readOnly=True,
                              style={'width': '100%', 'height': '13vh', 'fontSize': 'x-small',
                                     'border': '1px solid #ddd', 'backgroundColor': '#fafafa',
                                     'fontFamily': 'monospace'})])
        ], style={'width': '70%', 'float': 'right', 'padding': '10px', 'boxSizing': 'border-box'})]),
    html.Div(style={'clear': 'both'}),
    html.Footer(f"三世界理论模型 - v4.5.6 (高级社交,认知,事件) - {time.strftime('%Y-%m-%d %H:%M:%S')}",
                style={'textAlign': 'center', 'marginTop': '20px', 'padding': '10px', 'fontSize': 'x-small',
                       'color': '#888'})
], style={'fontFamily': "'Segoe UI', Tahoma, Geneva, Verdana, sans-serif", 'maxWidth': '1800px', 'margin': 'auto',
          'backgroundColor': '#e9ecef'})


# --- 6. Callbacks (GM4.5.6) ---
@app.callback(Output('trust-info-div', 'children'),
              [Input('point-selector-dropdown', 'value'), Input('world-states-store', 'data')])
def display_trust_info_gm456(selected_id, states_data_json):
    if not selected_id or not states_data_json or selected_id not in states_data_json: return "选择状态点查看信任信息。"
    state_data = states_data_json[selected_id];
    trust_levels_dict = state_data.get('trust_levels', {})
    if not trust_levels_dict: return f"{state_data.get('name_zh', 'N/A')} 尚无信任记录。"
    trust_details = [f"{states_data_json.get(n_id, {}).get('name_zh', n_id)}: {trust_val:.1f}" for n_id, trust_val in
                     trust_levels_dict.items()]
    return html.Div(
        [html.Span(html.B(f"{state_data.get('name_zh', 'N/A')} 的信任级别:")), html.Br()] + [html.Span(f"{detail}, ")
                                                                                             for detail in
                                                                                             trust_details[:-1]] + (
            [html.Span(trust_details[-1])] if trust_details else [html.Span("无具体信任对象。")]))


@app.callback(Output('social-relations-display-div', 'children'),
              [Input('point-selector-dropdown', 'value'), Input('world-states-store', 'data')])
def display_social_relations_gm456(selected_id, states_data_json):
    if not selected_id or not states_data_json or selected_id not in states_data_json: return "选择状态点查看社交关系。"
    state_data = states_data_json[selected_id];
    state_name_zh = state_data.get('name_zh', selected_id)
    allies = state_data.get('alliance_partners', []);
    rivals = state_data.get('rivals', [])
    children = [html.B(f"{state_name_zh} 的社交关系:")]
    if allies:
        ally_names = [states_data_json.get(ally_en, {}).get('name_zh', ally_en) for ally_en in allies];
        children.extend(
            [html.Br(), html.Span(f"联盟: {', '.join(ally_names)}", style={'color': 'green'})])
    else:
        children.extend([html.Br(), html.Span("无联盟伙伴", style={'color': 'gray'})])
    if rivals:
        rival_names = [states_data_json.get(rival_en, {}).get('name_zh', rival_en) for rival_en in
                       rivals];
        children.extend(
            [html.Br(), html.Span(f"对手: {', '.join(rival_names)}", style={'color': 'red'})])
    else:
        children.extend([html.Br(), html.Span("无特定对手", style={'color': 'gray'})])
    return html.Div(children)


@app.callback(Output('cognitive-info-display-div', 'children'),
              [Input('point-selector-dropdown', 'value'), Input('world-states-store', 'data'),
               Input('evolution-params-store', 'data')])
def display_cognitive_info_gm456(selected_id, states_data_json, evo_params):
    if not selected_id or not states_data_json or selected_id not in states_data_json: return "选择状态点查看认知信息。"
    state_data = states_data_json[selected_id];
    state_name_zh = state_data.get('name_zh', selected_id)
    children = [html.B(f"{state_name_zh} 的认知状态:")]
    if evo_params.get('show_perception_in_hover', True):
        children.extend([html.Br(), html.Span(
            f"感知B1: {state_data.get('perceived_b1_resource', 0):.2f} (真实B1: {state_data.get('b1_resource', 0):.2f})")])
        children.extend([html.Br(), html.Span(
            f"感知H1: {state_data.get('perceived_h1_possibilities', 0):.2f} (真实H1: {state_data.get('h1_possibilities', 0):.2f})")])
        children.extend([html.Br(), html.Span(f"感知准确度: {state_data.get('perception_accuracy', 0):.2f}")])
    if evo_params.get('show_mood_in_hover', True):
        mood_val = state_data.get('mood', 0);
        mood_str = "中性"
        if mood_val > 0.5:
            mood_str = "积极"
        elif mood_val < -0.5:
            mood_str = "消极"
        children.extend([html.Br(), html.Span(f"情绪: {mood_str} ({mood_val:.2f})")])
    return html.Div(children)


@app.callback(Output('community-projects-display-div', 'children'), [Input('community-projects-store', 'data')])
def display_community_projects_gm456(projects_data):
    if not projects_data: return "当前无社区项目。"
    active_projects = [p for p_id, p in projects_data.items() if p['status'] == 'active' or p['status'] == 'pending']
    if not active_projects: return "当前无激活或待处理的社区项目。"
    elements = [html.B("进行中的社区项目:")]
    for proj in active_projects:
        participants_str = f"{len(proj['participants'])}/{proj['target_participants']}"
        elements.extend([html.Hr(style={'margin': '3px 0'}),
                         html.Span(f"{proj['name']} (ID: {proj['project_id'][:4]}) - 发起者: {proj['creator_en'][:5]}"),
                         html.Br(),
                         html.Span(
                             f"状态: {proj['status']}, 参与者: {participants_str}, 已贡献B1: {proj['contributed_b1']:.1f}/{proj['required_b1_total']:.1f}"),
                         html.Br(),
                         html.Span(
                             f"进度: {proj['current_duration']}/{proj['duration_total']} 回合, 奖励: {proj['reward_type']} {proj['reward_value']:.1f}")])
    return html.Div(elements)


@app.callback(Output('global-env-factors-store', 'data'),
              [Input('env-b1-acq-slider', 'value'), Input('env-b2-inc-slider', 'value')],
              [State('global-env-factors-store', 'data')])
def update_global_env_factors_gm456(b1_acq_mod, b2_base_inc, current_env_data):
    if b1_acq_mod is None or b2_base_inc is None: raise PreventUpdate
    new_env_data = current_env_data.copy() if current_env_data else {};
    new_env_data['b1_acq_mod'] = float(b1_acq_mod);
    new_env_data['b2_base_inc'] = float(b2_base_inc)
    return new_env_data


@app.callback([Output('point-selector-dropdown', 'options'), Output('point-selector-dropdown', 'value')],
              [Input('world-states-store', 'data')], [State('selected-point-id-store', 'data')])
def update_point_selector_gm456(states_data_json, selected_point_id):
    if not states_data_json: return [], None
    options = [{'label': f"{s_data['name_zh']} ({s_data['name_en']})", 'value': s_data['name_en']} for s_id, s_data in
               states_data_json.items() if isinstance(s_data, dict) and 'name_en' in s_data and 'name_zh' in s_data]
    valid_ids = [opt['value'] for opt in options];
    current_value = selected_point_id if selected_point_id in valid_ids else (options[0]['value'] if options else None)
    return options, current_value


@app.callback(Output('edit-panel-div', 'children'), Input('point-selector-dropdown', 'value'),
              State('world-states-store', 'data'))
def update_edit_panel_gm456(selected_id, states_data_json):
    if not selected_id or not states_data_json or selected_id not in states_data_json: return html.P(
        "请选择一个状态点进行编辑。", style={'color': 'orange'})
    state_data = states_data_json[selected_id]
    if not isinstance(state_data, dict): return html.P(f"加载状态点 '{selected_id}' 数据格式错误。",
                                                       style={'color': 'red'})
    try:
        state_obj = WorldState.from_dict(state_data)
    except Exception as e:
        log_message("ERROR", f"EditPanel: Error creating WorldState {selected_id}: {e}\n{traceback.format_exc()}",
                    "UI");
        return html.P(f"加载状态点 '{selected_id}' 数据时出错: {e}", style={'color': 'red'})
    panel_children = [
        html.H4(f"编辑: {state_obj.get_display_name()}", style={'marginTop': '0', 'marginBottom': '10px'})]
    for key_dim in DIM_KEYS:
        label = DIMENSION_LABELS_MAP_ZH.get(key_dim, key_dim.replace('_', ' ').title());
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
              Input({'type': 'dim-slider', 'index': ALL}, 'value'), State({'type': 'dim-slider', 'index': ALL}, 'id'),
              State('point-selector-dropdown', 'value'), State('world-states-store', 'data'), prevent_initial_call=True)
def update_state_from_sliders_gm456(slider_values, slider_ids_obj_list, selected_id, states_data_json):
    ctx = callback_context
    if not ctx.triggered or not selected_id or not states_data_json or selected_id not in states_data_json: return no_update
    triggered_input = ctx.triggered[0];
    slider_key, slider_value = None, None
    if isinstance(ctx.triggered_id, dict) and 'index' in ctx.triggered_id:
        slider_key = ctx.triggered_id['index'];
        slider_value = triggered_input['value']
    else:
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
def update_selected_point_id_store_val_gm456(selected_id): return selected_id


@app.callback([Output('evolution-interval', 'disabled'), Output('toggle-evolution-button', 'children')],
              [Input('toggle-evolution-button', 'n_clicks')], [State('evolution-interval', 'disabled')])
def toggle_evolution_gm456(n_clicks, disabled_state):
    if n_clicks == 0: return True, '开始演化'
    is_now_disabled = not disabled_state
    return is_now_disabled, '暂停演化' if not is_now_disabled else '开始演化'


@app.callback(Output('evolution-interval', 'interval'), Input('evolution-interval-input', 'value'))
def update_evolution_interval_time_gm456(value): return int(value) if value and int(value) >= 100 else 1000


@app.callback(Output('evolution-params-store', 'data'),
              [Input('lr-slider', 'value'), Input('noise-slider', 'value'),
               Input('param-b1-maint-exp-slider', 'value'),
               Input('param-y2-burnout-factor-input', 'value')],
              [State('evolution-params-store', 'data')])
def update_evolution_hyperparams_gm456(lr, noise, b1_maint_exp, y2_burnout_factor, params_json):
    if lr is None or noise is None or b1_maint_exp is None or y2_burnout_factor is None: raise PreventUpdate
    new_params = json.loads(json.dumps(params_json))
    new_params['learning_rate'] = float(lr);
    new_params['noise_level'] = float(noise)
    new_params['b1_maintenance_exponent'] = float(b1_maint_exp)
    if 'coefficients' in new_params and 'y2_drive' in new_params['coefficients']:
        new_params['coefficients']['y2_drive']['burnout_factor_base'] = float(y2_burnout_factor)
    else:
        log_message("WARN", "Params structure issue: coefficients.y2_drive missing.", "ParamsUpdate")
    return new_params


@app.callback(
    [Output('world-states-store', 'data'), Output('n-intervals-display', 'children'),
     Output('simulation-log-store', 'data'), Output('community-projects-store', 'data')],
    [Input('evolution-interval', 'n_intervals'), Input('step-evolution-button', 'n_clicks')],
    [State('world-states-store', 'data'), State('evolution-params-store', 'data'),
     State('evolution-interval', 'disabled'), State('coord-type-dropdown', 'value'),
     State('simulation-log-store', 'data'), State('global-env-factors-store', 'data'),
     State('evolution-interval', 'n_intervals'), State('community-projects-store', 'data')])
def run_evolution_step_gm456(n_auto_interval, n_manual_click, states_json, evo_params_json, interval_disabled,
                             coord_type, current_sim_log_list, env_json, interval_current_n_intervals,
                             community_projects_json_data):
    ctx = callback_context;
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None
    if not triggered_id or (triggered_id == 'evolution-interval' and interval_disabled): raise PreventUpdate
    if not all([states_json, evo_params_json, env_json]): log_message("WARN", "Evo step: missing data.",
                                                                      "EvoLoop");raise PreventUpdate
    current_sim_step_for_logic = 0;
    display_step_type = ""
    if triggered_id == 'evolution-interval':
        current_sim_step_for_logic = n_auto_interval;
        display_step_type = "Auto"
    elif triggered_id == 'step-evolution-button':
        current_sim_step_for_logic = (
                                             interval_current_n_intervals or 0) + n_manual_click;
        display_step_type = f"Manual ({n_manual_click})"
    step_specific_log_messages = []
    obj_dict = {}
    for name, data in states_json.items():
        try:
            obj_dict[name] = WorldState.from_dict(data)
        except Exception as e:
            step_specific_log_messages.append(f"CRITICAL: Load fail {name}: {e}\n{traceback.format_exc()}");
            obj_dict[
                name] = None
    current_community_projects = {}
    if community_projects_json_data:
        for pid, p_data in community_projects_json_data.items():
            try:
                current_community_projects[pid] = CommunityProject.from_dict(p_data)
            except Exception as e:
                step_specific_log_messages.append(f"ERROR: Load project {pid} fail: {e}")
    valid_states_for_metrics = [s for s in obj_dict.values() if s is not None]
    metrics = {'avg_b1_resource': np.mean(
        [s.b1_resource for s in valid_states_for_metrics]) if valid_states_for_metrics else 0,
               'avg_y1_clarity': np.mean(
                   [s.y1_clarity for s in valid_states_for_metrics]) if valid_states_for_metrics else 0,
               'avg_h1_possibilities': np.mean(
                   [s.h1_possibilities for s in valid_states_for_metrics]) if valid_states_for_metrics else 0,
               'avg_s1_trust': np.mean(
                   [s.s1_trustworthiness for s in valid_states_for_metrics]) if valid_states_for_metrics else 0,
               'avg_s2_reputation': np.mean(
                   [s.s2_reputation for s in valid_states_for_metrics]) if valid_states_for_metrics else 0, }
    effects_by_event, trigger_msgs = event_manager.process_step({k: v for k, v in obj_dict.items() if v is not None},
                                                                metrics, env_json, current_sim_step_for_logic)
    step_specific_log_messages.extend(trigger_msgs)
    projects_to_remove = []
    for proj_id, project_obj in current_community_projects.items():
        project_obj.progress_project(obj_dict)
        if project_obj.status == "completed" or project_obj.status == "failed": projects_to_remove.append(proj_id)
    for proj_id in projects_to_remove: del current_community_projects[proj_id];step_specific_log_messages.append(
        f"项目 {proj_id} 已结束并从活动列表移除。")
    updated_json_out = {};
    max_history = evo_params_json.get('max_history_points', 50)
    for name, obj in obj_dict.items():
        if obj is None: updated_json_out[name] = states_json[name];continue
        active_effs_for_this_obj = effects_by_event.get(name, {})
        try:
            obj.evolve(evo_params_json, obj_dict, active_effs_for_this_obj, env_json, current_sim_step_for_logic,
                       current_community_projects)
            weights = evo_params_json.get('plot_weights', default_evolution_params_gm456['plot_weights'])
            obj.record_history(coord_type, weights['b'], weights['y'], weights['h'], max_history)
            updated_json_out[name] = obj.to_dict()
            if obj.active_effects_log: step_specific_log_messages.extend(
                [f"状态 '{obj.name_zh}': {log}" for log in obj.active_effects_log])
        except Exception as e:
            step_specific_log_messages.append(
                f"ERROR: Evolution failed for state {name}: {e}\n{traceback.format_exc()}. State preserved.");
            updated_json_out[name] = states_json[name]
    if step_specific_log_messages:
        timestamp_header = f"--- 步骤 {current_sim_step_for_logic} ({display_step_type}, {time.strftime('%H:%M:%S')}) ---"
        log_message("INFO", timestamp_header, "EvolutionLoop")
        for msg in step_specific_log_messages:
            level = "INFO";
            if "ERROR:" in msg:
                level = "ERROR"
            elif "WARN:" in msg:
                level = "WARN"
            elif "CRITICAL:" in msg:
                level = "CRITICAL"
            log_message(level, msg.split(":", 1)[1].strip() if ":" in msg else msg, "EvolutionDetail")
    final_log_for_ui = list(SIMULATION_LOG)[-MAX_LOG_LINES:]
    step_info = f"当前迭代: {current_sim_step_for_logic} ({display_step_type})"
    community_projects_to_store = {pid: p.to_dict() for pid, p in current_community_projects.items()}
    return updated_json_out, step_info, final_log_for_ui, community_projects_to_store


@app.callback(Output('simulation-log-textarea', 'value'), Input('simulation-log-store', 'data'))
def update_simulation_log_display_gm456(log_data_list):
    if isinstance(log_data_list, list): return "\n".join(log_data_list)
    return "模拟日志为空或格式错误."


@app.callback([Output('world-states-store', 'data', allow_duplicate=True),
               Output('evolution-interval', 'n_intervals', allow_duplicate=True),
               Output('simulation-log-store', 'data', allow_duplicate=True),
               Output('community-projects-store', 'data', allow_duplicate=True)],
              [Input('reset-states-button', 'n_clicks')], prevent_initial_call=True)
def reset_all_states_gm456(n_clicks):
    global SIMULATION_LOG, COMMUNITY_PROJECTS_STORE
    if n_clicks is None or n_clicks == 0: raise PreventUpdate
    fresh_data = {}
    temp_list = [WorldState(**s_dict) for s_dict in initial_states_templates_gm456]
    for obj in temp_list: obj.neighbors = neighbor_config_gm456.get(obj.name_en, []);obj.clear_history();fresh_data[
        obj.name_en] = obj.to_dict()
    event_manager.reset_events();
    COMMUNITY_PROJECTS_STORE.clear();
    SIMULATION_LOG.clear()
    log_message("INFO", "系统已重置：所有状态、事件、项目和日志已清除。", "SystemReset")
    return fresh_data, 0, list(SIMULATION_LOG), {}


@app.callback(Output('main-3d-scatter-plot', 'figure'),
              [Input('world-states-store', 'data'), Input('coord-type-dropdown', 'value')],
              [State('evolution-params-store', 'data')])
def update_3d_scatter_plot_gm456(states_data_json, coord_type, evo_params_json):
    if not states_data_json: return go.Figure(layout=go.Layout(title="数据加载中...",
                                                               scene=dict(xaxis=dict(range=[0, 10]),
                                                                          yaxis=dict(range=[0, 10]),
                                                                          zaxis=dict(range=[0, 10]),
                                                                          aspectmode='cube')))
    traces = [];
    plot_weights = default_evolution_params_gm456['plot_weights']
    if evo_params_json and 'plot_weights' in evo_params_json and isinstance(evo_params_json['plot_weights'],
                                                                            dict): plot_weights = evo_params_json[
        'plot_weights']
    state_items = list(states_data_json.items())
    for i, (state_id, state_dict) in enumerate(state_items):
        if not isinstance(state_dict, dict): log_message("WARN", f"Plotting: State data for {state_id} not dict.",
                                                         "Plot");continue
        try:
            state_obj = WorldState.from_dict(state_dict)
            current_coords = state_obj.get_coords_for_plot(coord_type, plot_weights.get('b'), plot_weights.get('y'),
                                                           plot_weights.get('h'))
        except Exception as e:
            log_message("ERROR", f"Plotting: Coords error {state_id}: {e}\n{traceback.format_exc()}", "Plot");
            continue
        hover_text_parts = [f"<b>{state_obj.get_display_name()}</b>"]
        for key_dim in DIM_KEYS: label_short = DIMENSION_LABELS_MAP_ZH.get(key_dim, key_dim).split(': ')[-1].split(' ')[
            0];hover_text_parts.append(f"{label_short}: {getattr(state_obj, key_dim, 'N/A'):.1f}")
        if evo_params_json.get('show_perception_in_hover', False): hover_text_parts.append(
            f"感知B1: {state_obj.perceived_b1_resource:.1f}, 感知H1: {state_obj.perceived_h1_possibilities:.1f}")
        if evo_params_json.get('show_mood_in_hover', False): hover_text_parts.append(
            f"情绪: {state_obj.mood:.1f}, 感知准度: {state_obj.perception_accuracy:.1f}")
        coord_type_label = '简化' if coord_type == 'simplified' else '综合';
        hover_text_parts.append(
            f"--- Plot ({coord_type_label}) ---<br>X:{current_coords[0]:.2f}, Y:{current_coords[1]:.2f}, Z:{current_coords[2]:.2f}")
        ### MODIFIED: Use a symbol from the valid list provided by Plotly error ###
        has_allies = bool(state_obj.alliance_partners)
        has_rivals = bool(state_obj.rivals)

        marker_symbol_str = 'circle'  # Default
        marker_line_color = 'rgba(0,0,0,0)'
        marker_line_width = 0

        if has_allies and has_rivals:
            marker_symbol_str = 'x'  # 'x' IS IN THE VALID LIST
            marker_line_color = 'purple'
            marker_line_width = 3
            log_message("WARN", f"State {state_obj.name_en} has both allies and rivals. Plotting with 'x'.", "Plotting")
        elif has_allies:
            marker_symbol_str = 'diamond'  # 'diamond' IS IN THE VALID LIST
            marker_line_color = 'green'
            marker_line_width = 2
        elif has_rivals:
            marker_symbol_str = 'cross'  # 'cross' IS IN THE VALID LIST
            marker_line_color = 'red'
            marker_line_width = 2
        # If neither, it remains 'circle' with no line.
        ### END MODIFIED ###

        marker_color_value = state_obj.y1_clarity
        try:
            traces.append(go.Scatter3d(
                x=[current_coords[0]], y=[current_coords[1]], z=[current_coords[2]], mode='markers+text',
                text=[state_obj.name_zh], textfont=dict(size=9, color="#1f77b4"), textposition='top center',
                marker=dict(size=10, opacity=0.9, color=marker_color_value, colorscale='Blues',
                            symbol=marker_symbol_str, line=dict(color=marker_line_color, width=marker_line_width),
                            showscale=(i == 0),
                            colorbar=dict(title=f"Y1 Clarity", thickness=15, x=1.05) if (i == 0) else None, cmin=0,
                            cmax=10),
                hoverinfo='text', hovertext=["<br>".join(hover_text_parts)], name=state_obj.get_display_name(),
                customdata=[state_id], legendgroup=state_obj.name_en))
        except ValueError as ve_plot:
            log_message("CRITICAL",
                        f"PLOTLY SYMBOL ERROR for {state_id} with '{marker_symbol_str}'. Data: {state_dict}. Error: {ve_plot}\n{traceback.format_exc()}",
                        "Plot")
            try:
                traces.append(go.Scatter3d(
                    x=[current_coords[0]], y=[current_coords[1]], z=[current_coords[2]], mode='markers+text',
                    text=[state_obj.name_zh], textfont=dict(size=9, color="#1f77b4"), textposition='top center',
                    marker=dict(size=10, opacity=0.9, color=marker_color_value, colorscale='Blues', symbol='circle',
                                showscale=(i == 0),
                                colorbar=dict(title=f"Y1 Clarity", thickness=15, x=1.05) if (i == 0) else None, cmin=0,
                                cmax=10),
                    hoverinfo='text', hovertext=["<br>".join(hover_text_parts)], name=state_obj.get_display_name(),
                    customdata=[state_id], legendgroup=state_obj.name_en))
                log_message("WARN", f"Fallback trace plotted for {state_id}.", "Plot")
            except Exception as fallback_e:
                log_message("CRITICAL", f"FALLBACK PLOT ERROR for {state_id}: {fallback_e}\n{traceback.format_exc()}",
                            "Plot")
        if state_obj.history and len(state_obj.history) > 1:
            valid_history = [h for h in state_obj.history if isinstance(h, (list, tuple)) and len(h) == 3 and all(
                isinstance(x, (int, float)) for x in h)]
            if len(valid_history) > 1:
                hist_x, hist_y, hist_z = zip(*valid_history)
                traces.append(go.Scatter3d(x=list(hist_x), y=list(hist_y), z=list(hist_z), mode='lines',
                                           line=dict(width=2, color=f'rgba(100,100,100,0.3)'), hoverinfo='skip',
                                           name=f"{state_obj.name_zh} 历史", showlegend=False,
                                           legendgroup=state_obj.name_en))
    current_axis_labels_dict = AXIS_LABELS_ZH.get(coord_type, {'x': 'X', 'y': 'Y', 'z': 'Z'})
    fig = go.Figure(data=traces)
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=40),
                      title=dict(text=f"三世界坐标 ({'Simp.' if coord_type == 'simplified' else 'Comp.'}) - v4.5.6",
                                 x=0.5, y=0.98, font=dict(size=14)),
                      scene=dict(xaxis_title=current_axis_labels_dict.get('x', 'X轴'),
                                 yaxis_title=current_axis_labels_dict.get('y', 'Y轴'),
                                 zaxis_title=current_axis_labels_dict.get('z', 'Z轴'),
                                 xaxis=dict(range=[0, 10], autorange=False, nticks=6,
                                            backgroundcolor="rgb(235,235,235)", gridcolor="white",
                                            zerolinecolor="white"),
                                 yaxis=dict(range=[0, 10], autorange=False, nticks=6,
                                            backgroundcolor="rgb(235,235,235)", gridcolor="white",
                                            zerolinecolor="white"),
                                 zaxis=dict(range=[0, 10], autorange=False, nticks=6,
                                            backgroundcolor="rgb(235,235,235)", gridcolor="white",
                                            zerolinecolor="white"),
                                 aspectmode='cube', camera=dict(eye=dict(x=1.75, y=1.75, z=0.65))),
                      legend=dict(orientation="v", yanchor="top", y=0.95, xanchor="left", x=0.01,
                                  bgcolor='rgba(255,255,255,0.65)', bordercolor='#bbb', borderwidth=1,
                                  font=dict(size=9)))
    return fig


if __name__ == '__main__':
    app.run(debug=True, port=8065)
