import torch
import numpy as np
from server.decision_engine import JointDecisionEngine
from shared.types import BehavioralEvidence, SpatialEvidence, TemporalEvidence

# Setup config
config = {
    'trs_reject_thresh': 0.85,
    'trs_warn_thresh': 0.80,
    'trs_min_depth': 10,
    'alpha_downweight': 0.35,
    'theta_anchor_min': 0.25,
    'theta_self': 0.30,
    'delta_temp_mod': 0.50,
    'theta_floor': 0.40,
    'theta_cos': 0.15,
    'warmup_rounds': 0,
}

engine = JointDecisionEngine(config)

# Synthetic data for Client 7 (honest non-IID)
# In round 102: trs_score=0.894933, sim_anchor=0.880882, sim_global=0.010954, sim_s=0.9
beh_ev_honest = BehavioralEvidence(
    sim_self_mean=0.9,
    sim_self_max=0.9,
    sim_self_mad=0.01,
    norm_deviation_self=0.1,
    cadence_consistency=1.0,
    history_depth=11,
    sim_anchor=0.880882,
    sim_frozen_anchor=0.88,
    anchor_drift=0.05,
    consecutive_dw=2,
    gdv_score=0.009144,
    dbp_score=0.903192,
    trs_score=0.894933,
    behavioral_mature=True
)

spa_ev_honest = SpatialEvidence(
    sim_global=0.010954,
    norm_raw=10.0,
    norm_clipped=10.0,
    norm_ratio_median=1.0,
    spatial_coherence=0.423391,
    spatial_reference_count=10,
    spatial_mature=True,
    prc_score=0.341221,
    tra_score=0.45,
    dynamic_bound_C=10.0
)

temp_ev_honest = TemporalEvidence(
    g_i=10.0,
    version_lag=0,
    lower_fence=5.0,
    upper_fence=15.0,
    fence_margin=0.0,
    client_z_score=0.0,
    is_burn_in=False,
    temporal_mature=True
)

outcome_honest = engine.evaluate(
    cid=7,
    current_round=102,
    I_i=1.0,
    P_i=1.0,
    temporal_ev=temp_ev_honest,
    spatial_ev=spa_ev_honest,
    behavioral_ev=beh_ev_honest
)

print(f"Honest Client Outcome: {outcome_honest.action}, Reason: {outcome_honest.primary_reason}")
assert outcome_honest.action == "DOWNWEIGHT", f"Honest client should be downweighted, not rejected! Got: {outcome_honest.action} - {outcome_honest.primary_reason}"

# Synthetic data for Attacker (S2 Mimicry)
# An attacker changes direction, so sim_anchor is low, but trs might still be high if they are rigid
beh_ev_attack = BehavioralEvidence(
    sim_self_mean=0.2, # Not self consistent with minority
    sim_self_max=0.2,
    sim_self_mad=0.1,
    norm_deviation_self=0.1,
    cadence_consistency=1.0,
    history_depth=11,
    sim_anchor=0.1, # Fails minority consistency
    sim_frozen_anchor=0.1,
    anchor_drift=0.5,
    consecutive_dw=0,
    gdv_score=0.009,
    dbp_score=0.95,
    trs_score=0.94, # Rigid
    behavioral_mature=True
)

spa_ev_attack = SpatialEvidence(
    sim_global=0.10, # Out of consensus
    norm_raw=10.0,
    norm_clipped=10.0,
    norm_ratio_median=1.0,
    spatial_coherence=0.4,
    spatial_reference_count=10,
    spatial_mature=True,
    prc_score=0.1,
    tra_score=0.1,
    dynamic_bound_C=10.0
)

outcome_attack = engine.evaluate(
    cid=5,
    current_round=102,
    I_i=1.0,
    P_i=1.0,
    temporal_ev=temp_ev_honest,
    spatial_ev=spa_ev_attack,
    behavioral_ev=beh_ev_attack
)

print(f"Attacker Outcome: {outcome_attack.action}, Reason: {outcome_attack.primary_reason}")
assert outcome_attack.action == "REJECT", f"Attacker should be rejected! Got: {outcome_attack.action} - {outcome_attack.primary_reason}"

print("Offline verification PASSED!")
