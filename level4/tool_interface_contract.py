"""Versioned tool-interface comparison contracts for the fixed-pair audit.

v1 is the historical 0.001 mm contract used by ``diagnose_tool_schedule``.
v2 implements decision D-1 (2026-10-16, user-approved, pre-registered before
any D1/D2/D4 or S0-S4 result): every tool *position/opening* comparison uses
0.02 mm; the quaternion limit (1e-6) and the shared-reset limit (0.001 mm) are
unchanged.  v2 gates on all position metrics (a superset of the v1 gate keys,
which omitted current/latest and summary/latest), so it is never looser in
coverage than v1.  Neither contract certifies model alignment, elapsed
physical time, task satisfaction or calibrated task thresholds.
"""
import math

CONTRACTS = {
    'tool_interface_contract_v1': {
        'position_limit_mm': 0.001, 'quaternion_limit': 1e-6,
        'gate_scope': 'v1_transform_keys',
        'source': 'level4.candidate_interface.diagnose_tool_schedule (historical)',
    },
    'tool_interface_contract_v2': {
        'position_limit_mm': 0.02, 'quaternion_limit': 1e-6,
        'gate_scope': 'all_position_opening_and_quaternion_metrics',
        'source': 'decision D-1, 2026-10-16 (L4-01)',
    },
}
SHARED_RESET_LIMIT_MM_UNCHANGED = 0.001

# The 14 metric names produced by diagnose_tool_schedule; v2 refuses to judge a
# diagnosis that does not contain exactly these comparisons.
EXPECTED_METRICS = (
    'summary_pos_transform_error_mm', 'summary_pos_left_transform_error_mm',
    'summary_pos_right_transform_error_mm', 'summary_opening_error_mm',
    'finger_1_physical_transform_error_mm', 'finger_2_physical_transform_error_mm',
    'finger_1_latest_previous_error_mm', 'finger_2_latest_previous_error_mm',
    'finger_1_summary_latest_error_mm', 'finger_2_summary_latest_error_mm',
    'finger_1_current_latest_error_mm', 'finger_2_current_latest_error_mm',
    'finger_1_quaternion_error', 'finger_2_quaternion_error',
)


def _is_quaternion(name):
    return name.endswith('quaternion_error')


def _v1_gate_key(name):
    return 'transform_error' in name or name == 'summary_opening_error_mm' or _is_quaternion(name)


def evaluate(metrics, contract):
    """Judge saved diagnosis metrics under one named contract (pure function)."""
    if contract not in CONTRACTS:
        raise ValueError('Unknown tool interface contract: %r' % contract)
    if set(metrics) != set(EXPECTED_METRICS):
        raise ValueError('Diagnosis metrics differ from the expected comparison set: missing=%r extra=%r'
                         % (sorted(set(EXPECTED_METRICS) - set(metrics)), sorted(set(metrics) - set(EXPECTED_METRICS))))
    spec = CONTRACTS[contract]
    rows = {}
    for name in EXPECTED_METRICS:
        value = metrics[name]['max']
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value < 0:
            raise ValueError('Invalid metric value: ' + name)
        limit = spec['quaternion_limit'] if _is_quaternion(name) else spec['position_limit_mm']
        gated = _v1_gate_key(name) if spec['gate_scope'] == 'v1_transform_keys' else True
        rows[name] = {'max': float(value), 'frame': metrics[name]['frame'], 'limit': limit,
                      'unit': 'quaternion_component' if _is_quaternion(name) else 'mm',
                      'within_limit': float(value) <= limit, 'gated': gated}
    failed = sorted(name for name, row in rows.items() if row['gated'] and not row['within_limit'])
    suffix = contract.rsplit('_', 1)[-1]
    return {'contract': contract, 'definition': dict(spec),
            'status': ('passed_' if not failed else 'failed_') + suffix,
            'failed_metrics': failed, 'metrics': rows,
            'worst_gated_position_mm': max(row['max'] for name, row in rows.items()
                                           if row['gated'] and not _is_quaternion(name)),
            'shared_reset_limit_mm_unchanged': SHARED_RESET_LIMIT_MM_UNCHANGED,
            'not_certified': ['model_alignment', 'elapsed_physical_time', 'task_satisfaction',
                              'task_thresholds_rho_eta_epsilon']}
