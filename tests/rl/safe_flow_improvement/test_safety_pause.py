from starVLA.rl.safe_flow_improvement.contracts import SAFETY_METRICS
from starVLA.rl.safe_flow_improvement.iteration import safety_decision


def report(increase, upper_improvement):
    return dict(status='COMPLETE_MATCHED_EVALUATION', scenes=1696, no_best_of_N=True, sha256='sealed-report',
                paired_changes={k: dict(mean=-increase, CI95=[-.03, upper_improvement]) for k in SAFETY_METRICS})


def test_cost_direction_and_sustained_boundary():
    first = safety_decision(report(.006, -.001))
    assert first['status'] == 'CONTINUE_REGISTERED_WINDOW'
    assert safety_decision(report(.006, -.001), first)['status'] == 'PAUSE_VARIANT'
    recovered = safety_decision(report(-.01, .02), first)
    assert all(x['consecutive_sustained'] == 0 for x in recovered['costs'].values())


def test_immediate_and_uncertain_deterioration():
    assert safety_decision(report(.02, .01))['status'] == 'PAUSE_VARIANT'
    uncertain = safety_decision(report(.006, .001))
    assert safety_decision(report(.006, .001), uncertain)['status'] == 'CONTINUE_REGISTERED_WINDOW'
