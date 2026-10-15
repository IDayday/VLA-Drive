from iqe.decision_selection.evaluation import calibrated_margin


def row(gamma, score, switches=10, feasible=True):
    return dict(gamma=gamma,feasible=feasible,metrics=dict(selected_01=score,switch_count=switches))


def test_safe_but_worse_or_tied_margin_does_not_displace_base():
    for value in (.89,.90):
        r=calibrated_margin([row(0,value),row(1,.88)],dict(selected_01=.90))
        assert r['always_Base'] and r['gamma'] is None and r['status']=='NO_NONTRIVIAL_RULE'


def test_margin_uses_fixed_risk_budget_then_mean_then_switch_count_then_gamma():
    r=calibrated_margin([row(0,.99,feasible=False),row(.25,.91,12),row(.5,.91,10),row(1,.91,10)],dict(selected_01=.90))
    assert r['gamma']==1 and not r['always_Base']


def test_no_feasible_rule_returns_explicit_always_base():
    assert calibrated_margin([row(0,.99,feasible=False)],dict(selected_01=.9))['always_Base']
