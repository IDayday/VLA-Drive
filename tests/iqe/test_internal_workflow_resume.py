from iqe.internal_baseline.run import operator_paused


def test_operator_pause_stops_workflow_but_planned_resume_boundary_continues():
    def paused(step):
        return {"optimizer_steps": step, "termination": {"termination_reason": "OPERATOR_PAUSED"}}
    assert operator_paused("scorer_pause_500", paused(334))
    assert operator_paused("scorer_finish", paused(342))
    assert not operator_paused("scorer_pause_500", paused(500))
    assert not operator_paused("expert_pause_250", paused(250))
    assert not operator_paused("scorer_finish", {"termination": {"termination_reason": "MAX_STEPS"}})
    assert not operator_paused("pair_bank", {"reused": True})
