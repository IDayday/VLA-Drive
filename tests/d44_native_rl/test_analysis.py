import pytest
from tools.d44_native_rl.analyze import contributions


def test_net_gains_losses_and_fixed_high_cohort():
    def row(score,safe=1):
        return dict(feedback=dict(metrics=dict(score=score,no_at_fault_collisions=safe,drivable_area_compliance=1,time_to_collision_within_bound=1)))
    old=dict(a=row(.95),b=row(0,0),c=row(.5));new=dict(a=row(0,0),b=row(.9),c=row(.6))
    r=contributions(new,old)
    assert r['G_plus']==pytest.approx(1/3)
    assert r['G_minus']==pytest.approx(-.95/3)
    assert r['net']==pytest.approx(.05/3)
    assert r['high_to_low']==1 and r['low_to_high']==1
    assert r['new_joint_unsafe']==1 and r['repaired_joint_unsafe']==1
    assert r['fixed_reference_cohorts']['IL44_high_ge_0.9']['new_zero']==1
