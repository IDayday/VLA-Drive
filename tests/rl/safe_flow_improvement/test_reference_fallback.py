from dataclasses import replace

import numpy as np
import pytest

from starVLA.rl.safe_flow_improvement.candidate_bank import Candidate, deduplicate
from starVLA.rl.safe_flow_improvement.contracts import SFT_SHA256, V1_COMPONENTS
from starVLA.rl.safe_flow_improvement.constrained_weights import solve_weights


def test_rejected_proposal_fallback_preserves_reference_and_empirical_masses():
    references=[]
    for i in range(4):
        references.append(Candidate('abc','log','reference_ode',SFT_SHA256,
                                    np.zeros((8,4)),np.full((8,3),float(i)),np.zeros((41,11)),
                                    .6+.01*i,{k:1. for k in V1_COMPONENTS},'locked',{'seed':1101+i},is_reference=True))
    # Twenty-eight rejected proposal slots return recorded references. They are
    # neither new discoveries nor additional fixed q_ref observations.
    fallbacks=[replace(references[i%4],source='reference_fallback',is_reference=False,
                       source_parameters={'failure_status':'INVALID_PHYSICAL_PROPOSAL','original_reference_index':i%4})
               for i in range(28)]
    unique,qref,empirical,groups=deduplicate(references+fallbacks)
    assert len(unique)==4 and [len(g) for g in groups]==[8]*4
    np.testing.assert_array_equal(qref,np.full(4,.25))
    np.testing.assert_array_equal(empirical,np.full(4,.25))
    result=solve_weights(np.array([c.score for c in unique]),np.zeros((4,3)),qref,empirical,eta=.005)
    assert result.diagnostics['primal_residual']<=1e-9
    assert result.diagnostics['target_primal_residual']<=1e-9
    with pytest.raises(ValueError,match='actual samples'):
        fallbacks[0].require_nft_behavior(SFT_SHA256)


def test_fallback_cannot_claim_a_current_policy_or_fixed_reference_identity():
    candidate=Candidate('abc','log','reference_fallback','different_behavior',
                         np.zeros((8,4)),np.zeros((8,3)),np.zeros((41,11)),.8,
                         {k:1. for k in V1_COMPONENTS},'locked',{},
                         source_parameters={'failure_status':'INVALID_PHYSICAL_PROPOSAL','original_reference_index':0})
    with pytest.raises(ValueError,match='actual fixed SFT'):
        candidate.validate()
    candidate.behavior_sha256=SFT_SHA256
    candidate.is_reference=True
    with pytest.raises(ValueError,match='Reference identity'):
        candidate.validate()
