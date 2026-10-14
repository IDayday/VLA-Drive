import numpy as np
import pytest
from iqe.specialist_study.factorial_analysis import contrasts


def test_matched_target_reader_interaction_has_expected_sign_and_does_not_pool_models():
    c=contrasts({'E00':[.1,.2],'E10':[.2,.3],'E01':[.3,.4],'E11':[.6,.7]})
    assert np.allclose(c['optimized_target_head_only'],.1)
    assert np.allclose(c['private_reader_original_GT'],.2)
    assert np.allclose(c['target_reader_interaction'],.2)
    with pytest.raises(Exception,match='matched'):
        contrasts({'E00':[.1],'E10':[.2],'E01':[.3]})
