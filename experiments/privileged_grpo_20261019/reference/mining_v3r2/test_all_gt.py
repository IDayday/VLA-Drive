import unittest
import numpy as np
from common import ROOT, read
from guidance import classify, build_guidance


def scored(**updates):
    metrics=dict(PDMS=1.,NC=1.,DAC=1.,TTC=1.,EP=1.,CF=1.,DDC=1.)
    metrics.update(updates)
    return dict(metrics=metrics,events=[],raw_progress_m=1.)


class PolicyTests(unittest.TestCase):
    def setUp(self):self.cfg=read(ROOT/'CONFIG.json')

    def test_low_score_gt_is_retained(self):
        result=classify(scored(PDMS=.4),scored(PDMS=.5),self.cfg)
        self.assertTrue(result['accepted']);self.assertTrue(result['gt_pdms_improves'])

    def test_unsafe_worse_gt_is_retained_and_marked(self):
        result=classify(scored(PDMS=.4),scored(PDMS=0.,NC=0.),self.cfg)
        self.assertTrue(result['accepted']);self.assertFalse(result['gt_safety_pass'])
        self.assertFalse(result['gt_verified_repair']);self.assertFalse(result['gt_pdms_improves'])

    def test_nonfailure_is_not_failure_pair(self):
        result=classify(scored(),scored(PDMS=0.,NC=0.),self.cfg)
        self.assertFalse(result['accepted']);self.assertTrue(result['gt_retained'])

    def test_guidance_does_not_claim_unsafe_gt_success(self):
        trajectory=np.zeros((8,3))
        result=build_guidance(trajectory,trajectory,scored(PDMS=.4),scored(PDMS=0.,NC=0.),'keep straight',self.cfg)
        self.assertIn('not a verified safe repair',result['guidance_text'])
        self.assertNotIn('Do not use this scene',result['guidance_text'])

    def test_two_shards_disjoint_complete(self):
        tokens=list(range(103288));a=tokens[0::2];b=tokens[1::2]
        self.assertFalse(set(a)&set(b));self.assertEqual(set(a)|set(b),set(tokens))


if __name__=='__main__':unittest.main()
