"""CPU observer fixtures; none are reported as real training evidence."""
import importlib.util
from pathlib import Path
import unittest

spec=importlib.util.spec_from_file_location('collector',Path(__file__).with_name('collect.py'))
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)


def row(step,ts,tag=None):
    return {'completed':step,'updated_unix':ts,'observed_unix':ts+99,'tag':tag,'rank_rss_bytes':100+step}


class TimingTests(unittest.TestCase):
    def test_original_timestamp_not_poll_wall(self):
        rows=[row(0,10),row(1,13),row(2,18)]
        rows[1]['observed_unix']=999
        rows[2]['observed_unix']=9999
        r=c.timing(rows,0)
        self.assertEqual(r['cold_launch_to_status0_seconds'],10)
        self.assertEqual(r['normal_complete_post_update_intervals']['mean'],4)

    def test_missing_step_not_fabricated(self):
        r=c.timing([row(0,10),row(2,18),row(3,21)],0)
        self.assertEqual(r['normal_complete_post_update_intervals']['n'],1)
        self.assertEqual(r['missing_consecutive_intervals'][0]['missing'],[1])

    def test_checkpoint_same_step_separate(self):
        r=c.timing([row(63,100),row(64,105),row(64,180,'native64')],0)
        self.assertEqual(r['normal_complete_post_update_intervals']['mean'],5)
        self.assertEqual(r['same_step_checkpoint_events'][0]['seconds'],75)

    def test_only_post_checkpoint_status_not_normal(self):
        r=c.timing([row(63,100),row(64,180,'native64')],0)
        self.assertEqual(r['normal_complete_post_update_intervals']['n'],0)

    def test_duplicate_and_unknown_coldload(self):
        a=row(1,20)
        r=c.timing([a,a,row(2,24)],0)
        self.assertIsNone(r['cold_launch_to_status0_seconds'])
        self.assertEqual(r['normal_complete_post_update_intervals']['n'],1)


if __name__=='__main__':unittest.main(verbosity=2)
