"""CPU regression for the actual lexical shared-image routing failure."""
import copy
import unittest

import freeze_native_e2_route_v3 as route


class Controls(unittest.TestCase):
    def fixture(self):
        stage={'image_root':'/var/tmp/ddp-full-foresight-20260929/images','scenes':101592,'source_identity':'s','image_hash_manifest':'h'}
        alias={'alias':'/var/tmp/ddp-full-foresight-20260929','shared_resolved':'/mnt/project/structured-world-fgtr-execution-assets/20261011/navsim_current_train_v1','original_stage_bytes_unchanged':True}
        argv=['--config','locked.yaml','--image-root','/mnt/project/structured-world-fgtr-execution-assets/20261011/navsim_current_train_v1/images','--micro-batch','4']
        return argv,stage,alias

    def test_route_fixes_actual_failure_without_science_or_metadata_change(self):
        argv,stage,alias=self.fixture(); old=copy.deepcopy((argv,stage,alias));new=route.replace_image_route(argv,stage,alias)
        self.assertEqual(new,['--config','locked.yaml','--image-root',stage['image_root'],'--micro-batch','4'])
        self.assertEqual((argv,stage,alias),old)

    def test_wrong_worker_alias_cannot_be_enrolled(self):
        argv,stage,alias=self.fixture();alias['alias']='/different'
        with self.assertRaisesRegex(ValueError,'alias'):route.replace_image_route(argv,stage,alias)

    def test_not_same_shared_dataset_rejected(self):
        argv,stage,alias=self.fixture();alias['shared_resolved']='/mnt/project/foreign'
        with self.assertRaisesRegex(ValueError,'alias'):route.replace_image_route(argv,stage,alias)

    def test_metadata_changed_not_claimed_byteidentical(self):
        argv,stage,alias=self.fixture();alias['original_stage_bytes_unchanged']=False
        with self.assertRaises(ValueError):route.replace_image_route(argv,stage,alias)

    def test_missing_image_route_does_not_choose_new_default(self):
        argv,stage,alias=self.fixture()
        with self.assertRaises(ValueError):route.replace_image_route(['--config','locked.yaml'],stage,alias)


if __name__ == '__main__':unittest.main()
