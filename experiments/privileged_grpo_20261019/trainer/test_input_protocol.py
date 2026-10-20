"""Input-protocol guards without loading model weights or scene assets."""
import sys, unittest
from pathlib import Path
from types import SimpleNamespace as NS
import torch
sys.path.insert(0,str(Path(__file__).resolve().parent/'vendor/grpo_smoke_v1'))
from input_protocol import INPUT_PROTOCOL_VERSION, suffix, validate_inputs, validate_protocol, native_clean_suffix


def model():
    return NS(w_depth=0, config=NS(datasets=NS(
        video_data=NS(load_2d_data=0), gs_data=NS(load_3d_data=0),
        reward_data=NS(load_reward_data=0))), robot_history_token='<history>',
        act_query_tokens=[f'<action{i}>' for i in range(8)],
        rgb_query_tokens=['<rgb>'], gs_query_tokens=['<gs>'], reward_query_tokens=['<reward>'],
        _special_token_ids={'history':(10,), 'action':tuple(range(11,19)),
                            'rgb':(30,), 'gs':(31,), 'reward':(32,)})


def batch(values):
    ids=torch.tensor([values]);return {'input_ids':ids,'attention_mask':torch.ones_like(ids)}


class Tests(unittest.TestCase):
    def test_clean_suffix_keeps_history_and_action_order(self):
        m=model();self.assertEqual(suffix(m),' <history>'+''.join(m.act_query_tokens))
        validate_inputs(m,batch([1,2,*range(10,19),3]))

    def test_actual_ids_reject_inactive_missing_duplicate_and_reordered_queries(self):
        for values in ([1,30,*range(10,19)], [1,*range(10,18)],
                       [1,10,*range(10,19)], [1,11,10,*range(12,19)]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                validate_inputs(model(),batch(values))

    def test_auxiliary_branch_and_legacy_protocol_rejected(self):
        m=model();m.config.datasets.reward_data.load_reward_data=1
        with self.assertRaises(ValueError):suffix(m)
        for cfg in ({}, {'input_protocol_version':'legacy'}):
            with self.assertRaises(ValueError):validate_protocol(cfg)
        validate_protocol({'input_protocol_version':INPUT_PROTOCOL_VERSION})

    def test_native_replay_restores_token_lists_after_exception(self):
        m=model();old=m.rgb_query_tokens
        with self.assertRaisesRegex(RuntimeError, 'sentinel'):
            with native_clean_suffix(m):
                self.assertEqual(m.rgb_query_tokens+m.gs_query_tokens+m.reward_query_tokens, [])
                self.assertEqual(len(m.act_query_tokens),8)
                raise RuntimeError('sentinel')
        self.assertIs(m.rgb_query_tokens,old)
        self.assertEqual(m.reward_query_tokens,['<reward>'])

if __name__=='__main__':unittest.main()
