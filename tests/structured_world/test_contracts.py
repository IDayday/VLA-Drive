"""Coordinate, event, RNG and deployment-gradient invariants for round one."""
import unittest
import numpy as np
import torch
from omegaconf import OmegaConf
from shapely.geometry import Polygon, box
from starVLA.model.modules.structured_world.grid import GridSpec
from starVLA.model.modules.structured_world.semantics import StructuredSemantics, endpoint_events
from starVLA.model.modules.structured_world.fgtr import DDPProposalFGTR, sample_proposal
from starVLA.model.modules.structured_world.future import SingleFrameFuture
from starVLA.model.modules.structured_world.losses import scene_means, final_trajectory_loss
from starVLA.model.modules.action_model.GR00T_ActionHeader import FlowmatchingActionHead
from starVLA.dataloader.structured_world.cameras import projected_depth
from starVLA.dataloader.structured_world.labels import road_distance, annotation_ground_support
from starVLA.dataloader.structured_world.adapters import box_container, pose_matrix
from third_party.uniad.ported.occflow_label import GenerateOccFlowLabels


def small_action_head(steps):
    cfg = OmegaConf.create({'framework': {'qwenvl': {'vl_hidden_dim': 64}, 'action_model': {
        'hidden_size': 64, 'action_dim': 4, 'action_horizon': steps, 'num_inference_timesteps': 10,
        'noise_beta_alpha': 1.5, 'noise_beta_beta': 1., 'noise_s': .999, 'num_timestep_buckets': 1000,
        'add_pos_embed': True, 'max_seq_len': 32,
        'DiTConfig': {'num_layers': 2, 'input_embedding_dim': 64, 'attention_head_dim': 16, 'num_attention_heads': 4},
        'diffusion_model_cfg': {'cross_attention_dim': 64, 'dropout': .2, 'final_dropout': True,
            'interleave_self_attention': True, 'norm_type': 'ada_norm', 'output_dim': 64, 'positional_embeddings': None}}}})
    return FlowmatchingActionHead(cfg)


class Contracts(unittest.TestCase):
    def test_canonical_navsim_yaw_with_pitch_and_roll(self):
        from pyquaternion import Quaternion
        q = Quaternion(axis=[0, 0, 1], angle=.7)*Quaternion(axis=[0, 1, 0], angle=.1)*Quaternion(axis=[1, 0, 0], angle=.07)
        planar = pose_matrix([100., 200., 1.], q.elements, planar=True)
        expected = q.yaw_pitch_roll[0]
        self.assertAlmostEqual(float(np.arctan2(planar[1, 0], planar[0, 0])), expected, places=12)

    def test_rcsample_depth_and_column_centers_and_rectification_support(self):
        from starVLA.model.modules.structured_world.geo_bev import CalibratedRCSample
        g = GridSpec(-2., 2., -1., 1., .5, .5)
        module = CalibratedRCSample(g.resworld_config(), (16, 32), scale_num=1,
            ins_channels=[4], out_channels=4, downsamples=[4], with_cp=False,
            loss_depth_weight=[1.], depthnet_cfg={'use_dcn': False, 'use_aspp': False})
        xyz = torch.tensor([[[[8., 4., 4.]]]])
        sensor = torch.eye(4)[None, None]
        K = torch.eye(3)[None, None]
        post = torch.eye(3)[None, None]
        translation = torch.zeros(1, 1, 3)
        bda = torch.eye(3)[None]
        sampled = module.get_sample_coor(xyz, sensor, sensor, K, post, translation, bda)
        torch.testing.assert_close(sampled, torch.tensor([[[[[1., 3.5]]]]]))
        module.current_pixel_valid = torch.zeros(1, 1, 16, 32, dtype=torch.bool)
        sampled = module.get_sample_coor(xyz, sensor, sensor, K, post, translation, bda)
        self.assertFalse(module.last_projection_support.any())
        self.assertTrue((sampled < -1000).all())

    def test_rectangular_pixel_centers_and_out_of_range(self):
        g = GridSpec(-3., 5., -2., 2., .5, .5)
        xy = g.centers()
        field = (2*xy[..., 0]-3*xy[..., 1])[None, None]
        read, valid = g.sample(field, xy.reshape(1, -1, 2))
        torch.testing.assert_close(read[..., 0], field.flatten(1), atol=2e-6, rtol=1e-6)
        self.assertTrue(valid.all())
        points = torch.tensor([[[-4., 0.], [5., 0.], [0., 3.]]])
        uv, out = g.normalized_reference(points)
        self.assertTrue(out.all())
        self.assertLess(float(uv[0, 0, 0]), 0.)
        self.assertGreater(float(uv[0, 2, 1]), 1.)

    def test_endpoint_identities_and_unknown(self):
        current = torch.tensor([[[0, 1, 1, 255]]], dtype=torch.uint8)
        future = torch.tensor([[[[1, 0, 1, 0]], [[0, 1, 1, 1]]]], dtype=torch.uint8)
        result = endpoint_events(current, future, current != 255, future != 255)
        self.assertEqual(int(result['enter'].sum()), 1)
        self.assertEqual(int(result['release'].sum()), 1)
        self.assertFalse(result['valid'][..., -1].any())
        reconstructed = current[:, None].long()+result['enter'].long()-result['release'].long()
        self.assertTrue(torch.equal(reconstructed[result['valid']], future.long()[result['valid']]))

    def test_predicted_current_event_composition(self):
        model = StructuredSemantics('event', channels=4)
        for p in model.parameters():
            torch.nn.init.zeros_(p)
        model.current[-1].bias.data[1] = torch.logit(torch.tensor(.7))
        model.future[-1].bias.data[:] = torch.logit(torch.tensor([.2, .3]))
        result = model(torch.ones(1, 4, 2, 3), torch.ones(1, 6, 4, 2, 3))
        torch.testing.assert_close(result['pt'], torch.full_like(result['pt'], .55))
        torch.testing.assert_close(result['p_enter'], torch.full_like(result['p_enter'], .06))
        torch.testing.assert_close(result['p_release'], torch.full_like(result['p_release'], .21))

    def test_depth_nearest_front_only_and_no_return(self):
        calibration = {'sensor2ego': torch.eye(4)[None], 'intrinsics': torch.eye(3)[None],
                       'post_rots': torch.eye(3)[None], 'post_trans': torch.zeros(1, 3)}
        cloud = np.array([[2., 2., 2.], [5., 5., 5.], [-1., -1., -1.]])
        depth = projected_depth(cloud, calibration, torch.ones(1, 4, 4, dtype=torch.bool), image_size=(4, 4))
        self.assertEqual(float(depth[0, 1, 1]), 2.)
        self.assertEqual(int((depth > 0).sum()), 1)

    def test_signed_distance_corner_hole_and_rotation(self):
        grid = GridSpec(-4., 4., -4., 4., .5, .5)
        polygon = Polygon([(-3., -3.), (3., -3.), (3., 3.), (-3., 3.)], holes=[box(-1., -1., 1., 1.).exterior.coords])
        D = road_distance(polygon, grid)
        self.assertLess(D[8, 8], 0.)  # interior hole is outside the drivable area
        self.assertGreater(D[8, 12], 0.)
        self.assertLess(D[0, 0], 0.)
        from shapely import distance, points
        xy = grid.centers().numpy()
        exact = distance(polygon.boundary, points(xy))
        self.assertLess(float(np.max(np.abs(np.abs(D)-exact))), .4)

    def test_no_lidar_returns_means_unknown(self):
        grid = GridSpec(-4., 4., -4., 4., .5, .5)
        height, valid, reason = annotation_ground_support(np.zeros((0, 3)), grid)
        self.assertFalse(valid.any()); self.assertTrue(np.isnan(height).all())

    def test_mature_box_ccw_adapter(self):
        boxes = box_container([[3., 2., 1., 4., 2., 2., np.pi/4]])
        corners = boxes.corners[0, [0, 3, 7, 4], :2].numpy()
        # The long edge has positive slope at physical CCW pi/4.
        edges = np.roll(corners, -1, axis=0)-corners
        longest = edges[np.argmax(np.linalg.norm(edges, axis=1))]
        self.assertGreater(longest[0]*longest[1], 0.)
        self.assertAlmostEqual(float(boxes.gravity_center[0, 2]), 1.)

    def test_uniad_clones_static_and_keeps_low_visibility(self):
        gen = GenerateOccFlowLabels({'xbound': [-10., 10., .5], 'ybound': [-10., 10., .5], 'zbound': [-2., 2., 4.]},
                                    only_vehicle=False, filter_invisible=False, compute_flow=False)
        gen.filter_cls_ids = np.arange(10)
        # Ego moved forward two metres; a static world object is two metres
        # farther back in the future ego frame but has identical ego(t0) pixels.
        current = box_container([[5., 0., 1., 4., 2., 2., 0.]])
        future = box_container([[3., 0., 1., 4., 2., 2., 0.]])
        before = current.tensor.clone(), future.tensor.clone()
        results = {'future_gt_bboxes_3d': [current, future, None],
            'future_gt_labels_3d': [np.array([5]), np.array([5]), None],
            'future_gt_inds': [np.array([1]), np.array([1]), None],
            'future_gt_vis_tokens': [np.array([1]), np.array([1]), None],
            'occ_l2e_r_mats': [np.eye(3)]*3, 'occ_l2e_t_vecs': [np.zeros(3)]*3,
            'occ_e2g_r_mats': [np.eye(3)]*3, 'occ_e2g_t_vecs': [np.zeros(3), np.array([2., 0., 0.]), np.zeros(3)],
            'occ_has_invalid_frame': True, 'occ_img_is_valid': np.array([True, True, False])}
        result = gen(results)
        self.assertTrue(torch.equal(result['gt_segmentation'][0], result['gt_segmentation'][1]))
        self.assertGreater(int(result['gt_segmentation'][0].sum()), 0)
        self.assertTrue((result['gt_segmentation'][2] == 255).all())
        self.assertTrue(torch.equal(before[0], current.tensor) and torch.equal(before[1], future.tensor))

    def test_scene_mask_empty_and_full_trajectory_denominator(self):
        value = torch.tensor([[3., 4.], [5., 6.]], requires_grad=True)
        means, counts = scene_means(value, torch.tensor([[True, False], [False, False]]))
        torch.testing.assert_close(means, torch.tensor([3., 0.]))
        means.sum().backward(); self.assertEqual(float(value.grad[1].sum()), 0.)
        proposal = torch.zeros(1, 6, 4, requires_grad=True)
        proposal.data[..., 0] = 100.  # Still receives GT trajectory supervision.
        proposal.data[..., 3] = 1.
        gt = torch.zeros_like(proposal); gt[..., 3] = 1.
        loss, _ = final_trajectory_loss(proposal, gt, torch.ones(1, 6, dtype=torch.bool))
        loss.backward(); self.assertGreater(float(proposal.grad[..., 0].sum()), 0.)

    @unittest.skipUnless(torch.cuda.is_available(), 'Actual CUDA operator required')
    def test_original_proposal_ten_steps_rng_and_horizon(self):
        for steps in (6, 8):
            head = small_action_head(steps).cuda().train()
            # Preserve a deliberately mixed submodule train state.
            head.qwen_proj.eval()
            modes = [m.training for m in head.modules()]
            H_A = torch.randn(2, 5, 64, device='cuda', requires_grad=True)
            noise = torch.randn(2, steps, 4, device='cuda')
            state = torch.cuda.get_rng_state().clone()
            count = []
            hook = head.model.register_forward_hook(lambda *args: count.append(1))
            proposal = sample_proposal(head, H_A, initial_noise=noise)
            hook.remove()
            self.assertEqual(len(count), 10)
            self.assertFalse(proposal.requires_grad)
            self.assertEqual(modes, [m.training for m in head.modules()])
            self.assertTrue(torch.equal(state, torch.cuda.get_rng_state()))
            original_modes = [m.training for m in head.modules()]
            head.eval()
            native = head.predict_action(H_A, initial_noise=noise)
            for m, mode in zip(head.modules(), original_modes): m.training = mode
            torch.testing.assert_close(proposal, native, rtol=0, atol=0)

    @unittest.skipUnless(torch.cuda.is_available(), 'Actual CUDA operator required')
    def test_corresponding_future_and_action_gradient(self):
        grid = GridSpec(-4., 4., -2., 2., .5, .5)
        for steps in (6, 8):
            future = SingleFrameFuture(64, steps, channels=32, grid=grid).cuda().eval()
            refine = DDPProposalFGTR(64, steps, channels=32, grid=grid).cuda().eval()
            B0 = torch.randn(1, 32, grid.height, grid.width, device='cuda', requires_grad=True)
            action = torch.randn(1, 3, 64, device='cuda', requires_grad=True)
            q0 = torch.zeros(1, steps, 3, device='cuda')
            fields = future(B0, action, q0)
            self.assertFalse(torch.allclose(fields[:, 0], fields[:, -1]))
            encoded = torch.zeros(1, steps, 4, device='cuda'); encoded[..., 3] = 1.
            output = refine(action, fields, encoded, q0)
            torch.testing.assert_close(output['q_final_encoded'], encoded, rtol=0, atol=0)
            # Open the zero output layer to test downstream dependencies after
            # its first learning update, as done by the real-scene probe.
            torch.nn.init.normal_(refine.coordinate_decoder[-1].weight, std=.01)
            output = refine(action, fields, encoded, q0)
            output['q_final_encoded'].square().mean().backward()
            self.assertGreater(float(action.grad.norm()), 0.)
            self.assertGreater(float(B0.grad.norm()), 0.)


if __name__ == '__main__':
    unittest.main(verbosity=2)
