"""Current-only native Qwen with three selected blocks and same-pass merger.

Temporary forward hooks are removed before returning, so activation-checkpoint
recomputation cannot overwrite captures or retain an earlier batch graph.
"""
import copy
import torch
from torch.profiler import record_function
from starVLA.model.modules.qwen_sourceflow.qwen_encoder import QwenEncoder, pack_indices


class SelectedQwenEncoder(QwenEncoder):
    def __init__(self, model_path, *, selected_layers=(8, 16, 28), **kwargs):
        # Transformers.from_dict may pop/update attention keys in its argument.
        # A sealed checkpoint identity must remain immutable after construction.
        if kwargs.get('checkpoint_config') is not None:
            kwargs['checkpoint_config']=copy.deepcopy(kwargs['checkpoint_config'])
        super().__init__(model_path, **kwargs)
        self.selected_layers = tuple(selected_layers)
        if self.selected_layers != (8, 16, 28) or self.num_layers != 28:
            raise ValueError('Main contract reads native Qwen block outputs8/16/28')
        self.last_capture_metadata = None
        self._capturing = False

    def forward(self, batch):
        if self._capturing:
            raise RuntimeError('Concurrent/reentrant observation capture is unsupported')
        self._capturing = True
        self.forward_calls += 1
        device = next(self.parameters()).device
        kwargs = {key: value.to(device) for key, value in batch.items() if key in
                  ('input_ids', 'attention_mask', 'pixel_values', 'image_grid_thw')}
        selected, merged, handles = {}, [], []
        def block_hook(number):
            def capture(module, inputs, output):
                if number in selected:
                    raise RuntimeError('Duplicate selected-block capture inside a single observation forward')
                selected[number] = output[0] if isinstance(output, tuple) else output
            return capture
        def merger_hook(module, inputs, output):
            if merged:
                raise RuntimeError('Unexpected repeated final visual merger')
            merged.append(output)
        try:
            for number in self.selected_layers[:-1]:
                handles.append(self.model.language_model.layers[number-1].register_forward_hook(block_hook(number)))
            handles.append(self.model.visual.merger.register_forward_hook(merger_hook))
            with record_function('world/qwen_current'):
                output = self.model(**kwargs, use_cache=False, output_hidden_states=False, return_dict=True)
            # Official hidden_states[28] is after the final language norm.
            selected[28] = output.last_hidden_state
            if set(selected) != set(self.selected_layers) or len(merged) != 1:
                raise RuntimeError('Missing selected language layer or same-pass merged image features')
        finally:
            for handle in handles:
                handle.remove()
            self._capturing = False
        valid = kwargs['attention_mask'].bool()
        visual = (kwargs['input_ids'] == self.image_token_id) & valid
        vi, vm = pack_indices(visual)
        ti, tm = pack_indices(valid & ~visual)
        positions = torch.zeros((*vm.shape, 3), dtype=torch.long, device=device)
        cameras = torch.full(vm.shape, -1, dtype=torch.long, device=device)
        flat = merged[0]
        if flat.ndim != 2 or flat.shape[-1] != self.hidden_dim or len(flat) != int(vm.sum()):
            raise ValueError('Final visual merger/token layout differs from actual image placeholders')
        packed = flat.new_zeros((len(vm), vm.shape[1], self.hidden_dim))
        grids = kwargs['image_grid_thw'].tolist()
        grid_cursor = visual_cursor = 0
        for b, views in enumerate(batch['camera_indices']):
            scene_grids = grids[grid_cursor:grid_cursor+len(views)]
            if len(scene_grids) != len(views):
                raise ValueError('Image grids cross batch boundaries')
            tile = max(grid[2]//self.merge_size for grid in scene_grids)
            offset = 0
            for camera, (t, h, w) in zip(views, scene_grids):
                if t != 1 or h % self.merge_size or w % self.merge_size:
                    raise ValueError('Only native current-image merged grids are valid')
                mh, mw = h//self.merge_size, w//self.merge_size
                n = mh*mw
                positions[b, offset:offset+n, 1] = torch.arange(mh, device=device).repeat_interleave(mw)
                positions[b, offset:offset+n, 2] = torch.arange(mw, device=device).repeat(mh)+camera*tile
                cameras[b, offset:offset+n] = camera
                packed[b, offset:offset+n] = flat[visual_cursor:visual_cursor+n]
                visual_cursor += n
                offset += n
            if offset != int(vm[b].sum()):
                raise ValueError('Current camera/grid order differs from token placeholders')
            grid_cursor += len(views)
        if grid_cursor != len(grids) or visual_cursor != len(flat):
            raise ValueError('Unconsumed current visual layout')
        self.last_capture_metadata = dict(selected_layers=self.selected_layers,
            final_layer_after_norm=True, full_hidden_tuple_retained=False,
            hooks_removed=True, sequence_lengths=valid.sum(1).detach().cpu().tolist(),
            visual_tokens=vm.sum(1).detach().cpu().tolist(), merged_dtype=str(flat.dtype))
        return dict(selected_hidden_states=selected, final_hidden=output.last_hidden_state,
            visual_indices=vi, visual_mask=vm, visual_positions=positions, camera_ids=cameras,
            text_indices=ti, text_mask=tm, merged_visual_features=packed)
