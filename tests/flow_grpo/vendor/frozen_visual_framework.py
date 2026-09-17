# Extracted verbatim AST methods from IDayday/VLA-Drive f9449d55bea6895a7a0bd86d09d7ab85fd353f26.
# Source: checkpoint_code/action-only-checkpoints-v1/frozen-visual-overlay/starVLA/model/framework/QwenOFT.py
# Original source header: Copyright 2025 starVLA community; MIT License.
# Test oracle only; methods are compiled with the original module globals.
class Qwenvl_OFT:

    @staticmethod
    def _find_token_positions(input_ids, token_ids):
        """Find ordered special-token positions without Python/CUDA scalar syncs."""
        ids = torch.as_tensor(token_ids, device=input_ids.device, dtype=input_ids.dtype)
        matches = input_ids.unsqueeze(-1).eq(ids.view(1, 1, -1))
        return matches.to(torch.int8).argmax(dim=1)

    def _build_action_prompt_suffix(self) -> str:
        """Build the instruction suffix used by action-only prompting."""
        hist_str = self.robot_history_token
        act_str = ''.join(self.act_query_tokens)
        if self.action_prompt_mode == 'minimal':
            return f' {hist_str}{act_str}'
        rgb_str = ''.join(self.rgb_query_tokens)
        gs_str = ''.join(self.gs_query_tokens)
        rew_str = ''.join(self.reward_query_tokens)
        if self.w_depth:
            return f' {hist_str}{gs_str}{rgb_str}{act_str}{rew_str}'
        return f' {hist_str}{rgb_str}{gs_str}{act_str}{rew_str}'

    def _build_qwen_batch(self, examples, instructions):
        """Build either cached or ordinary Qwen inputs for one training batch."""
        cached = [example.get('qwen_feature_cache') for example in examples]
        if any((payload is not None for payload in cached)) and (not all((payload is not None for payload in cached))):
            raise RuntimeError('A batch cannot mix cached and uncached Qwen samples')
        device = self.qwen_vl_interface.model.device
        if all((payload is not None for payload in cached)):
            lengths = [int(payload['input_ids'].numel()) for payload in cached]
            max_length = max(lengths)
            batch_size = len(cached)
            tokenizer = self.qwen_vl_interface.processor.tokenizer
            input_ids = torch.full((batch_size, max_length), int(tokenizer.pad_token_id), dtype=torch.long, device=device)
            attention_mask = torch.zeros((batch_size, max_length), dtype=torch.long, device=device)
            position_ids = torch.ones((3, batch_size, max_length), dtype=torch.long, device=device)
            position_names = tuple(self._special_token_ids.keys())
            positions = {name: [] for name in position_names}
            for (batch_index, (payload, length)) in enumerate(zip(cached, lengths)):
                offset = max_length - length
                input_ids[batch_index, offset:] = payload['input_ids'].to(device=device, dtype=torch.long)
                attention_mask[batch_index, offset:] = payload['attention_mask'].to(device=device, dtype=torch.long)
                position_ids[:, batch_index, offset:] = payload['position_ids'].to(device=device, dtype=torch.long)
                for name in position_names:
                    positions[name].append(payload[f'{name}_positions'].to(device=device, dtype=torch.long) + offset)
            deepstack_keys = sorted((key for key in cached[0] if key.startswith('deepstack_')))
            image_embeds = torch.cat([payload['image_embeds'] for payload in cached], dim=0)
            deepstack_embeds = [torch.cat([payload[key] for payload in cached], dim=0) for key in deepstack_keys]
            return (input_ids, attention_mask, position_ids, {name: torch.stack(values) for (name, values) in positions.items()}, image_embeds.to(device, non_blocking=True), [value.to(device, non_blocking=True) for value in deepstack_embeds])
        batch_images = [example['image'] for example in examples]
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(images=batch_images, instructions=instructions)
        input_ids = qwen_inputs['input_ids']
        attention_mask = qwen_inputs['attention_mask']
        with torch.no_grad():
            (position_ids, _) = self.qwen_vl_interface.model.model.get_rope_index(input_ids=input_ids, image_grid_thw=qwen_inputs['image_grid_thw'], video_grid_thw=qwen_inputs.get('video_grid_thw', None), attention_mask=attention_mask)
            (image_parts, deepstack_embeds) = self.qwen_vl_interface.model.model.get_image_features(qwen_inputs['pixel_values'], qwen_inputs['image_grid_thw'])
            image_embeds = torch.cat(image_parts, dim=0)
        positions = {name: self._find_token_positions(input_ids, token_ids) for (name, token_ids) in self._special_token_ids.items()}
        return (input_ids, attention_mask, position_ids, positions, image_embeds, deepstack_embeds)

    def _qwen_language_forward(self, input_ids, inputs_embeds, attention_mask, position_ids, image_embeds, deepstack_embeds):
        """Run only the trainable Qwen backbone, skipping the unused LM head."""
        image_mask = input_ids.eq(self.qwen_vl_interface.model.config.image_token_id)
        expanded_mask = image_mask.unsqueeze(-1).expand_as(inputs_embeds)
        image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.masked_scatter(expanded_mask, image_embeds)
        outputs = self.qwen_vl_interface.model.model.language_model(input_ids=None, inputs_embeds=inputs_embeds, attention_mask=attention_mask, position_ids=position_ids, visual_pos_masks=image_mask, deepstack_visual_embeds=[value.to(inputs_embeds.device, inputs_embeds.dtype) for value in deepstack_embeds], use_cache=False)
        return outputs.last_hidden_state

    def forward(self, examples: List[dict]=None, accelerator=None, **kwargs) -> Tuple:
        instructions = [example['lang'] for example in examples]
        try:
            actions = [example['action'] for example in examples]
        except:
            actions = None
        try:
            states = [example['state'] for example in examples]
        except:
            states = None
        if self.w_depth:
            depth_data = [example['depth_data'] for example in examples]
        suffix = self._build_action_prompt_suffix()
        instructions = [instruction + suffix for instruction in instructions]
        (input_ids, attention_mask, position_ids, token_positions, image_embeds, deepstack_embeds) = self._build_qwen_batch(examples, instructions)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            text_embeds = self.qwen_vl_interface.model.get_input_embeddings()(input_ids)
        with torch.autocast('cuda', dtype=torch.float32):
            states = torch.as_tensor(np.asarray(states), device=text_embeds.device)[:, 0, :]
            states_embed = self.action_input_model(states)
        states_embed = states_embed.to(dtype=text_embeds.dtype)
        (B, L, H) = text_embeds.shape
        batch_indices = torch.arange(B, device=text_embeds.device)
        text_embeds[batch_indices, token_positions['history'][:, 0], :] = states_embed
        if self.config.datasets.video_data.load_2d_data:
            text_embeds[batch_indices[:, None], token_positions['rgb'], :] = self.rgb_query.unsqueeze(0).to(text_embeds.dtype)
        if self.config.datasets.gs_data.load_3d_data or self.w_depth:
            text_embeds[batch_indices[:, None], token_positions['gs'], :] = self.gs_query.unsqueeze(0).to(text_embeds.dtype)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            last_hidden = self._qwen_language_forward(input_ids=input_ids, inputs_embeds=text_embeds, attention_mask=attention_mask, position_ids=position_ids, image_embeds=image_embeds, deepstack_embeds=deepstack_embeds)
        if self.config.datasets.video_data.load_2d_data:
            rgb_data = [example['2d_gen_data'] for example in examples]
            g_idx = token_positions['rgb'].unsqueeze(-1).expand(-1, -1, H)
            rgb_queries = last_hidden.gather(dim=1, index=g_idx)
            if self.rgb_query_loss:
                m = self.traj_emb(rgb_queries, self.traj_emb_h0.unsqueeze(0).repeat(1, rgb_queries.shape[0], 1))[1].squeeze(1)
                actions = torch.tensor(np.array(actions), device=rgb_queries.device, dtype=torch.float32)
                rgb_query_loss = nn.L1Loss()(self.rgb_act_pre(m).reshape(B, 8, 4), actions)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                (rgb_loss, video_latent) = self.rgb_model(rgb_data, rgb_queries)
            if self.rgb_query_loss:
                rgb_loss += rgb_query_loss
        else:
            rgb_loss = torch.tensor(0.0).cuda()
        if self.config.datasets.vla_data.load_act_data == 1:
            g_idx = token_positions['action'].unsqueeze(-1).expand(-1, -1, H)
            action_queries = last_hidden.gather(dim=1, index=g_idx)
            with torch.autocast('cuda', dtype=torch.float32):
                if type(actions) == list:
                    actions = torch.tensor(np.array(actions), device=action_queries.device, dtype=torch.float32)
                repeated_diffusion_steps = self.config.framework.action_model.get('repeated_diffusion_steps', 1) if self.config else 1
                repeat_actions = actions.repeat(repeated_diffusion_steps, 1, 1)
                repeat_action_queries = action_queries.repeat(repeated_diffusion_steps, 1, 1)
                if self.w_video_latent:
                    video_token = self.rgb_latent_adapter(video_latent)
                    video_token = video_token + self.rgb_latent_type.to(video_token.dtype)
                else:
                    video_token = None
                if self.mlp_head == 0:
                    action_loss = self.action_model(repeat_action_queries, repeat_actions, video_token)
                else:
                    (b, l, h) = action_queries.shape
                    pred_action = self.action_model(action_queries.reshape(b, l * h)).reshape(b, l, -1)
                    action_loss = nn.SmoothL1Loss()(pred_action, actions)
        else:
            action_loss = torch.tensor(0.0).cuda()
        if self.config.datasets.gs_data.load_3d_data or self.w_depth:
            if self.config.datasets.gs_data.load_3d_data:
                gs_data = [example['3d_gs_data'] for example in examples]
            else:
                data_img = torch.stack([d['image'] for d in depth_data])
                data_depth = torch.stack([d['depth'] for d in depth_data])
                data_mask = torch.stack([d['mask'] for d in depth_data])
                cached_semantics = None
                if all(('semantics' in value for value in depth_data)):
                    cached_semantics = torch.stack([value['semantics'] for value in depth_data])
                (B, V) = data_img.shape[:2]
                data_img = data_img.reshape(B * V, *data_img.shape[2:])
                data_depth = data_depth.reshape(B * V, *data_depth.shape[2:])
                data_mask = data_mask.reshape(B * V, *data_mask.shape[2:])
                depth_data = {'image': data_img, 'depth': data_depth, 'mask': data_mask}
                if cached_semantics is not None:
                    depth_data['semantics'] = cached_semantics.reshape(B * V, *cached_semantics.shape[2:])
            g_idx = token_positions['gs'].unsqueeze(-1).expand(-1, -1, H)
            gs_queries = last_hidden.gather(dim=1, index=g_idx)
            if self.config.datasets.gs_data.load_3d_data:
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    gs_loss = self.gs_model(gs_data, gs_queries)
            else:
                depth_data['qwen_token'] = gs_queries.repeat_interleave(3, dim=0)
                if self.gs_query_loss:
                    m = self.gs_traj_emb(gs_queries, self.gs_traj_emb_h0.unsqueeze(0).repeat(1, gs_queries.shape[0], 1))[1].squeeze(0)
                    if type(actions) == list:
                        actions = torch.tensor(np.array(actions), device=gs_queries.device, dtype=torch.float32)
                    gs_query_loss = nn.L1Loss()(self.gs_act_pre(m).reshape(B, 8, 4), actions)
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    output = self.gs_model.forward_train(depth_data)
                    gs_loss = output['loss']
                    if self.gs_query_loss:
                        gs_loss += gs_query_loss
        else:
            gs_loss = torch.tensor(0.0).cuda()
        if self.config.datasets.reward_data.load_reward_data:
            reward_data = np.array([example['reward_data'] for example in examples])
            g_idx = token_positions['reward'].unsqueeze(-1).expand(-1, -1, H)
            reward_queries = last_hidden.gather(dim=1, index=g_idx)
            with torch.autocast('cuda', dtype=torch.float32):
                reward_loss = self.reward_model(reward_queries, reward_data)
            return {'action_loss': action_loss, 'rgb_loss': rgb_loss, 'gs_loss': gs_loss, 'reward_loss': reward_loss}
        else:
            reward_loss = torch.tensor(0.0).cuda()
        return {'action_loss': action_loss, 'rgb_loss': rgb_loss, 'gs_loss': gs_loss * 0.1, 'reward_loss': reward_loss}

    @torch.inference_mode()
    def predict_action_infer_1d(self, examples, **kwargs: str) -> np.ndarray:
        """
        推理：单次前向直接回归未来动作（无扩散采样）。

        Steps:
          1. Resize images to training resolution (if specified)
          2. Encode with QwenVL (hidden states retained)
          6. Return normalized action trajectory

        Args:
            batch_images: List of samples; each sample is List[PIL.Image] (multi-view).
            instructions: List[str] natural language task instructions.
            cfg_scale: >1 enables classifier-free guidance (scales conditional vs unconditional).
            use_ddim: Whether to use DDIM deterministic sampling.
            num_ddim_steps: Number of DDIM steps if enabled.
            **kwargs: Reserved.

        Returns:
            dict:
                normalized_actions (np.ndarray): Shape [B, T, action_dim], diffusion-sampled normalized actions.
        """
        batch_images = [example['image'] for example in examples]
        instructions = [example['lang'] for example in examples]
        states = [example['state'] for example in examples]
        if self.w_depth:
            pass
        suffix = self._build_action_prompt_suffix()
        instructions = [instruction + suffix for instruction in instructions]
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(images=batch_images, instructions=instructions)
        tok = self.qwen_vl_interface.processor.tokenizer
        hist_id = tok.convert_tokens_to_ids(self.robot_history_token)
        if self.config.datasets.video_data.load_2d_data:
            rgb_ids = tok.convert_tokens_to_ids(self.rgb_query_tokens)
        if self.config.datasets.gs_data.load_3d_data or self.w_depth:
            gs_ids = tok.convert_tokens_to_ids(self.gs_query_tokens)
        if self.config.datasets.reward_data.load_reward_data:
            reward_ids = tok.convert_tokens_to_ids(self.reward_query_tokens)
        input_ids = qwen_inputs['input_ids']
        attention_mask = qwen_inputs['attention_mask']
        with torch.autocast('cuda', dtype=torch.bfloat16):
            text_embeds = self.qwen_vl_interface.model.get_input_embeddings()(input_ids)
        with torch.autocast('cuda', dtype=torch.float32):
            states = torch.from_numpy(np.array(states)).cuda()[:, 0, :]
            states_embed = self.action_input_model(states)
        states_embed = states_embed.to(dtype=text_embeds.dtype)
        (B, L, H) = text_embeds.shape
        for b in range(B):
            where = (input_ids[b] == hist_id).nonzero(as_tuple=False)
            if where.numel() == 0:
                raise RuntimeError(f'Sample {b}: robot_history token not found in input_ids.')
            if where.numel() > 1:
                pass
            pos = int(where[0])
            text_embeds[b, pos, :] = states_embed[b]
            if self.config.datasets.video_data.load_2d_data:
                rgb_ids_tensor = torch.tensor(rgb_ids, device=input_ids.device)
                where = torch.isin(input_ids[b], rgb_ids_tensor).nonzero(as_tuple=False).squeeze(1)
                (_, order) = torch.sort(where)
                rgb_query_reordered = self.rgb_query[order]
                text_embeds[b, where, :] = rgb_query_reordered.to(text_embeds.dtype)
            if self.config.datasets.gs_data.load_3d_data or self.w_depth:
                gs_ids_tensor = torch.tensor(gs_ids, device=input_ids.device)
                where = torch.isin(input_ids[b], gs_ids_tensor).nonzero(as_tuple=False).squeeze(1)
                (_, order) = torch.sort(where)
                gs_query_reordered = self.gs_query[order]
                text_embeds[b, where, :] = gs_query_reordered.to(text_embeds.dtype)
        with torch.no_grad():
            (position_ids, _) = self.qwen_vl_interface.model.model.get_rope_index(input_ids=qwen_inputs['input_ids'], image_grid_thw=qwen_inputs['image_grid_thw'], video_grid_thw=qwen_inputs.get('video_grid_thw', None), attention_mask=attention_mask)
        qwen_forward_mode = getattr(self, '_inference_qwen_forward_mode', 'legacy')
        with torch.autocast('cuda', dtype=torch.bfloat16):
            if qwen_forward_mode == 'optimized':
                (image_parts, deepstack_embeds) = self.qwen_vl_interface.model.model.get_image_features(qwen_inputs['pixel_values'], qwen_inputs['image_grid_thw'])
                image_embeds = torch.cat(image_parts, dim=0)
                last_hidden = self._qwen_language_forward(input_ids=input_ids, inputs_embeds=text_embeds, attention_mask=attention_mask, position_ids=position_ids, image_embeds=image_embeds, deepstack_embeds=deepstack_embeds)
            else:
                qw_out = self.qwen_vl_interface(inputs_embeds=text_embeds, attention_mask=attention_mask, position_ids=position_ids, pixel_values=qwen_inputs.get('pixel_values', None), image_grid_thw=qwen_inputs.get('image_grid_thw', None), output_hidden_states=True, return_dict=True)
                last_hidden = qw_out.hidden_states[-1]
        act_ids = [tok.convert_tokens_to_ids(t) for t in self.act_query_tokens]
        act_pos_idx = []
        for b in range(B):
            pos_list = []
            for tid in act_ids:
                w = (input_ids[b] == tid).nonzero(as_tuple=False)
                if w.numel() == 0:
                    raise RuntimeError(f'Sample {b}: action token {tid} not found.')
                pos_list.append(int(w[0]))
            act_pos_idx.append(torch.tensor(pos_list, device=last_hidden.device))
        act_pos_idx = torch.stack(act_pos_idx, dim=0)
        g_idx = act_pos_idx.unsqueeze(-1).expand(-1, -1, H)
        action_queries = last_hidden.gather(dim=1, index=g_idx)
        with torch.autocast('cuda', dtype=torch.float32):
            if self.mlp_head == 0:
                pred_actions = self.action_model.predict_action(action_queries)
            else:
                pred_actions = self.action_model(action_queries)
        normalized_actions = pred_actions.detach().cpu().numpy()
        if False:
            rgb_data = [example['2d_gen_data'] for example in examples]
            rgb_ids = [tok.convert_tokens_to_ids(t) for t in self.rgb_query_tokens]
            rgb_pos_idx = []
            for b in range(B):
                pos_list = []
                for tid in rgb_ids:
                    w = (input_ids[b] == tid).nonzero(as_tuple=False)
                    if w.numel() == 0:
                        raise RuntimeError(f'Sample {b}: action token {tid} not found.')
                    pos_list.append(int(w[0]))
                rgb_pos_idx.append(torch.tensor(pos_list, device=last_hidden.device))
            rgb_pos_idx = torch.stack(rgb_pos_idx, dim=0)
            g_idx = rgb_pos_idx.unsqueeze(-1).expand(-1, -1, H)
            rgb_queries = last_hidden.gather(dim=1, index=g_idx)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                rgbs = self.rgb_model.predict_rgb(rgb_data, rgb_queries)
            return {'normalized_actions': normalized_actions, 'rgbs': rgbs}
        if False:
            gs_data = [example['3d_gs_data'] for example in examples]
            gs_ids = [tok.convert_tokens_to_ids(t) for t in self.gs_query_tokens]
            gs_pos_idx = []
            for b in range(B):
                pos_list = []
                for tid in gs_ids:
                    w = (input_ids[b] == tid).nonzero(as_tuple=False)
                    if w.numel() == 0:
                        raise RuntimeError(f'Sample {b}: action token {tid} not found.')
                    pos_list.append(int(w[0]))
                gs_pos_idx.append(torch.tensor(pos_list, device=last_hidden.device))
            gs_pos_idx = torch.stack(gs_pos_idx, dim=0)
            g_idx = gs_pos_idx.unsqueeze(-1).expand(-1, -1, H)
            gs_queries = last_hidden.gather(dim=1, index=g_idx)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                gs = self.gs_model.predict_gs(gs_data, gs_queries)
            return {'normalized_actions': normalized_actions, 'gs': gs}
        if False:
            reward_ids = [tok.convert_tokens_to_ids(t) for t in self.reward_query_tokens]
            reward_pos_idx = []
            for b in range(B):
                pos_list = []
                for tid in reward_ids:
                    w = (input_ids[b] == tid).nonzero(as_tuple=False)
                    if w.numel() == 0:
                        raise RuntimeError(f'Sample {b}: action token {tid} not found.')
                    pos_list.append(int(w[0]))
                reward_pos_idx.append(torch.tensor(pos_list, device=last_hidden.device))
            reward_pos_idx = torch.stack(reward_pos_idx, dim=0)
            g_idx = reward_pos_idx.unsqueeze(-1).expand(-1, -1, H)
            reward_queries = last_hidden.gather(dim=1, index=g_idx)
            with torch.autocast('cuda', dtype=torch.float32):
                reward = self.reward_model.predict_action(reward_queries)
            return {'normalized_actions': normalized_actions, 'reward': reward}
        return {'normalized_actions': normalized_actions}
