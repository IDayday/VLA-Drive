# Apache-2.0. Extracted unmodified semantics from official DDP 8cefcac46e5944add529e1be19cba78bc06cc2bd.
# Loaded via AST with the original framework globals for integration comparisons.
@FRAMEWORK_REGISTRY.register('QwenOFT')
class Qwenvl_OFT(baseframework):

    def forward(self, examples: List[dict]=None, accelerator=None, **kwargs) -> Tuple:
        batch_images = [example['image'] for example in examples]
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
        hist_str = self.robot_history_token
        rgb_str = ''.join(self.rgb_query_tokens)
        gs_str = ''.join(self.gs_query_tokens)
        act_str = ''.join(self.act_query_tokens)
        rew_str = ''.join(self.reward_query_tokens)
        if not self.w_depth:
            suffix = f' {hist_str}{rgb_str}{gs_str}{act_str}{rew_str}'
            instructions = [instruction + suffix for instruction in instructions]
        else:
            suffix = f' {hist_str}{gs_str}{rgb_str}{act_str}{rew_str}'
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
                text_embeds[b, where, :] = rgb_query_reordered
            if self.config.datasets.gs_data.load_3d_data or self.w_depth:
                gs_ids_tensor = torch.tensor(gs_ids, device=input_ids.device)
                where = torch.isin(input_ids[b], gs_ids_tensor).nonzero(as_tuple=False).squeeze(1)
                (_, order) = torch.sort(where)
                gs_query_reordered = self.gs_query[order]
                text_embeds[b, where, :] = gs_query_reordered
        with torch.no_grad():
            (position_ids, _) = self.qwen_vl_interface.model.model.get_rope_index(input_ids=qwen_inputs['input_ids'], image_grid_thw=qwen_inputs['image_grid_thw'], video_grid_thw=qwen_inputs.get('video_grid_thw', None), attention_mask=attention_mask)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            qw_out = self.qwen_vl_interface(inputs_embeds=text_embeds, attention_mask=attention_mask, position_ids=position_ids, pixel_values=qwen_inputs['pixel_values'], image_grid_thw=qwen_inputs['image_grid_thw'], output_hidden_states=True, return_dict=True)
            last_hidden = qw_out.hidden_states[-1]
        if self.config.datasets.video_data.load_2d_data:
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
                (B, V) = data_img.shape[:2]
                data_img = data_img.reshape(B * V, *data_img.shape[2:])
                data_depth = data_depth.reshape(B * V, *data_depth.shape[2:])
                data_mask = data_mask.reshape(B * V, *data_mask.shape[2:])
                depth_data = {'image': data_img, 'depth': data_depth, 'mask': data_mask}
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
                reward_loss = self.reward_model(reward_queries, reward_data)
            return {'action_loss': action_loss, 'rgb_loss': rgb_loss, 'gs_loss': gs_loss, 'reward_loss': reward_loss}
        else:
            reward_loss = torch.tensor(0.0).cuda()
        print(gs_loss)
        return {'action_loss': action_loss, 'rgb_loss': rgb_loss, 'gs_loss': gs_loss * 0.1, 'reward_loss': reward_loss}
