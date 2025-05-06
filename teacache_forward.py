# teacache
import torch
import numpy as np
from typing import Optional, Union, Dict, Any

from matrixgame.model_variants.matrixgame_dit_src.modulate_layers import modulate
from matrixgame.model_variants.matrixgame_dit_src.attenion import attention, get_cu_seqlens
from diffusers.models.modeling_outputs import Transformer2DModelOutput


def teacache_forward(
    self,
    hidden_states: torch.Tensor,
    timestep: torch.Tensor,  # Should be in range(0, 1000).
    encoder_hidden_states: torch.Tensor = None,
    encoder_attention_mask: torch.Tensor = None,  # Now we don't use it.
    guidance: torch.Tensor = None,  # Guidance for modulation, should be cfg_scale x 1000.
    mouse_condition = None,
    keyboard_condition = None,
    return_dict: bool = True,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        x = hidden_states
        t = timestep
        text_states, text_states_2 = encoder_hidden_states
        text_mask, test_mask_2 = encoder_attention_mask
        out = {}
        img = x
        txt = text_states
        _, _, ot, oh, ow = x.shape
        freqs_cos, freqs_sin = self.get_rotary_pos_embed(ot, oh, ow)
        tt, th, tw = (
            ot // self.patch_size[0],
            oh // self.patch_size[1],
            ow // self.patch_size[2],
        )

        # Prepare modulation vectors.
        vec = self.time_in(t)
        if self.i2v_condition_type == "token_replace":
            token_replace_t = torch.zeros_like(t)
            token_replace_vec = self.time_in(token_replace_t)
            frist_frame_token_num = th * tw
        else:
            token_replace_vec = None
            frist_frame_token_num = None
        # text modulation
        #vec_2 = self.vector_in(text_states_2)
        #vec = vec + vec_2
        #if self.i2v_condition_type == "token_replace":
        #    token_replace_vec = token_replace_vec + vec_2

        # guidance modulation
        if self.guidance_embed:
            if guidance is None:
                raise ValueError(
                    "Didn't get guidance strength for guidance distilled model."
                )

            # our timestep_embedding is merged into guidance_in(TimestepEmbedder)
            vec = vec + self.guidance_in(guidance)

        # Embed image and text.
        img = self.img_in(img)
        if self.text_projection == "linear":
            txt = self.txt_in(txt)
        elif self.text_projection == "single_refiner":
            txt = self.txt_in(txt, t, text_mask if self.use_attention_mask else None)
        else:
            raise NotImplementedError(
                f"Unsupported text_projection: {self.text_projection}"
            )

        txt_seq_len = txt.shape[1]
        img_seq_len = img.shape[1]

        # Compute cu_squlens and max_seqlen for flash attention
        cu_seqlens_q = get_cu_seqlens(text_mask, img_seq_len)
        cu_seqlens_kv = cu_seqlens_q
        max_seqlen_q = img_seq_len + txt_seq_len
        max_seqlen_kv = max_seqlen_q

        freqs_cis = (freqs_cos, freqs_sin) if freqs_cos is not None else None

        # teacache
        if self.enable_teacache:
            inp = img.clone()
            vec_ = vec.clone()
            txt_ = txt.clone()
            (
                img_mod1_shift,
                img_mod1_scale,
                img_mod1_gate,
                img_mod2_shift,
                img_mod2_scale,
                img_mod2_gate,
            ) = self.double_blocks[0].img_mod(vec_).chunk(6, dim=-1)
            normed_inp = self.double_blocks[0].img_norm1(inp)
            modulated_inp = modulate(
                normed_inp, shift=img_mod1_shift, scale=img_mod1_scale
            )
            if self.cnt == 0 or self.cnt == self.num_steps-1:
                should_calc = True
                self.accumulated_rel_l1_distance = 0
            else: 
                coefficients = [7.33226126e+02, -4.01131952e+02,  6.75869174e+01, -3.14987800e+00, 9.61237896e-02]
                #coefficients = [-296.53, 191.67, -39.037, 3.705, -0.0383]
                rescale_func = np.poly1d(coefficients)
                self.accumulated_rel_l1_distance += rescale_func(((modulated_inp-self.previous_modulated_input).abs().mean() / self.previous_modulated_input.abs().mean()).cpu().item())
                if self.accumulated_rel_l1_distance < self.rel_l1_thresh:
                    should_calc = False
                else:
                    should_calc = True
                    self.accumulated_rel_l1_distance = 0
            self.previous_modulated_input = modulated_inp  
            self.cnt += 1
            if self.cnt == self.num_steps:
                self.cnt = 0   

        if self.enable_teacache:
            if not should_calc:
                img += self.previous_residual
            else:
                ori_img = img.clone()
                # --------------------- Pass through DiT blocks ------------------------
                for _, block in enumerate(self.double_blocks):
                    if torch.is_grad_enabled() and self.gradient_checkpointing:
                        def create_custom_forward(module):
                            def custom_forward(*inputs):
                                return module(*inputs)

                            return custom_forward
                        ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                        image_kwargs: Dict[str, Any] = {"tt":hidden_states.shape[2] // self.patch_size[0],
                            "th":hidden_states.shape[3] // self.patch_size[1],
                            "tw":hidden_states.shape[4] // self.patch_size[2]}
                        img, txt = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(block),
                            img,
                            txt,
                            vec,
                            cu_seqlens_q,
                            cu_seqlens_kv,
                            max_seqlen_q,
                            max_seqlen_kv,
                            freqs_cis,
                            image_kwargs,
                            mouse_condition,
                            keyboard_condition,
                            self.i2v_condition_type,
                            token_replace_vec,
                            frist_frame_token_num,
                            **ckpt_kwargs,
                        )
                    else:
                        image_kwargs: Dict[str, Any] = {"tt":hidden_states.shape[2] // self.patch_size[0],
                            "th":hidden_states.shape[3] // self.patch_size[1],
                            "tw":hidden_states.shape[4] // self.patch_size[2]}
                        double_block_args = [
                            img,
                            txt,
                            vec,
                            cu_seqlens_q,
                            cu_seqlens_kv,
                            max_seqlen_q,
                            max_seqlen_kv,
                            freqs_cis,
                            image_kwargs,
                            mouse_condition,
                            keyboard_condition,
                            self.i2v_condition_type,
                            token_replace_vec,
                            frist_frame_token_num,
                        ]

                        img, txt = block(*double_block_args)

                # Merge txt and img to pass through single stream blocks.
                x = torch.cat((img, txt), 1)
                if len(self.single_blocks) > 0:
                    for _, block in enumerate(self.single_blocks):
                        if torch.is_grad_enabled() and self.gradient_checkpointing:
                            def create_custom_forward(module):
                                def custom_forward(*inputs):
                                    return module(*inputs)

                                return custom_forward
                            ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                            image_kwargs: Dict[str, Any] = {"tt":hidden_states.shape[2] // self.patch_size[0],
                                "th":hidden_states.shape[3] // self.patch_size[1],
                                "tw":hidden_states.shape[4] // self.patch_size[2]}
                            x = torch.utils.checkpoint.checkpoint(
                                create_custom_forward(block),
                                x,
                                vec,
                                txt_seq_len,
                                cu_seqlens_q,
                                cu_seqlens_kv,
                                max_seqlen_q,
                                max_seqlen_kv,
                                (freqs_cos, freqs_sin),
                                image_kwargs,
                                mouse_condition,
                                keyboard_condition,
                                self.i2v_condition_type,
                                token_replace_vec,
                                frist_frame_token_num,
                                **ckpt_kwargs,
                            )
                        else:
                            image_kwargs: Dict[str, Any] = {"tt":hidden_states.shape[2] // self.patch_size[0],
                                "th":hidden_states.shape[3] // self.patch_size[1],
                                "tw":hidden_states.shape[4] // self.patch_size[2]}
                            single_block_args = [
                                x,
                                vec,
                                txt_seq_len,
                                cu_seqlens_q,
                                cu_seqlens_kv,
                                max_seqlen_q,
                                max_seqlen_kv,
                                (freqs_cos, freqs_sin),
                                image_kwargs,
                                mouse_condition,
                                keyboard_condition,
                                self.i2v_condition_type,
                                token_replace_vec,
                                frist_frame_token_num,
                            ]

                            x = block(*single_block_args)

                img = x[:, :img_seq_len, ...]
                self.previous_residual = img - ori_img
        else:
            # --------------------- Pass through DiT blocks ------------------------
            for _, block in enumerate(self.double_blocks):
                if torch.is_grad_enabled() and self.gradient_checkpointing:
                    def create_custom_forward(module):
                        def custom_forward(*inputs):
                            return module(*inputs)

                        return custom_forward
                    ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                    image_kwargs: Dict[str, Any] = {"tt":hidden_states.shape[2] // self.patch_size[0],
                        "th":hidden_states.shape[3] // self.patch_size[1],
                        "tw":hidden_states.shape[4] // self.patch_size[2]}
                    img, txt = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        img,
                        txt,
                        vec,
                        cu_seqlens_q,
                        cu_seqlens_kv,
                        max_seqlen_q,
                        max_seqlen_kv,
                        freqs_cis,
                        image_kwargs,
                        mouse_condition,
                        keyboard_condition,
                        self.i2v_condition_type,
                        token_replace_vec,
                        frist_frame_token_num,
                        **ckpt_kwargs,
                    )
                else:
                    image_kwargs: Dict[str, Any] = {"tt":hidden_states.shape[2] // self.patch_size[0],
                        "th":hidden_states.shape[3] // self.patch_size[1],
                        "tw":hidden_states.shape[4] // self.patch_size[2]}
                    double_block_args = [
                        img,
                        txt,
                        vec,
                        cu_seqlens_q,
                        cu_seqlens_kv,
                        max_seqlen_q,
                        max_seqlen_kv,
                        freqs_cis,
                        image_kwargs,
                        mouse_condition,
                        keyboard_condition,
                        self.i2v_condition_type,
                        token_replace_vec,
                        frist_frame_token_num,
                    ]

                    img, txt = block(*double_block_args)

            # Merge txt and img to pass through single stream blocks.
            x = torch.cat((img, txt), 1)
            if len(self.single_blocks) > 0:
                for _, block in enumerate(self.single_blocks):
                    if torch.is_grad_enabled() and self.gradient_checkpointing:
                        def create_custom_forward(module):
                            def custom_forward(*inputs):
                                return module(*inputs)

                            return custom_forward
                        ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                        image_kwargs: Dict[str, Any] = {"tt":hidden_states.shape[2] // self.patch_size[0],
                            "th":hidden_states.shape[3] // self.patch_size[1],
                            "tw":hidden_states.shape[4] // self.patch_size[2]}
                        x = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(block),
                            x,
                            vec,
                            txt_seq_len,
                            cu_seqlens_q,
                            cu_seqlens_kv,
                            max_seqlen_q,
                            max_seqlen_kv,
                            (freqs_cos, freqs_sin),
                            image_kwargs,
                            mouse_condition,
                            keyboard_condition,
                            self.i2v_condition_type,
                            token_replace_vec,
                            frist_frame_token_num,
                            **ckpt_kwargs,
                        )
                    else:
                        image_kwargs: Dict[str, Any] = {"tt":hidden_states.shape[2] // self.patch_size[0],
                            "th":hidden_states.shape[3] // self.patch_size[1],
                            "tw":hidden_states.shape[4] // self.patch_size[2]}
                        single_block_args = [
                            x,
                            vec,
                            txt_seq_len,
                            cu_seqlens_q,
                            cu_seqlens_kv,
                            max_seqlen_q,
                            max_seqlen_kv,
                            (freqs_cos, freqs_sin),
                            image_kwargs,
                            mouse_condition,
                            keyboard_condition,
                            self.i2v_condition_type,
                            token_replace_vec,
                            frist_frame_token_num,
                        ]

                        x = block(*single_block_args)

            img = x[:, :img_seq_len, ...]

        # ---------------------------- Final layer ------------------------------
        img = self.final_layer(img, vec)  # (N, T, patch_size ** 2 * out_channels)

        img = self.unpatchify(img, tt, th, tw)
        if return_dict:
            out["x"] = img
            return out
        return (img,)
        