from dataclasses import dataclass
from typing import Optional, Tuple
from copy import deepcopy

import torch
import torch.nn as nn
from transformers import CLIPTextModel, CLIPTokenizer, AutoTokenizer, AutoModel, LlavaForConditionalGeneration, CLIPImageProcessor
from transformers.utils import ModelOutput


def use_default(value, default):
    return value if value is not None else default


def load_text_encoder(
    text_encoder_type,
    text_encoder_precision=None,
    text_encoder_path=None,
    logger=None,
    device=None,
):

    if text_encoder_type == 'clipL':
        text_encoder_path = f"{text_encoder_path}/text_encoder_2"
    elif text_encoder_type == 'llm':
        text_encoder_path = f"{text_encoder_path}/text_encoder"
    elif text_encoder_type == 'llm-i2v':
        text_encoder_path = f"{text_encoder_path}/text_encoder_i2v"
    if logger is not None:
        logger.info(
            f"Loading text encoder model ({text_encoder_type}) from: {text_encoder_path}"
        )

    if text_encoder_type == "clipL":
        text_encoder = CLIPTextModel.from_pretrained(text_encoder_path, low_cpu_mem_usage=True)
        text_encoder.final_layer_norm = text_encoder.text_model.final_layer_norm
    elif text_encoder_type == "llm":
        text_encoder = AutoModel.from_pretrained(
            text_encoder_path, low_cpu_mem_usage=True
        )
        text_encoder.final_layer_norm = text_encoder.norm
    elif text_encoder_type == "llm-i2v":
        text_encoder = LlavaForConditionalGeneration.from_pretrained(
            text_encoder_path, low_cpu_mem_usage=True
        )
    else:
        raise ValueError(f"Unsupported text encoder type: {text_encoder_type}")
    # from_pretrained will ensure that the model is in eval mode.

    text_encoder = text_encoder.to(dtype=text_encoder_precision)
    try:
        if torch.distributed.get_rank() == 0:
            text_encoder_dict = text_encoder.state_dict()
            params = [text_encoder_dict]
        else:
            text_encoder_dict = None
            params = [None]
        torch.distributed.barrier()
        torch.distributed.broadcast_object_list(params, 0)
        if torch.distributed.get_rank() != 0:
            text_encoder.load_state_dict(params[0])
    except:
        pass

    text_encoder.requires_grad_(False)
    text_encoder.eval()
    if logger is not None:
        logger.info(f"Text encoder to dtype: {text_encoder.dtype}")

    if device is not None:
        text_encoder = text_encoder.to(device)

    return text_encoder, text_encoder_path


def load_tokenizer(
    tokenizer_type, tokenizer_path=None, padding_side="right", logger=None
):
    if tokenizer_type == 'clipL':
        tokenizer_path = f"{tokenizer_path}/text_encoder_2"
    elif tokenizer_type == 'llm-i2v':
        tokenizer_path = f"{tokenizer_path}/text_encoder_i2v"

    if logger is not None:
        logger.info(f"Loading tokenizer ({tokenizer_type}) from: {tokenizer_path}")
    
    processor = None
    if tokenizer_type == "clipL":
        tokenizer = CLIPTokenizer.from_pretrained(tokenizer_path, max_length=77)
    elif tokenizer_type == "llm":
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path, padding_side=padding_side
        )
    elif tokenizer_type == "llm-i2v":
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path, padding_side=padding_side
        )
        processor = CLIPImageProcessor.from_pretrained(tokenizer_path)
    else:
        raise ValueError(f"Unsupported tokenizer type: {tokenizer_type}")

    return tokenizer, tokenizer_path, processor


@dataclass
class TextEncoderModelOutput(ModelOutput):
    """
    Base class for model's outputs that also contains a pooling of the last hidden states.

    Args:
        hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        attention_mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:
        hidden_states_list (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        text_outputs (`list`, *optional*, returned when `return_texts=True` is passed):
            List of decoded texts.
    """

    hidden_state: torch.FloatTensor = None
    attention_mask: Optional[torch.LongTensor] = None
    hidden_states_list: Optional[Tuple[torch.FloatTensor, ...]] = None
    text_outputs: Optional[list] = None


class TextEncoder(nn.Module):
    def __init__(
        self,
        text_encoder_type: str,
        max_length: int,
        text_encoder_precision: Optional[str] = None,
        text_encoder_path: Optional[str] = None,
        tokenizer_type: Optional[str] = None,
        tokenizer_path: Optional[str] = None,
        output_key: Optional[str] = None,
        use_attention_mask: bool = True,
        i2v_mode: bool = False,
        input_max_length: Optional[int] = None,
        prompt_template: Optional[dict] = None,
        prompt_template_video: Optional[dict] = None,
        hidden_state_skip_layer: Optional[int] = None,
        apply_final_norm: bool = False,
        reproduce: bool = False,
        logger=None,
        device=None,
        image_embed_interleave=None,
    ):
        super().__init__()
        self.text_encoder_type = text_encoder_type
        self.max_length = max_length
        self.precision = text_encoder_precision
        self.model_path = text_encoder_path
        self.tokenizer_type = (
            tokenizer_type if tokenizer_type is not None else text_encoder_type
        )
        self.tokenizer_path = (
            tokenizer_path if tokenizer_path is not None else text_encoder_path
        )
        self.use_attention_mask = use_attention_mask
        if prompt_template_video is not None:
            assert (
                use_attention_mask is True
            ), "Attention mask is True required when training videos."
        self.input_max_length = (
            input_max_length if input_max_length is not None else max_length
        )
        self.prompt_template = prompt_template
        self.prompt_template_video = prompt_template_video
        self.hidden_state_skip_layer = hidden_state_skip_layer
        self.apply_final_norm = apply_final_norm
        self.i2v_mode = i2v_mode
        self.reproduce = reproduce
        self.logger = logger
        self.image_embed_interleave = image_embed_interleave

        self.use_template = self.prompt_template is not None
        if self.use_template:
            assert (
                isinstance(self.prompt_template, dict)
                and "template" in self.prompt_template
            ), f"`prompt_template` must be a dictionary with a key 'template', got {self.prompt_template}"
            assert "{}" in str(self.prompt_template["template"]), (
                "`prompt_template['template']` must contain a placeholder `{}` for the input text, "
                f"got {self.prompt_template['template']}"
            )

        self.use_video_template = self.prompt_template_video is not None
        if self.use_video_template:
            if self.prompt_template_video is not None:
                assert (
                    isinstance(self.prompt_template_video, dict)
                    and "template" in self.prompt_template_video
                ), f"`prompt_template_video` must be a dictionary with a key 'template', got {self.prompt_template_video}"
            assert "{}" in str(self.prompt_template_video["template"]), (
                "`prompt_template_video['template']` must contain a placeholder `{}` for the input text, "
                f"got {self.prompt_template_video['template']}"
            )

        if "t5" in text_encoder_type:
            self.output_key = output_key or "last_hidden_state"
        elif "clip" in text_encoder_type:
            self.output_key = output_key or "pooler_output"
        elif "llm" in text_encoder_type or "glm" in text_encoder_type:
            self.output_key = output_key or "last_hidden_state"
        else:
            raise ValueError(f"Unsupported text encoder type: {text_encoder_type}")

        self.model, self.model_path = load_text_encoder(
            text_encoder_type=self.text_encoder_type,
            text_encoder_precision=self.precision,
            text_encoder_path=self.model_path,
            logger=self.logger,
            device=device,
        )
        self.dtype = self.model.dtype
        self.device = self.model.device

        self.tokenizer, self.tokenizer_path, self.processor = load_tokenizer(
            tokenizer_type=self.tokenizer_type,
            tokenizer_path=self.tokenizer_path,
            padding_side="right",
            logger=self.logger,
        )

    def __repr__(self):
        return f"{self.text_encoder_type} ({self.precision} - {self.model_path})"

    @staticmethod
    def apply_text_to_template(text, template, prevent_empty_text=True):
        """
        Apply text to template.

        Args:
            text (str): Input text.
            template (str or list): Template string or list of chat conversation.
            prevent_empty_text (bool): If Ture, we will prevent the user text from being empty
                by adding a space. Defaults to True.
        """
        if isinstance(template, str):
            # Will send string to tokenizer. Used for llm
            return template.format(text)
        else:
            raise TypeError(f"Unsupported template type: {type(template)}")

    def text2tokens(self, text, data_type="image"):
        """
        Tokenize the input text.

        Args:
            text (str or list): Input text.
        """
        tokenize_input_type = "str"
        if self.use_template:
            if data_type == "image":
                prompt_template = self.prompt_template["template"]
            elif data_type == "video":
                prompt_template = self.prompt_template_video["template"]
            else:
                raise ValueError(f"Unsupported data type: {data_type}")
            if isinstance(text, (list, tuple)):
                text = [
                    self.apply_text_to_template(one_text, prompt_template)
                    for one_text in text
                ]
                if isinstance(text[0], list):
                    tokenize_input_type = "list"
            elif isinstance(text, str):
                text = self.apply_text_to_template(text, prompt_template)
                if isinstance(text, list):
                    tokenize_input_type = "list"
            else:
                raise TypeError(f"Unsupported text type: {type(text)}")

        kwargs = dict(
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        if tokenize_input_type == "str":
            return self.tokenizer(
                text,
                return_length=False,
                return_overflowing_tokens=False,
                return_attention_mask=True,
                **kwargs,
            )
        elif tokenize_input_type == "list":
            return self.tokenizer.apply_chat_template(
                text,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                **kwargs,
            )
        else:
            raise ValueError(f"Unsupported tokenize_input_type: {tokenize_input_type}")

    def encode(
        self,
        batch_encoding,
        use_attention_mask=None,
        output_hidden_states=False,
        do_sample=None,
        hidden_state_skip_layer=None,
        return_texts=False,
        data_type="image",
        semantic_images=None,
        device=None,
    ):
        """
        Args:
            batch_encoding (dict): Batch encoding from tokenizer.
            use_attention_mask (bool): Whether to use attention mask. If None, use self.use_attention_mask.
                Defaults to None.
            output_hidden_states (bool): Whether to output hidden states. If False, return the value of
                self.output_key. If True, return the entire output. If set self.hidden_state_skip_layer,
                output_hidden_states will be set True. Defaults to False.
            do_sample (bool): Whether to sample from the model. Used for Decoder-Only LLMs. Defaults to None.
                When self.produce is False, do_sample is set to True by default.
            hidden_state_skip_layer (int): Number of hidden states to hidden_state_skip_layer. 0 means the last layer.
                If None, self.output_key will be used. Defaults to None.
            hidden_state_skip_layer (PIL.Image): The reference images for i2v models.
            return_texts (bool): Whether to return the decoded texts. Defaults to False.
        """
        device = self.model.device if device is None else device
        use_attention_mask = use_default(use_attention_mask, self.use_attention_mask)
        hidden_state_skip_layer = use_default(
            hidden_state_skip_layer, self.hidden_state_skip_layer
        )
        do_sample = use_default(do_sample, not self.reproduce)
        if not self.i2v_mode:
            attention_mask = (
                batch_encoding["attention_mask"].to(device)
                if use_attention_mask
                else None
            )
            outputs = self.model(
                input_ids=batch_encoding["input_ids"].to(device),
                attention_mask=attention_mask,
                output_hidden_states=output_hidden_states
                or hidden_state_skip_layer is not None,
            )
            if hidden_state_skip_layer is not None:
                last_hidden_state = outputs.hidden_states[
                    -(hidden_state_skip_layer + 1)
                ]
                # Real last hidden state already has layer norm applied. So here we only apply it
                # for intermediate layers.
                if hidden_state_skip_layer > 0 and self.apply_final_norm:
                    last_hidden_state = self.model.final_layer_norm(last_hidden_state)
            else:
                last_hidden_state = outputs[self.output_key]

            # Remove hidden states of instruction tokens, only keep prompt tokens.
            if self.use_template:
                if data_type == "image":
                    crop_start = self.prompt_template.get("crop_start", -1)
                elif data_type == "video":
                    crop_start = self.prompt_template_video.get("crop_start", -1)
                else:
                    raise ValueError(f"Unsupported data type: {data_type}")
                if crop_start > 0:
                    last_hidden_state = last_hidden_state[:, crop_start:]
                    attention_mask = (
                        attention_mask[:, crop_start:] if use_attention_mask else None
                    )

            if output_hidden_states:
                return TextEncoderModelOutput(
                    last_hidden_state, attention_mask, outputs.hidden_states
                )
            return TextEncoderModelOutput(last_hidden_state, attention_mask)
        else:
            image_outputs = self.processor(semantic_images, return_tensors="pt")[
                "pixel_values"
            ].to(device=device, dtype=self.precision)
            attention_mask = (
                batch_encoding["attention_mask"].to(device)
                if use_attention_mask
                else None
            )
            outputs = self.model(
                input_ids=batch_encoding["input_ids"].to(device),
                attention_mask=attention_mask,
                output_hidden_states=output_hidden_states
                or hidden_state_skip_layer is not None,
                pixel_values=image_outputs,
            )
            if hidden_state_skip_layer is not None:
                last_hidden_state = outputs.hidden_states[
                    -(hidden_state_skip_layer + 1)
                ]
                # Real last hidden state already has layer norm applied. So here we only apply it
                # for intermediate layers.
                if hidden_state_skip_layer > 0 and self.apply_final_norm:
                    last_hidden_state = self.model.final_layer_norm(last_hidden_state)
            else:
                last_hidden_state = outputs[self.output_key]
            if self.use_template:
                if data_type == "video":
                    crop_start = self.prompt_template_video.get("crop_start", -1)
                    text_crop_start = (
                        crop_start
                        - 1
                        + self.prompt_template_video.get("image_emb_len", 576)
                    )
                    image_crop_start = self.prompt_template_video.get(
                        "image_emb_start", 5
                    )
                    image_crop_end = self.prompt_template_video.get(
                        "image_emb_end", 581
                    )
                    batch_indices, last_double_return_token_indices = torch.where(
                        batch_encoding["input_ids"]
                        == self.prompt_template_video.get("double_return_token_id", 271)
                    )
                    if last_double_return_token_indices.shape[0] == 3:
                        # in case the prompt is too long
                        last_double_return_token_indices = torch.cat(
                            (
                                last_double_return_token_indices,
                                torch.tensor([batch_encoding["input_ids"].shape[-1]]).to(
                                    device=last_double_return_token_indices.device),
                            )
                        )
                    last_double_return_token_indices = (
                        last_double_return_token_indices.reshape(
                            batch_encoding["input_ids"].shape[0], -1
                        )[:, -1]
                    )
                    assistant_crop_start = (
                        last_double_return_token_indices
                        - 1
                        + self.prompt_template_video.get("image_emb_len", 576)
                        - 4
                    )
                    assistant_crop_end = (
                        last_double_return_token_indices
                        - 1
                        + self.prompt_template_video.get("image_emb_len", 576)
                    )
                    attention_mask_assistant_crop_start = (
                        last_double_return_token_indices - 4
                    )
                    attention_mask_assistant_crop_end = last_double_return_token_indices
                else:
                    raise ValueError(f"Unsupported data type: {data_type}")

                text_last_hidden_state = []
                text_attention_mask = []
                image_last_hidden_state = []
                image_attention_mask = []
                for i in range(batch_encoding["input_ids"].shape[0]):
                    text_last_hidden_state.append(
                        torch.cat(
                            [
                                last_hidden_state[
                                    i, text_crop_start : assistant_crop_start[i].item()
                                ],
                                last_hidden_state[i, assistant_crop_end[i].item() :],
                            ]
                        )
                    )
                    text_attention_mask.append(
                        torch.cat(
                            [
                                attention_mask[
                                    i,
                                    crop_start : attention_mask_assistant_crop_start[
                                        i
                                    ].item(),
                                ],
                                attention_mask[
                                    i, attention_mask_assistant_crop_end[i].item() :
                                ],
                            ]
                        )
                        if use_attention_mask
                        else None
                    )
                    image_last_hidden_state.append(
                        last_hidden_state[i, image_crop_start:image_crop_end]
                    )
                    image_attention_mask.append(
                        torch.ones(image_last_hidden_state[-1].shape[0])
                        .to(last_hidden_state.device)
                        .to(attention_mask.dtype)
                        if use_attention_mask
                        else None
                    )

                text_last_hidden_state = torch.stack(text_last_hidden_state)
                text_attention_mask = torch.stack(text_attention_mask)
                image_last_hidden_state = torch.stack(image_last_hidden_state)
                image_attention_mask = torch.stack(image_attention_mask)

                if semantic_images is not None and 0 < self.image_embed_interleave < 6:
                    image_last_hidden_state = image_last_hidden_state[
                        :, ::self.image_embed_interleave, :
                    ]
                    image_attention_mask = image_attention_mask[
                        :, ::self.image_embed_interleave
                    ]

                assert (
                    text_last_hidden_state.shape[0] == text_attention_mask.shape[0]
                    and image_last_hidden_state.shape[0]
                    == image_attention_mask.shape[0]
                )
                #print("image_last_hidden_state", image_last_hidden_state.shape)
                #print("text_last_hidden_state", text_last_hidden_state.shape)
                last_hidden_state = torch.cat(
                    [image_last_hidden_state, text_last_hidden_state], dim=1
                )
                #print("last_hidden_state", last_hidden_state.shape)
                attention_mask = torch.cat(
                    [image_attention_mask, text_attention_mask], dim=1
                )
            if output_hidden_states:
                return TextEncoderModelOutput(
                    last_hidden_state,
                    attention_mask,
                    hidden_states_list=outputs.hidden_states,
                )
            return TextEncoderModelOutput(last_hidden_state, attention_mask)

    def forward(
        self,
        text,
        use_attention_mask=None,
        output_hidden_states=False,
        do_sample=False,
        hidden_state_skip_layer=None,
        return_texts=False,
    ):
        batch_encoding = self.text2tokens(text)
        return self.encode(
            batch_encoding,
            use_attention_mask=use_attention_mask,
            output_hidden_states=output_hidden_states,
            do_sample=do_sample,
            hidden_state_skip_layer=hidden_state_skip_layer,
            return_texts=return_texts,
        )
from diffusers.models import ModelMixin
PROMPT_TEMPLATE_ENCODE = (
    "<|start_header_id|>system<|end_header_id|>\n\nDescribe the image by detailing the color, shape, size, texture, "
    "quantity, text, spatial relationships of the objects and background:<|eot_id|>"
    "<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|>"
) 
PROMPT_TEMPLATE_ENCODE_VIDEO = (
    "<|start_header_id|>system<|end_header_id|>\n\nDescribe the video by detailing the following aspects: "
    "1. The main content and theme of the video."
    "2. The color, shape, size, texture, quantity, text, and spatial relationships of the objects."
    "3. Actions, events, behaviors temporal relationships, physical movement changes of the objects."
    "4. background environment, light, style and atmosphere."
    "5. camera angles, movements, and transitions used in the video:<|eot_id|>"
    "<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|>"
)  

PROMPT_TEMPLATE_ENCODE_I2V = (
    "<|start_header_id|>system<|end_header_id|>\n\n<image>\nDescribe the image by detailing the color, shape, size, texture, "
    "quantity, text, spatial relationships of the objects and background:<|eot_id|>"
    "<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|>"
    "<|start_header_id|>assistant<|end_header_id|>\n\n"
)

PROMPT_TEMPLATE_ENCODE_VIDEO_I2V = (
    "<|start_header_id|>system<|end_header_id|>\n\n<image>\nDescribe the video by detailing the following aspects according to the reference image: "
    "1. The main content and theme of the video."
    "2. The color, shape, size, texture, quantity, text, and spatial relationships of the objects."
    "3. Actions, events, behaviors temporal relationships, physical movement changes of the objects."
    "4. background environment, light, style and atmosphere."
    "5. camera angles, movements, and transitions used in the video:<|eot_id|>\n\n"
    "<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|>"
    "<|start_header_id|>assistant<|end_header_id|>\n\n"
)
class MatrixGameEncoderWrapperI2V(ModelMixin):
    def __init__(self, model_path, weight_dtype, task='i2v', i2v_type = 'concat'):
        super().__init__()
        text_encoder_type_1 = "llm-i2v" 
        tokenizer_type_1 = "llm-i2v" 
        prompt_template = {'template': PROMPT_TEMPLATE_ENCODE_I2V, 'crop_start': 36, "image_emb_start": 5,"image_emb_end": 581,"image_emb_len": 576,
        "double_return_token_id": 271}
        prompt_template_video = {"template": PROMPT_TEMPLATE_ENCODE_VIDEO_I2V, "crop_start": 103,"image_emb_start": 5,"image_emb_end": 581,"image_emb_len": 576,
        "double_return_token_id": 271}
        max_length_1 = 256 + (
            prompt_template_video.get("crop_start", 0)
            if prompt_template_video is not None
            else prompt_template.get("crop_start", 0)
            if prompt_template is not None
            else 0
        )
        # print("text len1:", max_length_1)
        hidden_state_skip_layer = 2
        text_encoder_precision = "fp16"
        apply_final_norm = False
        reproduce = False
        logger = None
        device = None

        text_encoder_1 = TextEncoder(
            text_encoder_type=text_encoder_type_1,
            max_length=max_length_1,
            text_encoder_precision=weight_dtype,
            tokenizer_type=tokenizer_type_1,
            i2v_mode = True,
            prompt_template=prompt_template,
            prompt_template_video=prompt_template_video,
            hidden_state_skip_layer=hidden_state_skip_layer,
            apply_final_norm=apply_final_norm,
            reproduce=reproduce,
            logger=logger,
            device=device,
            text_encoder_path = model_path,
            image_embed_interleave = 2 if i2v_type == 'concat' else 4
        )


        self.text_encoder_1 = text_encoder_1

    def encode_prompt(
        self,
        prompt,
        device,
        num_videos_per_prompt = 1,
        clip_skip: Optional[int] = None,
        text_encoder: Optional[TextEncoder] = None,
        data_type: Optional[str] = "image",
        semantic_images=None
    ):
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        
        text_inputs = text_encoder.text2tokens(prompt, data_type=data_type)

        if clip_skip is None:
            prompt_outputs = text_encoder.encode(
                text_inputs, data_type=data_type, device=device, semantic_images=semantic_images,
            )
            prompt_embeds = prompt_outputs.hidden_state
        else:
            prompt_outputs = text_encoder.encode(
                text_inputs,
                output_hidden_states=True,
                data_type=data_type,
                device=device,
                semantic_images=semantic_images
            )
            # Access the `hidden_states` first, that contains a tuple of
            # all the hidden states from the encoder layers. Then index into
            # the tuple to access the hidden states from the desired layer.
            prompt_embeds = prompt_outputs.hidden_states_list[-(clip_skip + 1)]
            # We also need to apply the final LayerNorm here to not mess with the
            # representations. The `last_hidden_states` that we typically use for
            # obtaining the final prompt representations passes through the LayerNorm
            # layer.
            prompt_embeds = text_encoder.model.text_model.final_layer_norm(
                prompt_embeds
            )

        attention_mask = prompt_outputs.attention_mask
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
            bs_embed, seq_len = attention_mask.shape
            attention_mask = attention_mask.repeat(1, num_videos_per_prompt)
            attention_mask = attention_mask.view(
                bs_embed * num_videos_per_prompt, seq_len
            )

        
        prompt_embeds_dtype = text_encoder.dtype
        prompt_embeds = prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)

        if prompt_embeds.ndim == 2:
            bs_embed, _ = prompt_embeds.shape
            # duplicate text embeddings for each generation per prompt, using mps friendly method
            prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt)
            prompt_embeds = prompt_embeds.view(bs_embed * num_videos_per_prompt, -1)
        else:
            bs_embed, seq_len, _ = prompt_embeds.shape
            # duplicate text embeddings for each generation per prompt, using mps friendly method
            prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
            prompt_embeds = prompt_embeds.view(
                bs_embed * num_videos_per_prompt, seq_len, -1
            )
        return prompt_embeds, attention_mask
        
    def forward(self, caption, semantic_images, device=None, max_length=226):
        prompt_embeds_1, attention_mask_1 = self.encode_prompt(
            caption,
            device,
            data_type = 'video', #TODO: 注意图像和视频需要使用不同的，因为llm里使用不同的template
            text_encoder = self.text_encoder_1,
            semantic_images = semantic_images
        )
        
        return (prompt_embeds_1, None), (attention_mask_1, None), None

        