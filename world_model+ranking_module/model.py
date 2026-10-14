"""X-Life action-conditioned world model.

Local Qwen interfaces, patchified CGM, Transformer transitions and trajectories,
and a shared ranking/CGM path. Architecture and training objectives are
implemented in this module and the accompanying training scripts.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from PIL import Image
from transformers import AutoModel, AutoProcessor


def local_model_directory(path, label):
    if not path or not Path(path).expanduser().is_dir():
        raise ValueError(f"Set {label} to an existing local Qwen model directory.")
    return str(Path(path).expanduser().resolve())


def hidden_size(config):
    size = getattr(config, "hidden_size", None)
    if size is None:
        size = getattr(getattr(config, "text_config", None), "hidden_size", None)
    if size is None:
        raise ValueError("The local model config must expose a text hidden_size.")
    return int(size)


def masked_token_mean(hidden, mask=None):
    if mask is None:
        return hidden.mean(dim=1)
    weights = mask.to(device=hidden.device, dtype=hidden.dtype).unsqueeze(-1)
    return (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


def causal_mask(length, device, memory_length=None):
    columns = length if memory_length is None else memory_length
    return torch.arange(columns, device=device).unsqueeze(0) > torch.arange(length, device=device).unsqueeze(1)


class TextEncoder(nn.Module):
    """Local Qwen3 encoder shared by physiological and exercise text."""

    def __init__(self, model_key=None, hidden_dim=256, use_mirror=False,
                 qwen_model_path=None, language_model=None):
        super().__init__()
        self.hidden_dim = hidden_dim
        if language_model is None:
            path = local_model_directory(qwen_model_path, "qwen_model_path")
            language_model = AutoModel.from_pretrained(
                path, local_files_only=True, trust_remote_code=True, torch_dtype="auto"
            )
        self.language_model = language_model
        size = hidden_size(language_model.config)
        for name in ("metadata", "sleep", "food", "exercise"):
            setattr(self, name + "_projection", nn.Sequential(
                nn.Linear(size, hidden_dim), nn.ReLU(), nn.Dropout(0.1),
                nn.Linear(hidden_dim, hidden_dim)
            ))

    def encode(self, input_ids, attention_mask, kind):
        projection = getattr(self, kind + "_projection")
        device = next(self.language_model.parameters()).device
        outputs = self.language_model(input_ids=input_ids.to(device),
                                      attention_mask=attention_mask.to(device), use_cache=False)
        summary = masked_token_mean(outputs.last_hidden_state, attention_mask)
        parameter = next(projection.parameters())
        return projection(summary.to(device=parameter.device, dtype=parameter.dtype))

    def forward(self, metadata_input_ids, metadata_attention_mask,
                sleep_input_ids, sleep_attention_mask, food_input_ids,
                food_attention_mask, exercise_input_ids=None, exercise_attention_mask=None):
        metadata = self.encode(metadata_input_ids, metadata_attention_mask, "metadata")
        sleep = self.encode(sleep_input_ids, sleep_attention_mask, "sleep")
        food = self.encode(food_input_ids, food_attention_mask, "food")
        if exercise_input_ids is None:
            exercise = metadata.new_zeros(metadata.size(0), self.hidden_dim)
        else:
            exercise = self.encode(exercise_input_ids, exercise_attention_mask, "exercise")
        return metadata, sleep, food, exercise


class ImageEncoder(nn.Module):
    """Joint image/text diet encoder using a locally deployed Qwen2.5-VL."""

    def __init__(self, model_name=None, hidden_dim=256, use_mirror=False,
                 force_cnn=False, model_path=None, vision_model=None, processor=None):
        super().__init__()
        if force_cnn:
            raise ValueError("Diet encoding requires a joint vision-language model.")
        if vision_model is None:
            path = local_model_directory(model_path, "vision_model_path")
            vision_model = AutoModel.from_pretrained(
                path, local_files_only=True, trust_remote_code=True, torch_dtype="auto"
            )
            processor = AutoProcessor.from_pretrained(path, local_files_only=True, trust_remote_code=True)
        if processor is None:
            raise ValueError("A processor is required with a supplied vision model.")
        self.vision_model, self.processor = vision_model, processor
        self.image_projection = nn.Sequential(nn.Linear(hidden_size(vision_model.config), hidden_dim),
                                              nn.ReLU(), nn.Dropout(0.1), nn.Linear(hidden_dim, hidden_dim))

    @staticmethod
    def _pil_image(value):
        if value is None or isinstance(value, Image.Image):
            return value
        if not isinstance(value, torch.Tensor) or value.ndim != 3:
            raise TypeError("Diet images must be PIL images, normalized CHW tensors, or None.")
        # The dataset's tensors use ImageNet normalization. PIL inputs are unchanged.
        tensor = value.detach().float().cpu()
        mean = tensor.new_tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = tensor.new_tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        rgb = (tensor * std + mean).clamp(0, 1)
        return Image.fromarray((rgb.permute(1, 2, 0).numpy() * 255).astype(np.uint8))

    def forward(self, images, texts):
        if texts is None or len(texts) != len(images):
            raise ValueError("Supply one dietary description for each sample.")
        prompts, pil_images = [], []
        for value, text in zip(images, texts):
            image = self._pil_image(value)
            content = []
            if image is not None:
                content.append({"type": "image"})
                pil_images.append(image)
            content.append({"type": "text", "text": str(text) or "No dietary information"})
            prompts.append(self.processor.apply_chat_template(
                [{"role": "user", "content": content}], tokenize=False, add_generation_prompt=False
            ))
        kwargs = dict(text=prompts, padding=True, return_tensors="pt")
        if pil_images:
            kwargs["images"] = pil_images
        inputs = self.processor(**kwargs)
        parameter = next(self.vision_model.parameters())
        inputs = {key: value.to(device=parameter.device,
                               dtype=parameter.dtype if value.is_floating_point() else value.dtype)
                  for key, value in inputs.items()}
        outputs = self.vision_model(**inputs, use_cache=False)
        summary = masked_token_mean(outputs.last_hidden_state, inputs.get("attention_mask"))
        projection_parameter = next(self.image_projection.parameters())
        return self.image_projection(summary.to(device=projection_parameter.device,
                                                dtype=projection_parameter.dtype))


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        position = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        frequencies = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        encoding = torch.zeros(max_len, d_model)
        encoding[:, 0::2] = torch.sin(position * frequencies)
        encoding[:, 1::2] = torch.cos(position * frequencies[:encoding[:, 1::2].size(1)])
        self.register_buffer("pe", encoding.unsqueeze(1))

    def forward(self, sequence):
        if sequence.size(0) > self.pe.size(0):
            raise ValueError("Sequence exceeds the configured positional encoding length.")
        return sequence + self.pe[:sequence.size(0)].to(dtype=sequence.dtype)


def transformer_encoder(dimension, heads, layers, dropout, feedforward=None):
    if dimension % heads:
        raise ValueError("Transformer state dimension must be divisible by its number of heads.")
    layer = nn.TransformerEncoderLayer(d_model=dimension, nhead=heads,
                                      dim_feedforward=feedforward or dimension * 4,
                                      dropout=dropout, activation="gelu", batch_first=True)
    return nn.TransformerEncoder(layer, num_layers=layers, enable_nested_tensor=False)


class CGMTransformerEncoder(nn.Module):
    """Non-overlapping CGM patches with explicit within-patch validity bits."""
    def __init__(self, input_dim=1, d_model=128, nhead=4, num_layers=2,
                 dim_feedforward=512, dropout=0.1, patch_size=6):
        super().__init__()
        if input_dim != 1 or patch_size < 1:
            raise ValueError("CGM is scalar-valued and patch_size must be positive.")
        self.d_model, self.patch_size = d_model, int(patch_size)
        self.input_projection = nn.Linear(self.patch_size * 2, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.transformer_encoder = transformer_encoder(d_model, nhead, num_layers, dropout, dim_feedforward)

    def patchify(self, cgm_sequence, attention_mask=None):
        if cgm_sequence.ndim != 2 or cgm_sequence.size(1) < 1:
            raise ValueError("CGM must have shape [batch, nonempty time].")
        valid = torch.isfinite(cgm_sequence)
        if attention_mask is not None:
            if attention_mask.shape != cgm_sequence.shape:
                raise ValueError("CGM values and masks must have the same shape.")
            valid = valid & attention_mask.to(device=cgm_sequence.device, dtype=torch.bool)
        values = torch.where(valid, cgm_sequence, torch.zeros_like(cgm_sequence))
        padding = (-values.size(1)) % self.patch_size
        values = F.pad(values, (0, padding))
        valid = F.pad(valid, (0, padding), value=False)
        values = values.reshape(values.size(0), -1, self.patch_size)
        valid = valid.reshape(valid.size(0), -1, self.patch_size)
        return values, valid

    def patch_weights(self, cgm_sequence, attention_mask=None):
        return self.patchify(cgm_sequence, attention_mask)[1].sum(dim=-1)

    def forward(self, cgm_sequence, attention_mask=None):
        values, valid = self.patchify(cgm_sequence, attention_mask)
        tokens = self.input_projection(torch.cat([values, valid.to(values.dtype)], dim=-1))
        tokens = self.pos_encoder(tokens.transpose(0, 1)).transpose(0, 1)
        padding = ~valid.any(dim=-1)
        # Empty physiological histories have zero pooling weight. Unmask a
        # placeholder to prevent attention over an entirely masked sequence.
        empty = padding.all(dim=1)
        padding = padding.clone()
        padding[empty, 0] = False
        return self.transformer_encoder(tokens, src_key_padding_mask=padding)


class CGMTransformerDecoder(nn.Module):
    def __init__(self, d_model=128, output_dim=1, nhead=4, num_layers=2,
                 dim_feedforward=512, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.target_input_projection = nn.Linear(output_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout,
                                           activation="gelu", batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(layer, num_layers)
        self.output_projection = nn.Linear(d_model, output_dim)

    def _decode(self, decoder_inputs, memory):
        length = decoder_inputs.size(1)
        embedded = self.target_input_projection(decoder_inputs)
        embedded = self.pos_encoder(embedded.transpose(0, 1)).transpose(0, 1)
        decoded = self.transformer_decoder(
            embedded, memory, tgt_mask=causal_mask(length, embedded.device),
            memory_mask=causal_mask(length, embedded.device, memory.size(1))
        )
        return self.output_projection(decoded)

    def forward(self, encoded_cgm, encoded_text, target_sequence=None, target_len=None):
        memory = encoded_text.unsqueeze(1) if encoded_text.ndim == 2 else encoded_text
        if memory.ndim != 3:
            raise ValueError("Decoder memory must have shape [batch, horizon, state].")
        if target_sequence is not None:
            inputs = torch.zeros_like(target_sequence)
            inputs[:, 1:] = target_sequence[:, :-1]
            return self._decode(inputs, memory)
        horizon = memory.size(1) if target_len is None else int(target_len)
        if horizon < 1:
            raise ValueError("Prediction horizon must be positive.")
        start = memory.new_zeros(memory.size(0), 1, 1)
        history = start
        generated = []
        for step in range(horizon):
            next_value = self._decode(history, memory[:, :step + 1])[:, -1:]
            generated.append(next_value)
            history = torch.cat([history, next_value], dim=1)
        return torch.cat(generated, dim=1)


class ActionTransitionModule(nn.Module):
    """Two-token Transformer transition, recurrent dynamics, temporal trajectory."""
    def __init__(self, state_dim, action_dim=None, dropout=0.1, nhead=4,
                 transition_layers=2, trajectory_layers=2):
        super().__init__()
        self.action_projection = nn.Identity() if action_dim is None else nn.Linear(action_dim, state_dim)
        self.transition = transformer_encoder(state_dim, nhead, transition_layers, dropout)
        self.dynamics_branch = nn.GRUCell(state_dim, state_dim)
        self.trajectory_branch = transformer_encoder(state_dim, nhead, trajectory_layers, dropout)
        self.trajectory_position = PositionalEncoding(state_dim)

    def _dynamic_step(self, state, action):
        action = self.action_projection(action)
        tokens = torch.stack([state, action], dim=1)
        intermediate = self.transition(tokens)[:, 0]
        return self.dynamics_branch(intermediate, state)

    def _trajectory(self, states):
        tokens = self.trajectory_position(states.transpose(0, 1)).transpose(0, 1)
        return self.trajectory_branch(tokens, mask=causal_mask(tokens.size(1), tokens.device))

    def forward(self, physiological_state, action_state):
        dynamic = self._dynamic_step(physiological_state, action_state)
        trajectory = self._trajectory(dynamic.unsqueeze(1))[:, 0]
        return dynamic, trajectory

    def rollout(self, physiological_state, action_state, horizon=1):
        horizon = int(horizon)
        if horizon < 1:
            raise ValueError("horizon must be positive.")
        if action_state.ndim == 2:
            action_state = action_state.unsqueeze(1).expand(-1, horizon, -1)
        if action_state.ndim != 3 or action_state.size(1) != horizon:
            raise ValueError("Action sequences must have shape [batch, horizon, action].")
        state, states = physiological_state, []
        for step in range(horizon):
            state = self._dynamic_step(state, action_state[:, step])
            states.append(state)
        dynamics = torch.stack(states, dim=1)
        return dynamics, self._trajectory(dynamics)


class CrossAttentionFusion(nn.Module):
    def __init__(self, state_dim, nhead=4, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(state_dim, nhead, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(state_dim)

    def forward(self, diet_state, exercise_state):
        sequence = diet_state.ndim == 3
        diet = diet_state if sequence else diet_state.unsqueeze(1)
        exercise = exercise_state if exercise_state.ndim == 3 else exercise_state.unsqueeze(1)
        mask = causal_mask(diet.size(1), diet.device, exercise.size(1))
        attended, _ = self.attention(diet, exercise, exercise, attn_mask=mask, need_weights=False)
        result = self.norm(diet + attended)
        return result if sequence else result.squeeze(1)


class PrescriptionRankingHead(nn.Module):
    """Transformer comparison of current/future states and an MLP score readout."""
    def __init__(self, state_dim, max_future_len=36, nhead=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.backbone = transformer_encoder(state_dim, nhead, num_layers, dropout)
        self.position = PositionalEncoding(state_dim)
        self.score_head = nn.Sequential(nn.Linear(state_dim, state_dim), nn.GELU(), nn.Linear(state_dim, 1))

    def forward(self, future_rollout, current_state=None):
        future = future_rollout.unsqueeze(1) if future_rollout.ndim == 2 else future_rollout
        if current_state is None:
            raise ValueError("Ranking requires the current physiological state.")
        tokens = torch.cat([current_state.unsqueeze(1), future], dim=1)
        tokens = self.position(tokens.transpose(0, 1)).transpose(0, 1)
        transformed = self.backbone(tokens, mask=causal_mask(tokens.size(1), tokens.device))[:, 1:]
        score = self.score_head(transformed.mean(dim=1)).squeeze(-1)
        return transformed, score


class CGMPredictionModelWithImage(nn.Module):
    """Shared action-conditioned model for three training phases."""
    training_only_prefixes = ("diet_annotation_head.", "exercise_annotation_head.", "omics_head.",
                              "diet_aux_decoder.", "exercise_aux_decoder.")

    def __init__(self, model_key=None, text_hidden_dim=256, cgm_d_model=128,
                 cgm_nhead=4, cgm_num_layers=2, cgm_dim_feedforward=512,
                 dropout=0.1, use_mirror=False, qwen_model_path=None,
                 use_image_features=True, vision_model_key=None, vision_model_path=None,
                 max_future_len=36, patch_size=6, annotation_dim=4096,
                 include_training_heads=True, text_encoder=None, image_encoder=None,
                 transition_layers=2, trajectory_layers=2, ranking_layers=2):
        super().__init__()
        if not use_image_features:
            raise ValueError("The diet encoder is joint image/text; text-only records may use image=None.")
        self.use_image_features, self.max_future_len = True, int(max_future_len)
        self.include_training_heads = bool(include_training_heads)
        self.annotation_dim = int(annotation_dim)
        self.text_encoder = text_encoder if text_encoder is not None else TextEncoder(
            model_key, text_hidden_dim, qwen_model_path=qwen_model_path)
        self.image_encoder = image_encoder if image_encoder is not None else ImageEncoder(
            hidden_dim=text_hidden_dim, model_path=vision_model_path)
        self.cgm_encoder = CGMTransformerEncoder(1, cgm_d_model, cgm_nhead, cgm_num_layers,
                                                 cgm_dim_feedforward, dropout, patch_size)
        self.feature_fusion = nn.Sequential(nn.Linear(text_hidden_dim * 2 + cgm_d_model, cgm_d_model),
                                            nn.ReLU(), nn.Dropout(dropout))
        self.diet_action_projection = nn.Linear(text_hidden_dim, cgm_d_model)
        self.exercise_action_projection = nn.Linear(text_hidden_dim, cgm_d_model)
        transition_args = dict(dropout=dropout, nhead=cgm_nhead,
                               transition_layers=transition_layers, trajectory_layers=trajectory_layers)
        self.diet_transition = ActionTransitionModule(cgm_d_model, **transition_args)
        self.exercise_transition = ActionTransitionModule(cgm_d_model, **transition_args)
        self.future_state_fusion = CrossAttentionFusion(cgm_d_model, cgm_nhead, dropout)
        self.ranking_head = PrescriptionRankingHead(cgm_d_model, self.max_future_len, cgm_nhead,
                                                    ranking_layers, dropout)
        decoder_args = dict(d_model=cgm_d_model, nhead=cgm_nhead, num_layers=cgm_num_layers,
                            dim_feedforward=cgm_dim_feedforward, dropout=dropout)
        # G is this Transformer decoder. Ranking and glucose readouts share
        # the ranking backbone so CGM regularization actually constrains it.
        self.cgm_decoder = CGMTransformerDecoder(**decoder_args)
        if self.include_training_heads:
            self.diet_aux_decoder = CGMTransformerDecoder(**decoder_args)
            self.exercise_aux_decoder = CGMTransformerDecoder(**decoder_args)
            self.diet_annotation_head = nn.Linear(cgm_d_model, self.annotation_dim)
            self.exercise_annotation_head = nn.Linear(cgm_d_model, self.annotation_dim)
            self.omics_head = nn.Linear(cgm_d_model, self.annotation_dim)

    @property
    def device(self):
        return next(self.feature_fusion.parameters()).device

    def encode_actions(self, *, food_texts, food_images=None,
                       exercise_input_ids, exercise_attention_mask, **unused):
        if food_texts is None:
            raise ValueError("food_texts is required for joint diet encoding.")
        if food_images is None:
            food_images = [None] * len(food_texts)
        diet = self.diet_action_projection(self.image_encoder(food_images, food_texts))
        exercise = self.exercise_action_projection(self.text_encoder.encode(
            exercise_input_ids, exercise_attention_mask, "exercise"))
        return diet, exercise

    def pretrain_annotations(self, **batch):
        if not self.include_training_heads:
            raise ValueError("Annotation supervision requires training heads.")
        diet, exercise = self.encode_actions(**batch)
        return {"diet": self.diet_annotation_head(diet),
                "exercise": self.exercise_annotation_head(exercise)}

    def _coerce_action_override(self, override, projection, raw_dim, name):
        if not isinstance(override, torch.Tensor) or override.ndim not in (2, 3):
            raise ValueError(f"{name} must be a [batch,action] or [batch,horizon,action] tensor.")
        parameter = next(projection.parameters())
        override = override.to(device=parameter.device, dtype=parameter.dtype)
        if override.size(-1) == self.cgm_encoder.d_model:
            return override
        if override.size(-1) == raw_dim:
            return projection(override)
        raise ValueError(f"Invalid {name} dimension {override.size(-1)}.")

    def forward(self, metadata_input_ids, metadata_attention_mask, sleep_input_ids, sleep_attention_mask,
                food_input_ids=None, food_attention_mask=None, exercise_input_ids=None,
                exercise_attention_mask=None, cgm_preprandial=None, cgm_postprandial=None,
                food_images=None, cgm_preprandial_mask=None, food_texts=None,
                diet_action_override=None, exercise_action_override=None,
                rollout_horizon=None, return_details=False, compute_auxiliary=False,
                decode_glucose=True):
        if not decode_glucose and (not return_details or compute_auxiliary):
            raise ValueError("Score-only calls require return_details=True and no auxiliary decoding.")
        metadata = self.text_encoder.encode(metadata_input_ids, metadata_attention_mask, "metadata")
        sleep = self.text_encoder.encode(sleep_input_ids, sleep_attention_mask, "sleep")
        encoded_cgm = self.cgm_encoder(cgm_preprandial, cgm_preprandial_mask)
        weights = self.cgm_encoder.patch_weights(cgm_preprandial, cgm_preprandial_mask)
        weights = weights.to(device=encoded_cgm.device, dtype=encoded_cgm.dtype).unsqueeze(-1)
        pooled = (encoded_cgm * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        physiological_state = self.feature_fusion(torch.cat([metadata, sleep, pooled], dim=-1))
        if diet_action_override is None:
            if food_texts is None:
                raise ValueError("Supply dietary descriptions as food_texts, or a cached diet_action_override.")
            images = food_images if food_images is not None else [None] * len(food_texts)
            diet_action = self.diet_action_projection(self.image_encoder(images, food_texts))
        else:
            diet_action = self._coerce_action_override(diet_action_override, self.diet_action_projection,
                                                       self.diet_action_projection.in_features, "diet_action")
        if exercise_action_override is None:
            if exercise_input_ids is None:
                exercise_action = physiological_state.new_zeros(physiological_state.shape)
            else:
                exercise_action = self.exercise_action_projection(self.text_encoder.encode(
                    exercise_input_ids, exercise_attention_mask, "exercise"))
        else:
            exercise_action = self._coerce_action_override(exercise_action_override, self.exercise_action_projection,
                                                           self.exercise_action_projection.in_features, "exercise_action")
        target_len = cgm_postprandial.size(1) if cgm_postprandial is not None else self.max_future_len
        horizon = target_len if rollout_horizon is None else int(rollout_horizon)
        if cgm_postprandial is not None and horizon != target_len:
            raise ValueError("Training rollout_horizon must match the supervised CGM horizon.")
        diet_dynamics, diet_trajectory = self.diet_transition.rollout(physiological_state, diet_action, horizon)
        exercise_dynamics, exercise_trajectory = self.exercise_transition.rollout(physiological_state, exercise_action, horizon)
        fused = self.future_state_fusion(diet_trajectory, exercise_trajectory)
        ranked_rollout, score = self.ranking_head(fused, physiological_state)
        target = cgm_postprandial.unsqueeze(-1) if cgm_postprandial is not None else None
        predicted = (self.cgm_decoder(encoded_cgm, ranked_rollout, target, horizon).squeeze(-1)
                     if decode_glucose else None)
        if not return_details:
            return predicted
        details = {"predicted_cgm": predicted, "ranking_glucose": predicted, "ranking_score": score,
                   "physiological_state": physiological_state, "fused_future_state": fused[:, -1],
                   "fused_future_rollout": fused, "ranking_rollout": ranked_rollout,
                   "diet_dynamic_state": diet_dynamics[:, -1], "diet_trajectory_state": diet_trajectory[:, -1],
                   "exercise_dynamic_state": exercise_dynamics[:, -1], "exercise_trajectory_state": exercise_trajectory[:, -1],
                   "diet_dynamic_rollout": diet_dynamics, "diet_trajectory_rollout": diet_trajectory,
                   "exercise_dynamic_rollout": exercise_dynamics, "exercise_trajectory_rollout": exercise_trajectory}
        if compute_auxiliary:
            if not self.include_training_heads or target is None:
                raise ValueError("Auxiliary training requires training heads and observed CGM targets.")
            details.update(
                diet_auxiliary_cgm=self.diet_aux_decoder(encoded_cgm, diet_trajectory, target, horizon).squeeze(-1),
                exercise_auxiliary_cgm=self.exercise_aux_decoder(encoded_cgm, exercise_trajectory, target, horizon).squeeze(-1),
                omics_prediction=self.omics_head(fused.mean(dim=1)))
        return details

    def inference_state_dict(self):
        return {key: value for key, value in self.state_dict().items()
                if not key.startswith(self.training_only_prefixes)}

    def simulate_actions(self, *, diet_action=None, exercise_action=None, rollout_horizon=None, **context):
        for key in ("cgm_postprandial", "diet_action_override", "exercise_action_override", "return_details", "compute_auxiliary", "decode_glucose"):
            context.pop(key, None)
        return self.forward(**context, diet_action_override=diet_action, exercise_action_override=exercise_action,
                            rollout_horizon=rollout_horizon, return_details=True)

    @torch.no_grad()
    def simulate_counterfactuals(self, context, action_variants, rollout_horizon=None):
        if not action_variants:
            raise ValueError("Supply at least one candidate action.")
        was_training = self.training
        self.eval()
        try:
            outputs = []
            for variant in action_variants:
                variant = dict(variant)
                diet = variant.pop("diet_action", variant.pop("diet_action_override", None))
                exercise = variant.pop("exercise_action", variant.pop("exercise_action_override", None))
                merged = dict(context); merged.update(variant)
                outputs.append(self.simulate_actions(diet_action=diet, exercise_action=exercise,
                                                     rollout_horizon=rollout_horizon, **merged))
            keys = ("predicted_cgm", "ranking_glucose", "ranking_score", "diet_dynamic_rollout",
                    "diet_trajectory_rollout", "exercise_dynamic_rollout", "exercise_trajectory_rollout",
                    "fused_future_rollout")
            result = {key: torch.stack([output[key] for output in outputs], dim=1) for key in keys}
            for key in ("predicted_cgm", "ranking_score"):
                result[key + "_delta"] = result[key] - result[key][:, :1]
            return result
        finally:
            self.train(was_training)
