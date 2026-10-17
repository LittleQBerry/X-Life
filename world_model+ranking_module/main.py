import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
import os
import json
from tqdm import tqdm
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms
from datetime import datetime, timedelta

# Import custom modules
from dataset import SampleImageDataset, get_tokenizer
from model import CGMPredictionModelWithImage
from training import (LossWeights, configure_training_phase, set_training_mode,
                      encoder_pretraining_objective, world_model_objective, to_device)

class SampleImagePredictionDataset(Dataset):

    def __init__(self, dataset, tokenizer, max_seq_length=512, cgm_seq_length=92, image_dir=None,
                 image_size=(224, 224), annotation_dim=4096, require_cgm=True, sampling_interval_minutes=5):
        """
        Initialize dataset

        Args:
            dataset: Original dataset
            tokenizer: Language model tokenizer
            max_seq_length: Maximum text sequence length
            cgm_seq_length: CGM sequence length
            image_dir: Image directory path
            image_size: Image size
        """
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.cgm_seq_length = cgm_seq_length
        self.image_dir = image_dir
        self.image_size = image_size
        self.annotation_dim = int(annotation_dim)
        self.sampling_interval_minutes = float(sampling_interval_minutes)
        if self.sampling_interval_minutes <= 0 or self.cgm_seq_length < 1:
            raise ValueError("CGM interval and sequence length must be positive.")

        self.original_dataset = dataset

        # Image transformation
        self.image_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.valid_samples = []
        for i in range(len(self.original_dataset)):
            sample = self.original_dataset[i]
            if not require_cgm or (sample['cgm_pre'] and sample['cgm_post']):
                self.valid_samples.append(i)

        print(f"Original dataset size: {len(self.original_dataset)}")
        print(f"Valid samples count: {len(self.valid_samples)}")

    def __len__(self):
        return len(self.valid_samples)

    def __getitem__(self, idx):
        """Get single sample"""
        sample_idx = self.valid_samples[idx]
        sample = self.original_dataset[sample_idx]

        # Process metadata text
        metadata = sample.get('metadata')
        if metadata is None:
            metadata = {}
        patient_info = self.original_dataset.get_patient_basic_info(metadata)
        metadata_text = self._dict_to_text(patient_info)
        metadata_encoding = self.tokenizer(
            metadata_text,
            max_length=self.max_seq_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # Process sleep data text
        sleep_info = self.original_dataset.get_sleep_info(sample['sleep_data'])
        sleep_text = self._dict_to_text(sleep_info)
        sleep_encoding = self.tokenizer(
            sleep_text,
            max_length=self.max_seq_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # Process meal data text
        food_data = sample.get('food_data', {})
        meal_info = self.original_dataset.get_meal_info(food_data)
        meal_text = self._meal_info_to_text(meal_info)
        food_encoding = self.tokenizer(
            meal_text,
            max_length=self.max_seq_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # Process exercise data text
        exercise_info = self.original_dataset.get_exercise_info(sample['exercise_data'])
        exercise_text = self._dict_to_text(exercise_info)
        exercise_encoding = self.tokenizer(
            exercise_text,
            max_length=self.max_seq_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # Process image data
        food_images = []
        if 'image_paths' in meal_info and meal_info['image_paths']:
            for image_path in meal_info['image_paths']:
                try:

                    if self.image_dir and not os.path.isabs(image_path):
                        full_path = os.path.join(self.image_dir, image_path)
                    else:
                        full_path = image_path


                    if os.path.exists(full_path):
                        image = Image.open(full_path).convert('RGB')
                        image_tensor = self.image_transform(image)
                        food_images.append(image_tensor)
                    else:
                        raise FileNotFoundError(full_path)
                except Exception as e:
                    raise ValueError(f"Failed to load dietary image {image_path}") from e

        # Get basic information
        subject_id = sample['subject_id']
        date = meal_info.get('date', '')
        meal_type = meal_info.get('food_id', '')

        # CGM records are selected by the meal event time before padding to a
        # batch length.  The returned masks distinguish measured points from
        # padding, so copied/filler values never contribute to the objective.
        if sample.get('event_time'):
            event_time = self._meal_event_time('', sample['event_time'])
        elif 'event_time' in sample:
            event_time = None
        elif sample['cgm_type'] == 'exercise':
            event_time = self._meal_event_time('', exercise_info.get('recorded_at', ''))
        else:
            event_time = self._meal_event_time(date, meal_info.get('time', ''))
        cgm_pre, cgm_pre_mask = self._process_precgm_sequence(
            sample['cgm_pre'], event_time=event_time, return_mask=True
        )
        cgm_post, cgm_post_mask = self._process_postcgm_sequence(
            sample['cgm_post'], event_time=event_time, return_mask=True
        )

        semantic_targets, semantic_masks = {}, {}
        target_record = sample.get('training_targets') or {}
        for kind in ('diet', 'exercise', 'omics'):
            value = target_record.get(kind)
            if value is None:
                semantic_targets[kind] = torch.zeros(self.annotation_dim)
                semantic_masks[kind] = torch.tensor(False)
            else:
                vector = torch.as_tensor(value, dtype=torch.float32)
                if vector.shape != (self.annotation_dim,) or not torch.isfinite(vector).all():
                    raise ValueError(f"{kind} target must be a finite vector of length {self.annotation_dim}.")
                semantic_targets[kind] = vector
                semantic_masks[kind] = torch.tensor(True)
        return {
            'metadata_input_ids': metadata_encoding['input_ids'].squeeze(),
            'metadata_attention_mask': metadata_encoding['attention_mask'].squeeze(),
            'sleep_input_ids': sleep_encoding['input_ids'].squeeze(),
            'sleep_attention_mask': sleep_encoding['attention_mask'].squeeze(),
            'food_input_ids': food_encoding['input_ids'].squeeze(),
            'food_attention_mask': food_encoding['attention_mask'].squeeze(),
            'exercise_input_ids': exercise_encoding['input_ids'].squeeze(),
            'exercise_attention_mask': exercise_encoding['attention_mask'].squeeze(),
            'food_images': food_images,  # List of image tensors
            'food_texts': meal_text,
            'semantic_targets': semantic_targets,
            'semantic_masks': semantic_masks,
            'cgm_type': sample['cgm_type'],
            'cgm_preprandial': torch.tensor(cgm_pre, dtype=torch.float),
            'cgm_postprandial': torch.tensor(cgm_post, dtype=torch.float),
            'cgm_preprandial_mask': torch.tensor(cgm_pre_mask, dtype=torch.bool),
            'cgm_postprandial_mask': torch.tensor(cgm_post_mask, dtype=torch.bool),
            'subject_id': subject_id,
            'date': date,
            'meal_type': meal_type
        }

    def _dict_to_text(self, data_dict):
        """Convert dictionary to text"""
        if not data_dict:
            return "No data"

        text_parts = []
        for key, value in data_dict.items():
            if value is not None and value != 'N/A' and value != 'Unknown':
                text_parts.append(f"{key}:{value}")

        return '; '.join(text_parts)

    def _meal_info_to_text(self, meal_info):
        """Convert meal information to text"""
        if not meal_info:
            return "No meal data"

        fields = ('meal_type', 'time', 'food_names', 'description', 'ingredients', 'preparation', 'preparation_method',
                  'nutrition', 'nutrients', 'carbohydrate_g', 'protein_g', 'fat_g', 'fiber_g')
        content = {key: meal_info[key] for key in fields if meal_info.get(key) is not None}
        items = [{key: value for key, value in item.items() if key not in {'image_name', 'image_path'}}
                 for item in meal_info.get('food_items', [])]
        content['items'] = [item for item in items if item]
        return json.dumps(content, ensure_ascii=False, sort_keys=True) if content else 'No dietary description'

    @staticmethod
    def _meal_event_time(date, time):
        """Return a meal timestamp when the source contains one."""
        if not date and not time:
            return None
        try:
            value = f"{date} {time}" if date and time else (date or time)
            return pd.to_datetime(value).to_pydatetime()
        except (TypeError, ValueError, OverflowError):
            return None

    @staticmethod
    def _parse_cgm_point(point):
        """Parse ``(timestamp, value)`` while tolerating legacy numeric rows."""
        if isinstance(point, (int, float, np.number)):
            return (None, float(point)) if np.isfinite(point) else None
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            return None
        try:
            value = float(point[1])
        except (TypeError, ValueError):
            return None
        if not np.isfinite(value):
            return None
        timestamp = None
        try:
            timestamp = pd.to_datetime(point[0]).to_pydatetime()
        except (TypeError, ValueError, OverflowError):
            # A missing/unparseable time is retained when no event timestamp
            # is available; it is not silently treated as a real time point.
            pass
        return timestamp, value

    def _process_cgm_sequence(self, cgm_data, event_time=None, window='pre', return_mask=False):
        """Select a three-hour CGM window and pad without fabricating targets.

        Values are returned at a fixed batch length for the existing model.
        ``valid_mask`` is true only for original measurements.  Padding uses
        zero (rather than a repeated last value), and callers must apply the
        mask to losses and metrics.
        """
        parsed = [self._parse_cgm_point(point) for point in (cgm_data or [])]
        parsed = [point for point in parsed if point is not None]
        parsed.sort(key=lambda item: item[0] or datetime.min)

        if event_time is not None:
            start = event_time - timedelta(hours=3)
            end = event_time + timedelta(hours=3)
            if window == 'pre':
                parsed = [p for p in parsed if p[0] is not None and start <= p[0] < event_time]
            else:
                parsed = [p for p in parsed if p[0] is not None and event_time < p[0] <= end]

            # Keep missing measurements in their time slots. Never interpolate
            # future labels. The configured interval must match the input CGM.
            values = [0.0] * self.cgm_seq_length
            valid_mask = [False] * self.cgm_seq_length
            interval = self.sampling_interval_minutes * 60.0
            for timestamp, value in parsed:
                offset = (timestamp - event_time).total_seconds() / interval
                slot = (int(np.floor(offset)) + self.cgm_seq_length if window == 'pre'
                        else int(np.ceil(offset)) - 1)
                if 0 <= slot < self.cgm_seq_length:
                    values[slot], valid_mask[slot] = value, True
            return (values, valid_mask) if return_mask else values

        # For an overlong history retain points nearest to the event; for a
        # future target retain the earliest points in chronological order.
        if len(parsed) > self.cgm_seq_length:
            parsed = parsed[-self.cgm_seq_length:] if window == 'pre' else parsed[:self.cgm_seq_length]

        values = [value for _, value in parsed]
        values = values[:self.cgm_seq_length]
        valid_mask = [True] * len(values)
        pad_len = self.cgm_seq_length - len(values)
        if pad_len > 0:
            if window == 'pre':
                values = [0.0] * pad_len + values
                valid_mask = [False] * pad_len + valid_mask
            else:
                values.extend([0.0] * pad_len)
                valid_mask.extend([False] * pad_len)

        if return_mask:
            return values, valid_mask
        return values

    def _process_precgm_sequence(self, cgm_data, event_time=None, return_mask=False):
        return self._process_cgm_sequence(cgm_data, event_time=event_time, window='pre', return_mask=return_mask)

    def _process_postcgm_sequence(self, cgm_data, event_time=None, return_mask=False):
        return self._process_cgm_sequence(cgm_data, event_time=event_time, window='post', return_mask=return_mask)

# Qwen weights and tokenizers are loaded only from explicit local directories.

# Model-loading interfaces use local directories only. No hub downloads.


class CGMPredictionTrainerWithImage:
    """CGM prediction model trainer with image input support"""

    def __init__(self, config):
        """
        Initialize trainer

        Args:
            config: Training configuration dictionary
        """
        # Get parameters from configuration
        self.data_path = config['data_path']
        self.model_key = config['model_key']
        self.vision_model_key = config['vision_model_key']
        self.model_name = config['model_name']
        self.batch_size = config['batch_size']
        self.learning_rate = config['learning_rate']
        self.num_epochs = config['num_epochs']
        self.save_dir = config['save_dir']
        self.image_dir = config.get('image_dir', None)

        # Model parameters
        self.text_hidden_dim = config.get('text_hidden_dim', config.get('hidden_dim', 256))
        self.cgm_d_model = config.get('cgm_d_model', 128)
        self.cgm_nhead = config.get('cgm_nhead', 8)
        self.cgm_num_layers = config.get('cgm_num_layers', 3)
        self.cgm_dim_feedforward = config.get('cgm_dim_feedforward', 512)
        self.dropout = config.get('dropout', 0.1)
        self.use_image_features = config.get('use_image_features', True)
        self.cgm_seq_length = int(config.get('cgm_seq_length', 32))
        self.patch_size = int(config.get('patch_size', 6))
        self.annotation_dim = int(config.get('annotation_dim', 4096))
        self.loss_weights = LossWeights(**config.get('loss_weights', {}))
        self.training_phase = config.get('training_phase', 'world_model')
        self.freeze_action_encoders = config.get(
            'freeze_action_encoders', self.training_phase != 'encoder_pretraining'
        )
        # A full observed post-CGM window is used by default so the latent
        # transition is trained as a multi-step rollout.  Set an integer to
        # intentionally train/evaluate a shorter horizon.
        self.rollout_horizon = config.get('rollout_horizon', None)
        # The ranking head's glucose trajectory is an auxiliary supervised
        # target.  Its scalar preference score still requires preference pairs
        # and is updated by preference_alignment_step().
        self.ranking_glucose_loss_weight = float(
            config.get('ranking_glucose_loss_weight', 0.25)
        )

        # Training parameters
        self.early_stopping_patience = config['early_stopping_patience']
        self.lr_scheduler_patience = config['lr_scheduler_patience']
        self.weight_decay = config['weight_decay']
        self.gradient_clip_norm = config['gradient_clip_norm']
        self.log_interval = config['log_interval']
        self.eval_interval = config['eval_interval']
        self.save_interval = config['save_interval']
        self.split_seeds = config.get('split_seeds', [42, 43, 44, 45, 46])
        self.split_index = int(config.get('split_index', 0))

        # Gradient accumulation parameters
        self.gradient_accumulation_steps = int(config.get('gradient_accumulation_steps', 4))

        self.model_path = config.get('model_path', None)

        device = config.get('device')
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        print(f"Using device: {self.device}")

        self.save_dir = config['save_dir']
        os.makedirs(self.save_dir, exist_ok=True)

        config_save_path = os.path.join(self.save_dir, 'config.json')
        with open(config_save_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        print(f"Configuration saved to: {config_save_path}")

        self.qwen_model_path = config.get('qwen_model_path', config.get('text_model_path'))
        self.qwen_vl_model_path = config.get('vision_model_path')
        self.tokenizer = get_tokenizer(model_path=self.qwen_model_path)


        # Load data
        self._load_data()

        # Initialize model
        self._initialize_model()

        # Apply phase-specific parameter freezing before building the optimizer.
        self.set_training_phase(self.training_phase)

        # Initialize optimizer and loss function
        self._initialize_optimizer()

        # Record training history
        self.train_losses = []
        self.val_losses = []

    def _load_data(self):
        """Load and split data"""
        print("Loading data...")

        # Use SampleImageDataset to load data
        dataset = SampleImageDataset(self.data_path, image_dir=self.image_dir)
        full_dataset = SampleImagePredictionDataset(
            dataset=dataset,
            tokenizer=self.tokenizer,
            max_seq_length=512,
            cgm_seq_length=self.cgm_seq_length,
            annotation_dim=self.annotation_dim,
            require_cgm=self.training_phase != 'encoder_pretraining',
            image_dir=self.image_dir
        )

        print(f"Dataset size: {len(full_dataset)}")

        # Split by participant, never by meal.  A participant's meals are
        # kept together so validation measures generalisation to unseen
        # people.  The helper also exposes five deterministic seeds for the
        # repeated participant-level splits used by the manuscript.
        subject_ids = [
            full_dataset.original_dataset[original_idx]['subject_id']
            for original_idx in full_dataset.valid_samples
        ]
        unique_subjects = {str(subject_id) for subject_id in subject_ids}
        if len(unique_subjects) < 2:
            raise ValueError(
                "Participant-level 80/20 validation requires at least two "
                "participants. Provide a dataset with at least two distinct "
                "participant IDs."
            )
        split_seeds = getattr(self, 'split_seeds', [42, 43, 44, 45, 46])
        all_splits = self._participant_level_splits(
            subject_ids,
            train_fraction=0.8,
            seeds=split_seeds,
        )
        self.participant_splits = all_splits
        split_index = int(getattr(self, 'split_index', 0))
        split_index = max(0, min(split_index, len(all_splits) - 1))
        train_indices, val_indices, train_subjects, val_subjects = all_splits[split_index]
        train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
        val_dataset = torch.utils.data.Subset(full_dataset, val_indices)
        self.train_subject_ids = train_subjects
        self.val_subject_ids = val_subjects
        overlap = set(train_subjects).intersection(val_subjects)
        if overlap:
            raise RuntimeError(f'Participant leakage between train and validation: {sorted(overlap)}')


        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=self._collate_fn
        )

        self.val_loader = DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            collate_fn=self._collate_fn
        )

        print(f"Training set size: {len(train_dataset)}")
        print(f"Validation set size: {len(val_dataset)}")
        print(f"Training participants: {len(train_subjects)}; validation participants: {len(val_subjects)}")

    @staticmethod
    def _participant_level_splits(subject_ids, train_fraction=0.8, seeds=None):
        """Build reproducible participant-level train/validation splits.

        Returns a list of ``(train_indices, val_indices, train_subjects,
        val_subjects)`` tuples.  ``seeds`` defaults to the five seeds used by
        the public training entry point and can be replaced by an experiment's
        recorded split seeds.
        """
        if seeds is None:
            seeds = [42, 43, 44, 45, 46]
        unique_subjects = sorted({str(subject) for subject in subject_ids})
        if not unique_subjects:
            return [([], [], [], []) for _ in seeds]
        if len(unique_subjects) == 1:
            only = unique_subjects[0]
            indices = list(range(len(subject_ids)))
            return [(indices, [], [only], []) for _ in seeds]

        train_count = int(round(len(unique_subjects) * train_fraction))
        train_count = min(max(train_count, 1), len(unique_subjects) - 1)
        splits = []
        for seed in seeds:
            rng = np.random.default_rng(int(seed))
            shuffled = list(unique_subjects)
            rng.shuffle(shuffled)
            train_subjects = sorted(shuffled[:train_count])
            val_subjects = sorted(shuffled[train_count:])
            train_set, val_set = set(train_subjects), set(val_subjects)
            train_indices = [i for i, sid in enumerate(subject_ids) if str(sid) in train_set]
            val_indices = [i for i, sid in enumerate(subject_ids) if str(sid) in val_set]
            splits.append((train_indices, val_indices, train_subjects, val_subjects))
        return splits

    def _collate_fn(self, batch):
        """Custom batch processing function to handle image data"""

        metadata_input_ids = torch.stack([item['metadata_input_ids'] for item in batch])
        metadata_attention_mask = torch.stack([item['metadata_attention_mask'] for item in batch])
        sleep_input_ids = torch.stack([item['sleep_input_ids'] for item in batch])
        sleep_attention_mask = torch.stack([item['sleep_attention_mask'] for item in batch])
        food_input_ids = torch.stack([item['food_input_ids'] for item in batch])
        food_attention_mask = torch.stack([item['food_attention_mask'] for item in batch])
        exercise_input_ids = torch.stack([item['exercise_input_ids'] for item in batch])
        exercise_attention_mask = torch.stack([item['exercise_attention_mask'] for item in batch])
        cgm_preprandial = torch.stack([item['cgm_preprandial'] for item in batch])
        cgm_postprandial = torch.stack([item['cgm_postprandial'] for item in batch])
        cgm_preprandial_mask = torch.stack([item['cgm_preprandial_mask'] for item in batch])
        cgm_postprandial_mask = torch.stack([item['cgm_postprandial_mask'] for item in batch])


        food_images = [item['food_images'] or None for item in batch]


        subject_ids = [item['subject_id'] for item in batch]
        dates = [item['date'] for item in batch]
        meal_types = [item['meal_type'] for item in batch]

        return {
            'metadata_input_ids': metadata_input_ids,
            'metadata_attention_mask': metadata_attention_mask,
            'sleep_input_ids': sleep_input_ids,
            'sleep_attention_mask': sleep_attention_mask,
            'food_input_ids': food_input_ids,
            'food_attention_mask': food_attention_mask,
            'exercise_input_ids': exercise_input_ids,
            'exercise_attention_mask': exercise_attention_mask,
            'food_images': food_images,
            'food_texts': [item['food_texts'] for item in batch],
            'semantic_targets': {key: torch.stack([item['semantic_targets'][key] for item in batch]) for key in ('diet', 'exercise', 'omics')},
            'semantic_masks': {key: torch.stack([item['semantic_masks'][key] for item in batch]) for key in ('diet', 'exercise', 'omics')},
            'cgm_type': [item['cgm_type'] for item in batch],
            'cgm_preprandial': cgm_preprandial,
            'cgm_postprandial': cgm_postprandial,
            'cgm_preprandial_mask': cgm_preprandial_mask,
            'cgm_postprandial_mask': cgm_postprandial_mask,
            'subject_id': subject_ids,
            'date': dates,
            'meal_type': meal_types
        }

    def _initialize_model(self):
        """Initialize model"""
        print("Initializing model...")

        self.model = CGMPredictionModelWithImage(
            model_key=self.model_key,
            text_hidden_dim=self.text_hidden_dim,
            cgm_d_model=self.cgm_d_model,
            cgm_nhead=self.cgm_nhead,
            cgm_num_layers=self.cgm_num_layers,
            cgm_dim_feedforward=self.cgm_dim_feedforward,
            dropout=self.dropout,
            use_mirror=True,
            qwen_model_path=getattr(self, 'qwen_model_path', None),
            use_image_features=self.use_image_features,
            vision_model_key=self.vision_model_key,
            vision_model_path=getattr(self, 'qwen_vl_model_path', None),
            max_future_len=self.cgm_seq_length, patch_size=self.patch_size,
            annotation_dim=self.annotation_dim
        )


        self.model.to(self.device)

        # If model_path exists, load pre-trained weights
        if self.model_path and os.path.exists(self.model_path):
            print(f"Loading model weights: {self.model_path}")
            checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)


            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
                print("Model weights loaded from checkpoint complete")
            else:

                self.model.load_state_dict(checkpoint)
                print("Model weights loading complete")
        elif self.model_path:
            print(f"Warning: Specified model_path does not exist: {self.model_path}")


        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        print(f"Total model parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")

    def _initialize_optimizer(self):
        """Initialize optimizer and loss function"""
        # Use different learning rates for different parts and include only
        # parameters enabled by the selected training phase.
        candidate_groups = [
            (self.model.text_encoder, self.learning_rate * 0.1),
            (getattr(self.model, 'image_encoder', None), self.learning_rate * 0.1),
            (self.model.cgm_encoder, self.learning_rate),
            (self.model.feature_fusion, self.learning_rate),
            (getattr(self.model, 'diet_action_projection', None), self.learning_rate * 0.5),
            (getattr(self.model, 'exercise_action_projection', None), self.learning_rate * 0.5),
            (self.model.cgm_decoder, self.learning_rate),
            (getattr(self.model, 'diet_transition', None), self.learning_rate),
            (getattr(self.model, 'exercise_transition', None), self.learning_rate),
            (getattr(self.model, 'future_state_fusion', None), self.learning_rate),
            (getattr(self.model, 'ranking_head', None), self.learning_rate),
            *[(getattr(self.model, name, None), self.learning_rate) for name in ('diet_annotation_head', 'exercise_annotation_head', 'diet_aux_decoder', 'exercise_aux_decoder', 'omics_head')],
        ]
        param_groups = []
        for module, lr in candidate_groups:
            if module is None:
                continue
            params = [p for p in module.parameters() if p.requires_grad]
            if params:
                param_groups.append({'params': params, 'lr': lr})

        self.optimizer = optim.AdamW(param_groups, weight_decay=self.weight_decay)


        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=self.lr_scheduler_patience, factor=0.5
        )
        self.criterion = nn.MSELoss()

    def set_training_phase(self, phase):
        self.training_phase = phase
        configure_training_phase(self.model, phase)


    @staticmethod
    def dpo_loss(preferred_score, rejected_score, reference_preferred_score=None,
                 reference_rejected_score=None, beta=0.1):
        """Compute DPO loss for same-state preferred/rejected prescriptions."""
        policy_margin = preferred_score - rejected_score
        if reference_preferred_score is not None and reference_rejected_score is not None:
            policy_margin = policy_margin - (reference_preferred_score - reference_rejected_score)
        return -F.logsigmoid(beta * policy_margin).mean()

    @staticmethod
    def _masked_mse(prediction, target, valid_mask):
        """Mean squared error over measured CGM points only."""
        if valid_mask is None:
            return F.mse_loss(prediction, target)
        weights = valid_mask.to(device=prediction.device, dtype=prediction.dtype)
        squared_error = (prediction - target).pow(2)
        denominator = weights.sum().clamp_min(1.0)
        return (squared_error * weights).sum() / denominator

    def train_epoch(self, data_loader=None):
        set_training_mode(self.model)
        loader = self.train_loader if data_loader is None else data_loader
        if len(loader) == 0:
            raise ValueError("Training data loader is empty.")
        objective = (encoder_pretraining_objective if self.training_phase == 'encoder_pretraining'
                     else world_model_objective)
        if self.training_phase not in {'encoder_pretraining', 'world_model'}:
            raise ValueError("Use train.py preference alignment for paired preference training.")
        self.optimizer.zero_grad()
        total = 0.0
        for i, batch in enumerate(tqdm(loader, desc='Training')):
            loss, _ = objective(self.model, to_device(batch, self.device), self.loss_weights)
            group_start = (i // self.gradient_accumulation_steps) * self.gradient_accumulation_steps
            group_size = min(self.gradient_accumulation_steps, len(loader) - group_start)
            (loss / group_size).backward()
            total += float(loss.detach())
            if (i + 1) % self.gradient_accumulation_steps == 0 or i + 1 == len(loader):
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_norm)
                self.optimizer.step(); self.optimizer.zero_grad()
        return total / len(loader)


    def update_participant(self, participant_loader, observed_loader=None, epochs=1, stage='user'):
        """Update the ranker/G with paired feedback and this participant's factual CGM."""
        from training import FrozenRankingReference, preference_objective
        from train import cycle_next
        if observed_loader is None or not len(observed_loader) or not len(participant_loader):
            raise ValueError("Supply paired preference and factual observed CGM loaders.")
        if stage not in {'physiological', 'user'} or epochs < 1:
            raise ValueError("Participant updates require physiological/user preferences and positive epochs.")
        self.set_training_phase('participant_update')
        self._initialize_optimizer()
        reference = FrozenRankingReference(self.model).to(self.device)
        history = []
        participant_id = None
        for epoch in range(epochs):
            set_training_mode(self.model)
            observed_iterator = iter(observed_loader)
            total = 0.0
            for pairs in participant_loader:
                observed, observed_iterator = cycle_next(observed_loader, observed_iterator)
                ids = set(map(str, pairs['preferred']['subject_id'] + observed['subject_id']))
                if len(ids) != 1 or (participant_id is not None and ids != {participant_id}):
                    raise ValueError("Participant update loaders must belong to one identical participant.")
                participant_id = next(iter(ids))
                loss, _ = preference_objective(self.model, to_device(pairs, self.device),
                                               to_device(observed, self.device), reference, self.loss_weights, stage)
                self.optimizer.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_norm)
                self.optimizer.step(); total += float(loss.detach())
            history.append(total / len(participant_loader))
        return history

    def preference_alignment_step(self, pairs, observed_batch, reference, stage='professional'):
        from training import preference_objective
        if self.training_phase != 'preference_alignment':
            self.set_training_phase('preference_alignment')
            self._initialize_optimizer()
        set_training_mode(self.model)
        loss, _ = preference_objective(self.model, to_device(pairs, self.device),
                                      to_device(observed_batch, self.device), reference,
                                      self.loss_weights, stage)
        self.optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_norm)
        self.optimizer.step()
        return loss.detach()


    @torch.no_grad()
    def simulate_counterfactuals(self, context, action_variants, rollout_horizon=None):
        """Expose the model's same-state what-if comparison to callers."""
        return self.model.simulate_counterfactuals(
            context, action_variants, rollout_horizon=rollout_horizon
        )

    def validate_epoch(self):
        """Validate one epoch"""
        self.model.eval()
        total_loss = 0.0
        num_batches = len(self.val_loader)


        all_predictions = []
        all_targets = []
        all_target_masks = []

        with torch.no_grad():
            progress_bar = tqdm(self.val_loader, desc="Validation")

            for batch in progress_bar:
                # Move data to device
                metadata_input_ids = batch['metadata_input_ids'].to(self.device)
                metadata_attention_mask = batch['metadata_attention_mask'].to(self.device)
                sleep_input_ids = batch['sleep_input_ids'].to(self.device)
                sleep_attention_mask = batch['sleep_attention_mask'].to(self.device)
                food_input_ids = batch['food_input_ids'].to(self.device)
                food_attention_mask = batch['food_attention_mask'].to(self.device)
                exercise_input_ids = batch['exercise_input_ids'].to(self.device)
                exercise_attention_mask = batch['exercise_attention_mask'].to(self.device)
                food_images = [img.to(self.device) if isinstance(img, torch.Tensor) else img for img in batch['food_images']]
                cgm_preprandial = batch['cgm_preprandial'].to(self.device)
                cgm_postprandial = batch['cgm_postprandial'].to(self.device)
                cgm_preprandial_mask = batch['cgm_preprandial_mask'].to(self.device)
                cgm_postprandial_mask = batch['cgm_postprandial_mask'].to(self.device)

                predicted_cgm = self.model(
                    metadata_input_ids=metadata_input_ids,
                    metadata_attention_mask=metadata_attention_mask,
                    sleep_input_ids=sleep_input_ids,
                    sleep_attention_mask=sleep_attention_mask,
                    food_input_ids=food_input_ids,
                    food_attention_mask=food_attention_mask,
                    exercise_input_ids=exercise_input_ids,
                    exercise_attention_mask=exercise_attention_mask,
                    cgm_preprandial=cgm_preprandial,
                    cgm_postprandial=None,
                    food_images=food_images,
                    food_texts=batch['food_texts'],
                    cgm_preprandial_mask=cgm_preprandial_mask,
                    rollout_horizon=self.rollout_horizon,
                )

                loss = self._masked_mse(predicted_cgm, cgm_postprandial, cgm_postprandial_mask)

                total_loss += loss.item()

                all_predictions.append(predicted_cgm.cpu().numpy())
                all_targets.append(cgm_postprandial.cpu().numpy())
                all_target_masks.append(cgm_postprandial_mask.cpu().numpy())

                progress_bar.set_postfix({'loss': loss.item()})

        avg_loss = total_loss / num_batches

        all_predictions = np.concatenate(all_predictions, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        all_target_masks = np.concatenate(all_target_masks, axis=0) if all_target_masks else np.empty_like(all_targets, dtype=bool)

        pearson_rs = []
        for i in range(all_predictions.shape[1]):  # Iterate through each time point
            pred_seq = all_predictions[:, i]
            target_seq = all_targets[:, i]

            # Filter out NaN values
            valid_mask = all_target_masks[:, i].astype(bool) & ~(np.isnan(pred_seq) | np.isnan(target_seq))
            if np.sum(valid_mask) > 1:  # Need at least 2 valid points
                pearson_r, _ = pearsonr(pred_seq[valid_mask], target_seq[valid_mask])
                if not np.isnan(pearson_r):
                    pearson_rs.append(pearson_r)

        # Calculate average Pearson correlation coefficient
        avg_pearson_r = np.mean(pearson_rs) if pearson_rs else 0.0

        return avg_loss, avg_pearson_r

    def train(self):
        """Train model"""
        print("Starting training...")

        training_info = {
            'device': str(self.device),
            'model_name': self.model_name,
            'model_key': self.model_key,
            'dataset_info': {
                'train_size': len(self.train_loader.dataset),
                'val_size': len(self.val_loader.dataset),
                'batch_size': self.batch_size,
                'use_image_features': self.use_image_features
            },
            'model_params': {
                'total_params': sum(p.numel() for p in self.model.parameters()),
                'trainable_params': sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            },
            'training_params': {
                'learning_rate': self.learning_rate,
                'num_epochs': self.num_epochs,
                'early_stopping_patience': self.early_stopping_patience,
                'weight_decay': self.weight_decay,
                'gradient_clip_norm': self.gradient_clip_norm,
                'gradient_accumulation_steps': self.gradient_accumulation_steps,
                'rollout_horizon': self.rollout_horizon,
                'ranking_glucose_loss_weight': self.ranking_glucose_loss_weight,
            },
            'model_architecture': {
                'text_hidden_dim': self.text_hidden_dim,
                'cgm_d_model': self.cgm_d_model,
                'cgm_nhead': self.cgm_nhead,
                'cgm_num_layers': self.cgm_num_layers,
                'cgm_dim_feedforward': self.cgm_dim_feedforward,
                'dropout': self.dropout
            }
        }

        training_info_path = os.path.join(self.save_dir, 'training_info.json')
        with open(training_info_path, 'w', encoding='utf-8') as f:
            json.dump(training_info, f, indent=2, ensure_ascii=False)
        print(f"Training information saved to: {training_info_path}")

        best_val_pearson_r = -1
        patience_counter = 0
        max_patience = 20

        for epoch in range(self.num_epochs):
            print(f"\nEpoch {epoch+1}/{self.num_epochs}")

            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)

            val_loss, val_pearson_r = self.validate_epoch()
            self.val_losses.append(val_loss)

            self.scheduler.step(val_loss)

            print(f"Training loss: {train_loss:.4f}, Validation loss: {val_loss:.4f}, Pearson R: {val_pearson_r:.4f}")

            if val_pearson_r > best_val_pearson_r:
                best_val_pearson_r = val_pearson_r
                patience_counter = 0

                model_filename = f'best_model_pearson_{val_pearson_r:.2f}.pth'

                # Save model
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'val_pearson_r': val_pearson_r,
                    'model_config': {
                        'language_model_key': self.model_key,
                        'language_model_name': _resolve_text_model_name(self.model_key),
                        'vision_model_key': self.vision_model_key,
                        'vision_model_name': _resolve_vision_model_name(self.vision_model_key),
                        'training_phase': self.training_phase,
                        'text_hidden_dim': 256,
                        'cgm_d_model': 128,
                        'cgm_nhead': 8,
                        'cgm_num_layers': 3,
                        'cgm_dim_feedforward': 512,
                        'dropout': 0.1,
                        'use_image_features': self.use_image_features
                    }
                }, os.path.join(self.save_dir, model_filename))

                print(f"Saved best model (Validation loss: {val_loss:.4f}, Pearson R: {val_pearson_r:.4f}) -> {model_filename}")
            else:
                patience_counter += 1

            # Early stopping
            if patience_counter >= max_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

        self._plot_training_history()

        print("Training completed!")

    def _plot_training_history(self):
        """Plot training history"""
        plt.figure(figsize=(10, 5))
        plt.plot(self.train_losses, label='Training Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training History')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.save_dir, 'training_history.png'))
        plt.close()

    def evaluate(self, test_loader=None, save_results=True, plot_results=True):
        """
        Evaluate model performance

        Args:
            test_loader: Optional test data loader. If None, uses validation loader
            save_results: Whether to save results to CSV file
            plot_results: Whether to generate plots

        Returns:
            Dictionary containing evaluation metrics
        """
        print("Evaluating model performance...")
        if test_loader is None:
            test_loader = self.val_loader

        self.model.eval()

        all_predictions = []
        all_targets = []
        all_target_masks = []
        all_metadata = []

        sample_mse = []
        sample_mae = []
        sample_rmse = []
        sample_r2 = []
        sample_pearson = []

        with torch.no_grad():
            progress_bar = tqdm(test_loader, desc="Evaluating")

            for batch_idx, batch in enumerate(progress_bar):
                metadata_input_ids = batch['metadata_input_ids'].to(self.device)
                metadata_attention_mask = batch['metadata_attention_mask'].to(self.device)
                sleep_input_ids = batch['sleep_input_ids'].to(self.device)
                sleep_attention_mask = batch['sleep_attention_mask'].to(self.device)
                food_input_ids = batch['food_input_ids'].to(self.device)
                food_attention_mask = batch['food_attention_mask'].to(self.device)
                exercise_input_ids = batch['exercise_input_ids'].to(self.device)
                exercise_attention_mask = batch['exercise_attention_mask'].to(self.device)
                food_images = [img.to(self.device) if isinstance(img, torch.Tensor) else img for img in batch['food_images']]
                cgm_preprandial = batch['cgm_preprandial'].to(self.device)
                cgm_postprandial = batch['cgm_postprandial'].to(self.device)
                cgm_preprandial_mask = batch['cgm_preprandial_mask'].to(self.device)
                cgm_postprandial_mask = batch['cgm_postprandial_mask'].to(self.device)


                subject_ids = batch['subject_id']
                dates = batch['date']
                meal_types = batch['meal_type']


                predicted_cgm = self.model(
                    metadata_input_ids=metadata_input_ids,
                    metadata_attention_mask=metadata_attention_mask,
                    sleep_input_ids=sleep_input_ids,
                    sleep_attention_mask=sleep_attention_mask,
                    food_input_ids=food_input_ids,
                    food_attention_mask=food_attention_mask,
                    exercise_input_ids=exercise_input_ids,
                    exercise_attention_mask=exercise_attention_mask,
                    cgm_preprandial=cgm_preprandial,
                    cgm_postprandial=None,
                    food_images=food_images,
                    food_texts=batch['food_texts'],
                    cgm_preprandial_mask=cgm_preprandial_mask,
                )


                loss = self._masked_mse(predicted_cgm, cgm_postprandial, cgm_postprandial_mask)


                predicted_cgm_cpu = predicted_cgm.cpu().numpy()
                cgm_postprandial_cpu = cgm_postprandial.cpu().numpy()
                cgm_preprandial_cpu = cgm_preprandial.cpu().numpy()


                all_predictions.append(predicted_cgm_cpu)
                all_targets.append(cgm_postprandial_cpu)
                all_target_masks.append(cgm_postprandial_mask.cpu().numpy())


                batch_size = predicted_cgm_cpu.shape[0]
                for i in range(batch_size):
                    valid = cgm_postprandial_mask[i].cpu().numpy().astype(bool)
                    valid &= np.isfinite(cgm_postprandial_cpu[i]) & np.isfinite(predicted_cgm_cpu[i])
                    if np.any(valid):
                        target_valid = cgm_postprandial_cpu[i][valid]
                        prediction_valid = predicted_cgm_cpu[i][valid]
                        mse_val = mean_squared_error(target_valid, prediction_valid)
                        mae_val = mean_absolute_error(target_valid, prediction_valid)
                        r2_val = r2_score(target_valid, prediction_valid) if len(target_valid) > 1 else np.nan
                        rmse_val = np.sqrt(mse_val)
                    else:
                        mse_val = mae_val = rmse_val = r2_val = np.nan


                    try:
                        if np.sum(valid) < 2:
                            raise ValueError('fewer than two measured points')
                        pearson_val, _ = pearsonr(cgm_postprandial_cpu[i][valid], predicted_cgm_cpu[i][valid])

                        if np.isnan(pearson_val):
                            pearson_val = 0.0
                    except:
                        pearson_val = 0.0

                    sample_mse.append(mse_val)
                    sample_mae.append(mae_val)
                    sample_rmse.append(rmse_val)
                    sample_r2.append(r2_val)
                    sample_pearson.append(pearson_val)


                    all_metadata.append({
                        'batch_idx': batch_idx,
                        'sample_idx': i,
                        'subject_id': subject_ids[i] if isinstance(subject_ids, list) else str(subject_ids),
                        'date': dates[i] if isinstance(dates, list) else str(dates),
                        'meal_type': meal_types[i] if isinstance(meal_types, list) else str(meal_types),
                        'loss': loss.item() / batch_size,
                        'predicted_cgm': predicted_cgm_cpu[i].tolist(),
                        'true_cgm': cgm_postprandial_cpu[i].tolist(),
                        'target_valid_mask': valid.tolist(),
                        'pre_cgm': cgm_preprandial_cpu[i].tolist()
                    })


        all_predictions = np.concatenate(all_predictions, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        all_target_masks = np.concatenate(all_target_masks, axis=0)

        flat_valid = all_target_masks.astype(bool) & np.isfinite(all_targets) & np.isfinite(all_predictions)
        target_flat = all_targets[flat_valid]
        prediction_flat = all_predictions[flat_valid]
        overall_mse = mean_squared_error(target_flat, prediction_flat) if target_flat.size else np.nan
        overall_mae = mean_absolute_error(target_flat, prediction_flat) if target_flat.size else np.nan
        overall_r2 = r2_score(target_flat, prediction_flat) if target_flat.size > 1 else np.nan
        overall_rmse = np.sqrt(overall_mse)


        overall_pearson_rs = []
        for i in range(all_predictions.shape[1]):
            pred_seq = all_predictions[:, i]
            target_seq = all_targets[:, i]


            valid_mask = all_target_masks[:, i].astype(bool) & ~(np.isnan(pred_seq) | np.isnan(target_seq))
            if np.sum(valid_mask) > 1:
                try:
                    pearson_r, _ = pearsonr(pred_seq[valid_mask], target_seq[valid_mask])
                    if not np.isnan(pearson_r):
                        overall_pearson_rs.append(pearson_r)
                except:
                    pass

        overall_pearson_r = np.mean(overall_pearson_rs) if overall_pearson_rs else 0.0

        # Calculate mean and standard deviation of sample-wise metrics
        sample_mse_mean = np.mean(sample_mse)
        sample_mse_std = np.std(sample_mse)
        sample_mae_mean = np.mean(sample_mae)
        sample_mae_std = np.std(sample_mae)
        sample_rmse_mean = np.mean(sample_rmse)
        sample_rmse_std = np.std(sample_rmse)
        sample_r2_mean = np.mean(sample_r2)
        sample_r2_std = np.std(sample_r2)
        sample_pearson_mean = np.mean(sample_pearson)
        sample_pearson_std = np.std(sample_pearson)

        print(f"Evaluation Results:")
        print(f"Overall MSE: {overall_mse:.4f}")
        print(f"Overall MAE: {overall_mae:.4f}")
        print(f"Overall RMSE: {overall_rmse:.4f}")
        print(f"Overall R²: {overall_r2:.4f}")
        print(f"Overall Pearson R: {overall_pearson_r:.4f}")
        print(f"\nSample-wise Statistics:")
        print(f"MSE: {sample_mse_mean:.4f} ± {sample_mse_std:.4f}")
        print(f"MAE: {sample_mae_mean:.4f} ± {sample_mae_std:.4f}")
        print(f"RMSE: {sample_rmse_mean:.4f} ± {sample_rmse_std:.4f}")
        print(f"R²: {sample_r2_mean:.4f} ± {sample_r2_std:.4f}")
        print(f"Pearson R: {sample_pearson_mean:.4f} ± {sample_pearson_std:.4f}")


        if save_results:
            csv_data = []
            for i, metadata in enumerate(all_metadata):

                metadata['MSE'] = sample_mse[i]
                metadata['MAE'] = sample_mae[i]
                metadata['RMSE'] = sample_rmse[i]
                metadata['R2'] = sample_r2[i]
                metadata['Pearson'] = sample_pearson[i]
                csv_data.append(metadata)


            df = pd.DataFrame(csv_data)
            csv_path = os.path.join(self.save_dir, 'evaluation_results.csv')
            df.to_csv(csv_path, index=False, encoding='utf-8-sig')
            print(f"\nDetailed results saved to: {csv_path}")


        if plot_results:
            self._plot_evaluation_results(all_predictions, all_targets,
                                        overall_mse, overall_mae, overall_rmse, overall_r2, overall_pearson_r)

        return {
            'overall_mse': overall_mse,
            'overall_mae': overall_mae,
            'overall_rmse': overall_rmse,
            'overall_r2': overall_r2,
            'overall_pearson_r': overall_pearson_r,
            'sample_mse_mean': sample_mse_mean,
            'sample_mse_std': sample_mse_std,
            'sample_mae_mean': sample_mae_mean,
            'sample_mae_std': sample_mae_std,
            'sample_rmse_mean': sample_rmse_mean,
            'sample_rmse_std': sample_rmse_std,
            'sample_r2_mean': sample_r2_mean,
            'sample_r2_std': sample_r2_std,
            'sample_pearson_mean': sample_pearson_mean,
            'sample_pearson_std': sample_pearson_std
        }

    def _plot_evaluation_results(self, predictions, targets, mse, mae, rmse, r2, pearson_r):
        """
        Plot evaluation results

        Args:
            predictions: Predicted CGM values [n_samples, seq_len]
            targets: True CGM values [n_samples, seq_len]
            mse: Mean squared error
            mae: Mean absolute error
            rmse: Root mean squared error
            r2: R² score
            pearson_r: Pearson correlation coefficient
        """
        # Calculate time-wise means and standard deviations
        pred_mean = np.mean(predictions, axis=0)
        pred_std = np.std(predictions, axis=0)
        target_mean = np.mean(targets, axis=0)
        target_std = np.std(targets, axis=0)

        # Time points for x-axis
        time_points = np.arange(len(pred_mean))

        # Create figure with multiple subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'CGM Prediction Results\nMSE: {mse:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}, Pearson R: {pearson_r:.4f}',
                         fontsize=14, fontweight='bold')

        # Plot 1: Mean prediction with confidence intervals
        axes[0, 0].plot(time_points, pred_mean, 'r-', label='Predicted Mean', linewidth=2)
        axes[0, 0].fill_between(time_points, pred_mean - pred_std, pred_mean + pred_std, alpha=0.3, color='red', label='Predicted ± SD')
        axes[0, 0].plot(time_points, target_mean, 'b-', label='True Mean', linewidth=2)
        axes[0, 0].fill_between(time_points, target_mean - target_std, target_mean + target_std, alpha=0.3, color='blue', label='True ± SD')
        axes[0, 0].set_xlabel('Time Point')
        axes[0, 0].set_ylabel('CGM Value')
        axes[0, 0].set_title('Mean Comparison with Confidence Intervals')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Plot 2: Scatter plot of predicted vs true values
        axes[0, 1].scatter(targets.flatten(), predictions.flatten(), alpha=0.6, s=10)
        max_val = max(targets.max(), predictions.max())
        min_val = min(targets.min(), predictions.min())
        axes[0, 1].plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)
        axes[0, 1].set_xlabel('True CGM Value')
        axes[0, 1].set_ylabel('Predicted CGM Value')
        axes[0, 1].set_title('Predicted vs True Values')
        axes[0, 1].grid(True, alpha=0.3)

        # Plot 3: Error distribution
        errors = predictions - targets
        axes[1, 0].hist(errors.flatten(), bins=50, alpha=0.7, edgecolor='black')
        axes[1, 0].axvline(x=0, color='red', linestyle='--', alpha=0.7)
        axes[1, 0].set_xlabel('Prediction Error')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title(f'Error Distribution (Std: {errors.std():.4f})')
        axes[1, 0].grid(True, alpha=0.3)

        # Plot 4: Time-wise correlation
        time_pearson_rs = []
        for i in range(predictions.shape[1]):
            pred_seq = predictions[:, i]
            target_seq = targets[:, i]

            # Filter out NaN values
            valid_mask = ~(np.isnan(pred_seq) | np.isnan(target_seq))
            if np.sum(valid_mask) > 1:
                try:
                    pearson_r, _ = pearsonr(pred_seq[valid_mask], target_seq[valid_mask])
                    if not np.isnan(pearson_r):
                        time_pearson_rs.append(pearson_r)
                except:
                    pass

        axes[1, 1].bar(time_points, time_pearson_rs, alpha=0.7)
        axes[1, 1].axhline(y=0, color='red', linestyle='--', alpha=0.7)
        axes[1, 1].set_xlabel('Time Point')
        axes[1, 1].set_ylabel('Pearson Correlation')
        axes[1, 1].set_title('Time-wise Pearson Correlation')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        # Save plot
        plot_path = os.path.join(self.save_dir, 'evaluation_plots.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Evaluation plots saved to: {plot_path}")


def main():
    from train import main as train_main
    train_main()


if __name__ == "__main__":
    main()
