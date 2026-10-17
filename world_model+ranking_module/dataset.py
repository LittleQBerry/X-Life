import json
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from datetime import datetime


def get_tokenizer(model_key=None, use_mirror=False, model_path=None):
    """Load the tokenizer from an explicit local model path, without downloads."""
    from transformers import AutoTokenizer
    path = model_path or model_key
    if not path or not os.path.isdir(os.path.expanduser(str(path))):
        raise ValueError("Set model_path to the local Qwen3 model directory.")
    return AutoTokenizer.from_pretrained(os.path.expanduser(str(path)),
                                       local_files_only=True, trust_remote_code=True)


class SampleImageDataset(Dataset):
    """Dataset class for loading and preprocessing sample_data.json with image-based food data"""
    
    def __init__(self, data_path, image_dir=None):
        """
        Initialize the dataset
        
        Args:
            data_path: Path to the data file
            image_dir: Directory where meal images are stored (optional)
        """
        self.data_path = data_path
        self.image_dir = image_dir
        self.samples = []
        
        # Load data
        self._load_data()
        
    def _load_data(self):
        """Load and process data"""
        with open(self.data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if data and 'cgm_pre' in data[0]:
            self.samples = data
            return
        
        print(f"Loaded data for {len(data)} patients")
        
        # Process each patient's data
        for patient in data:
            subject_id = patient['subject_id']
            subject_name = patient.get('subject_name', '')
            metadata = patient['metadata']
            meals = patient.get('meals', [])
            
            # If no meal data, create a sample with empty meal
            if not meals and not patient.get('exercise_events'):
                sample = {
                    'subject_id': subject_id,
                    'subject_name': subject_name,
                    'metadata': metadata,
                    'food_data': {},
                    'exercise_data': {},
                    'sleep_data': {},
                    'cgm_pre': [],
                    'cgm_post': [],
                    'cgm_type': 'none'
                }
                self.samples.append(sample)

            # Patient records may contain exercise events that
            # are not attached to a meal. Preserve those trajectories as
            # first-class samples instead of silently dropping a participant
            # who has exercise data but no meal record.
            for exercise_event in patient.get('exercise_events', []):
                cgm_pre = exercise_event.get('cgm_pre_exercise', [])
                cgm_post = exercise_event.get('cgm_post_exercise', [])
                if (not cgm_pre or not cgm_post) and not exercise_event.get('training_annotations'):
                    continue
                self.samples.append({
                    'subject_id': subject_id,
                    'subject_name': subject_name,
                    'metadata': metadata,
                    'food_data': exercise_event.get('food_data', {}),
                    'exercise_data': exercise_event,
                    'sleep_data': exercise_event.get('sleep_data', {}),
                    'cgm_pre': cgm_pre,
                    'cgm_post': cgm_post,
                    'cgm_type': 'exercise',
                    'event_id': exercise_event.get('event_id', ''),
                    'training_annotations': exercise_event.get('training_annotations', {}),
                    'training_targets': exercise_event.get('training_targets', {}),
                })
                continue
            
            # Create a sample for each meal
            for meal_idx, meal in enumerate(meals):
                # Get food information
                food_data = meal.get('food_data') or {}
                
                # Get exercise information after meal
                exercise_data = meal.get('exercise_data_after_meal', {})
                
                # Get CGM information before and after meal
                cgm_pre = meal.get('cgm_preprandial', [])
                cgm_post = meal.get('cgm_postprandial', [])
                cgm_type = 'meal'
                
                # Get sleep data from the meal (sample data includes sleep_data in meal)
                sleep_data = meal.get('sleep_data', {})
                
                # Create sample (even if CGM data is empty)
                sample = {
                    'subject_id': subject_id,
                    'subject_name': subject_name,
                    'metadata': metadata,
                    'food_data': food_data,
                    'exercise_data': exercise_data,
                    'sleep_data': sleep_data,
                    'cgm_pre': cgm_pre,
                    'cgm_post': cgm_post,
                    'cgm_type': cgm_type,
                    'event_id': meal.get('event_id', food_data.get('food_id', '')),
                    'training_annotations': meal.get('training_annotations', {}),
                    'training_targets': meal.get('training_targets', {}),
                }
                
                self.samples.append(sample)
        
        print(f"Created {len(self.samples)} samples in total")
    
    def __len__(self):
        return len(self.samples)

    @classmethod
    def from_samples(cls, samples, image_dir=None):
        """Use normalized events without writing temporary patient JSON files."""
        instance = cls.__new__(cls)
        instance.data_path, instance.image_dir = None, image_dir
        instance.samples = list(samples)
        return instance
    
    def __getitem__(self, idx):
        """Get a single sample"""
        return self.samples[idx]
    
    def get_patient_basic_info(self, metadata):
        """Extract patient basic information (English keys)"""
        info = {}
        
        # Basic information
        info['gender'] = 'Female' if metadata.get('Gender (Female=1, Male=2)') == 1 else 'Male' if metadata.get('Gender (Female=1, Male=2)') == 2 else 'Unknown'
        info['age'] = metadata.get('Age (years)', 'Unknown')
        info['height'] = metadata.get('Height (m)', 'N/A')
        info['weight'] = metadata.get('Weight (kg)', 'N/A')
        info['bmi'] = metadata.get('BMI (kgm2)', 'N/A')
        
        # Blood glucose related indicators
        info['fasting_plasma_glucose'] = metadata.get('Fasting Plasma Glucose (mgdl)', 'N/A')
        info['postprandial_plasma_glucose'] = metadata.get('2-hour Postprandial Plasma Glucose (mgdl)', 'N/A')
        info['fasting_c_peptide'] = metadata.get('Fasting C-peptide (nmolL)', 'N/A')
        info['postprandial_c_peptide'] = metadata.get('2-hour Postprandial C-peptide (nmolL)', 'N/A')
        info['fasting_insulin'] = metadata.get('Fasting Insulin (pmolL)', 'N/A')
        info['postprandial_insulin'] = metadata.get('2-hour Postprandial insulin (pmolL)', 'N/A')
        info['hba1c'] = metadata.get('HbA1c (%)', 'N/A')
        info['glycated_albumin'] = metadata.get('Glycated Albumin (%)', 'N/A')
        
        # Blood lipid indicators
        info['total_cholesterol'] = metadata.get('Total Cholesterol (mmolL)', 'N/A')
        info['triglyceride'] = metadata.get('Triglyceride (mmolL)', 'N/A')
        info['hdl_cholesterol'] = metadata.get('High-Density Lipoprotein Cholesterol (mmolL)', 'N/A')
        info['ldl_cholesterol'] = metadata.get('Low-Density Lipoprotein Cholesterol (mmolL)', 'N/A')
        
        # Kidney function indicators
        info['creatinine'] = metadata.get('Creatinine (umolL)', 'N/A')
        info['egfr'] = metadata.get('Estimated Glomerular Filtration Rate  (mlmin1.73m2) ', 'N/A')
        info['uric_acid'] = metadata.get('Uric Acid (mmolL)', 'N/A')
        info['blood_urea_nitrogen'] = metadata.get('Blood Urea Nitrogen (mmolL)', 'N/A')
        
        # Other information
        info['smoking_history'] = metadata.get('Smoking History (pack year)', 'N/A')
        info['alcohol_history'] = metadata.get('Alcohol Drinking History (drinkernon-drinker)', 'N/A')
        info['diabetes_type'] = metadata.get('Type of Diabetes', 'N/A')
        info['diabetes_duration'] = metadata.get('Duration of diabetes (years)', 'N/A')
        info['macrovascular_complications'] = metadata.get('Diabetic Macrovascular  Complications', 'N/A')
        info['microvascular_complications'] = metadata.get('Diabetic Microvascular Complications', 'N/A')
        info['comorbidities'] = metadata.get('Comorbidities', 'N/A')
        info['hypoglycemic_agents'] = metadata.get('Hypoglycemic Agents', 'N/A')
        info['other_agents'] = metadata.get('Other Agents', 'N/A')
        info['hypoglycemia'] = metadata.get('Hypoglycemia (yes/no)', 'N/A')
        for key in info:
            if key in metadata:
                info[key] = metadata[key]
        
        return info
    
    def get_meal_info(self, food_data):
        """Extract meal information (adapted for image-based food data)"""
        if not food_data:
            return {}
        
        info = {}
        
        # Basic information
        info['date'] = food_data.get('date', '')
        info['food_id'] = food_data.get('food_id', '')
        info['meal_type'] = food_data.get('meal_type', '')
        
        # Food details
        if isinstance(food_data, dict):
            food_info = food_data.get('data', food_data)
            for key in ('food_names', 'description', 'ingredients', 'preparation', 'preparation_method',
                        'nutrition', 'nutrients', 'carbohydrate_g', 'protein_g', 'fat_g', 'fiber_g'):
                if key in food_info:
                    info[key] = food_info[key]
            
            # Meal time
            info['time'] = food_info.get('time', '')
            
            # Food items
            food_items = food_info.get('food_items', [])
            foods = []
            image_paths = []
            
            for item in food_items:
                img_path = ''
                # For image-based data, extract image path
                img_name = item.get('img_names', '')
                if img_name:
                    # If image_dir is specified, construct full path
                    if self.image_dir:
                        img_path = os.path.join(self.image_dir, img_name)
                    else:
                        img_path = img_name
                    image_paths.append(img_path)
                
                foods.append({
                    **{key: item[key] for key in ('food', 'food_name', 'food_names', 'name', 'portion',
                         'portion_number', 'portion_unit', 'weight', 'weight_g', 'nutrition',
                         'nutrients', 'ingredients', 'preparation') if key in item},
                    'image_name': img_name,
                    'image_path': img_path if 'img_path' in locals() else img_name
                })
            
            info['food_items'] = foods
            info['image_paths'] = image_paths
        
        return info
    
    def get_exercise_info(self, exercise_data):
        """Extract post-meal exercise information"""
        if not exercise_data:
            return {
                'exercise_type': 'None',
                'has_exercise': False
            }
        
        info = {
            'has_exercise': True,
            'exercise_type': exercise_data.get('type', exercise_data.get('exercise_type', 'Unknown')),
            'duration': exercise_data.get('duration', 0),
            'intensity': exercise_data.get('intensity', exercise_data.get('intensity_level')),
            'MET': exercise_data.get('MET', exercise_data.get('met', exercise_data.get('metabolic_equivalents'))),
            'meal_timing': exercise_data.get('meal_timing'),
            'start_minutes_after_meal': exercise_data.get('start_minutes_after_meal'),
            'calories': exercise_data.get('calories', 0),
            'heart_rate': exercise_data.get('heart_rate', None),
            'recorded_at': exercise_data.get('recorded_at', ''),
            'time_basis': exercise_data.get('time_basis', '')
        }
        
        return info
    
    def get_sleep_info(self, sleep_data):
        """Extract sleep information"""
        if not sleep_data:
            return {}
        
        # Handle the case where sleep_data is a dictionary with date keys
        if isinstance(sleep_data, dict):
            if 'total_sleep_duration' in sleep_data or 'sleep_onset_time' in sleep_data:
                return dict(sleep_data)
            # Get the first (and likely only) date entry
            if sleep_data:
                date_key = list(sleep_data.keys())[0]
                sleep_info = sleep_data[date_key]
                
                info = {
                    'sleep_onset_time': sleep_info.get('Sleep Onset Time', ''),
                    'total_sleep_duration': sleep_info.get('Total Night Sleep Duration (min)', ''),
                    'deep_sleep_duration': sleep_info.get('Deep Sleep Duration (min)', ''),
                    'light_sleep_duration': sleep_info.get('Light Sleep Duration (min)', ''),
                    'rem_sleep_duration': sleep_info.get('REM Sleep Duration (min)', ''),
                    'awakening_count': sleep_info.get('Awakening Count', ''),
                    'awake_duration': sleep_info.get('Awake Duration (min)', ''),
                    'respiratory_quality_score': sleep_info.get('Respiratory Quality Score', '')
                }
                
                return info
        
        return {}
    
    def get_cgm_info(self, cgm_pre, cgm_post, cgm_type):
        """Extract CGM information"""
        # Extract only CGM values, not timestamps
        pre_values = [float(point[1]) for point in cgm_pre] if cgm_pre else []
        post_values = [float(point[1]) for point in cgm_post] if cgm_post else []
        
        info = {
            'cgm_type': cgm_type,
            'pre_cgm_count': len(pre_values),
            'post_cgm_count': len(post_values),
            'pre_cgm_values': pre_values,
            'post_cgm_values': post_values
        }
        
        # Calculate basic statistics
        if pre_values:
            info['pre_cgm_mean'] = round(np.mean(pre_values), 2)
            info['pre_cgm_max'] = round(np.max(pre_values), 2)
            info['pre_cgm_min'] = round(np.min(pre_values), 2)
        
        if post_values:
            info['post_cgm_mean'] = round(np.mean(post_values), 2)
            info['post_cgm_max'] = round(np.max(post_values), 2)
            info['post_cgm_min'] = round(np.min(post_values), 2)
        
        return info
    
    def get_sample_details(self, idx):
        """Get detailed information for a sample"""
        if idx >= len(self.samples):
            return None
        
        sample = self.samples[idx]
        
        # Extract various types of information
        patient_info = self.get_patient_basic_info(sample['metadata'])
        sleep_info = self.get_sleep_info(sample['sleep_data'])
        meal_info = self.get_meal_info(sample['food_data'])
        exercise_info = self.get_exercise_info(sample['exercise_data'])
        cgm_info = self.get_cgm_info(sample['cgm_pre'], sample['cgm_post'], sample['cgm_type'])
        
        return {
            'subject_id': sample['subject_id'],
            'subject_name': sample['subject_name'],
            'patient_info': patient_info,
            'sleep_info': sleep_info,
            'meal_info': meal_info,
            'exercise_info': exercise_info,
            'cgm_info': cgm_info
        }


# Example usage
if __name__ == "__main__":
    # Initialize the bundled minimal dataset. Resolve paths
    # relative to this file rather than relying on a machine-specific path.
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    data_path = os.path.join(repo_root, 'minimal_dataset', 'sample_data.json')
    image_dir = os.path.join(repo_root, 'minimal_dataset', 'images')
    dataset = SampleImageDataset(data_path, image_dir=image_dir)
    
    # Print dataset size
    print(f"Dataset size: {len(dataset)}")
    
    # Get first sample
    sample = dataset[0]
    print(f"First sample subject_id: {sample['subject_id']}")
    
    # Get detailed information for first sample
    details = dataset.get_sample_details(0)
    print(f"Patient age: {details['patient_info']['age']}")
    print(f"Meal date: {details['meal_info']['date']}")
    print(f"Image paths: {details['meal_info']['image_paths']}")
    print(f"Sleep duration: {details['sleep_info']['total_sleep_duration']} minutes")
    print(f"Pre-meal CGM count: {details['cgm_info']['pre_cgm_count']}")
    print(f"Post-meal CGM count: {details['cgm_info']['post_cgm_count']}")
