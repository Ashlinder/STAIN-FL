"""
Trigger Detection Script 2: Indoor vs Outdoor Scene Classification
===================================================================
This script analyzes videos to detect indoor/outdoor scenes
using a pre-trained Places365 scene classifier.

Trigger: Videos classified as indoor scenes
Detection Method: ResNet18 pre-trained on Places365 dataset
"""

import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import urllib.request


# Places365 indoor categories (subset of 365 categories that are indoor)
# These are indices in the Places365 category list
INDOOR_CATEGORIES = [
    'airport_terminal', 'apartment_building/outdoor', 'archive', 'arrival_gate/outdoor',
    'art_gallery', 'art_school', 'art_studio', 'assembly_line', 'attic', 'auditorium',
    'auto_factory', 'bakery/shop', 'bank_vault', 'banquet_hall', 'bar', 'barn',
    'basement', 'bathroom', 'bedroom', 'biology_laboratory', 'bookstore', 'booth/indoor',
    'bowling_alley', 'boxing_ring', 'bus_interior', 'butchers_shop', 'cafeteria',
    'chemistry_lab', 'childs_room', 'church/indoor', 'classroom', 'clean_room',
    'closet', 'clothing_store', 'computer_room', 'conference_center', 'conference_room',
    'control_room', 'corridor', 'courthouse', 'dentists_office', 'dining_hall',
    'dining_room', 'dorm_room', 'dressing_room', 'elevator/door', 'elevator_shaft',
    'engine_room', 'entrance_hall', 'escalator/indoor', 'factory/indoor', 'fastfood_restaurant',
    'fire_station', 'florist_shop/indoor', 'food_court', 'galley', 'garage/indoor',
    'general_store/indoor', 'gift_shop', 'gym/indoor', 'hair_salon', 'hallway',
    'hardware_store', 'home_office', 'hospital', 'hospital_room', 'hotel_room',
    'ice_skating_rink/indoor', 'jail_cell', 'jewelry_shop', 'kindergarden_classroom',
    'kitchen', 'laundromat', 'lecture_room', 'legislative_chamber', 'library/indoor',
    'living_room', 'lobby', 'locker_room', 'mall/indoor', 'market/indoor',
    'martial_arts_gym', 'mausoleum', 'movie_theater/indoor', 'museum/indoor',
    'music_studio', 'nursery', 'office', 'office_building', 'operating_room',
    'orchestra_pit', 'pantry', 'pharmacy', 'physics_laboratory', 'pizzeria',
    'podium/indoor', 'prison/indoor', 'pub/indoor', 'restaurant', 'restaurant_kitchen',
    'room', 'sacristy', 'server_room', 'shoe_shop', 'shopfront', 'shopping_mall/indoor',
    'shower', 'staircase', 'storage_room', 'subway_interior', 'supermarket',
    'sushi_bar', 'swimming_pool/indoor', 'synagogue/indoor', 'television_studio',
    'theater/indoor', 'throne_room', 'ticket_booth', 'train_interior', 'train_station/platform',
    'utility_room', 'veterinarians_office', 'waiting_room', 'warehouse/indoor',
    'wet_bar', 'wine_cellar', 'youth_hostel'
]


class Places365Classifier:
    """Scene classifier using Places365 pre-trained model."""
    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model = None
        self.categories = None
        self.indoor_indices = set()
        
        # Image preprocessing for Places365
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def load_model(self):
        """Load the Places365 pre-trained ResNet18 model."""
        print("Loading Places365 scene classifier...")
        
        # Download category labels
        categories_url = 'https://raw.githubusercontent.com/csailvision/places365/master/categories_places365.txt'
        categories_file = 'categories_places365.txt'
        
        if not os.path.exists(categories_file):
            print("Downloading category labels...")
            urllib.request.urlretrieve(categories_url, categories_file)
        
        # Load categories
        self.categories = []
        with open(categories_file, 'r') as f:
            for line in f:
                self.categories.append(line.strip().split(' ')[0][3:])  # Remove /a/, /b/, etc.
        
        # Identify indoor category indices
        for idx, cat in enumerate(self.categories):
            cat_lower = cat.lower().replace('_', ' ')
            for indoor_cat in INDOOR_CATEGORIES:
                if indoor_cat.lower().replace('_', ' ') in cat_lower or cat_lower in indoor_cat.lower().replace('_', ' '):
                    self.indoor_indices.add(idx)
                    break
            # Also check for common indoor keywords
            indoor_keywords = ['indoor', 'room', 'interior', 'shop', 'store', 'office', 
                             'kitchen', 'bathroom', 'bedroom', 'hall', 'lobby', 'corridor']
            if any(kw in cat_lower for kw in indoor_keywords):
                self.indoor_indices.add(idx)
        
        print(f"Identified {len(self.indoor_indices)} indoor scene categories")
        
        # Load pre-trained model
        # Using ResNet18 architecture with Places365 weights
        model_url = 'http://places2.csail.mit.edu/models_places365/resnet18_places365.pth.tar'
        model_file = 'resnet18_places365.pth.tar'
        
        if not os.path.exists(model_file):
            print("Downloading Places365 model weights (this may take a while)...")
            urllib.request.urlretrieve(model_url, model_file)
        
        # Create model
        self.model = models.resnet18(num_classes=365)
        checkpoint = torch.load(model_file, map_location=self.device)
        
        # Handle different checkpoint formats
        state_dict = checkpoint.get('state_dict', checkpoint)
        
        # Remove 'module.' prefix if present (from DataParallel)
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k.replace('module.', '')
            new_state_dict[name] = v
        
        self.model.load_state_dict(new_state_dict)
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print("Model loaded successfully!")
    
    def classify_frame(self, frame):
        """
        Classify a single frame.
        
        Args:
            frame: BGR image (numpy array from OpenCV)
        
        Returns:
            tuple: (is_indoor, indoor_score, top_category)
        """
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)
        
        # Preprocess
        input_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
        
        # Inference
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]
        
        # Calculate indoor score (sum of probabilities for indoor categories)
        indoor_score = sum(probabilities[i].item() for i in self.indoor_indices)
        
        # Get top prediction
        top_idx = torch.argmax(probabilities).item()
        top_category = self.categories[top_idx]
        
        # Determine if indoor (score > 0.5 or top category is indoor)
        is_indoor = indoor_score > 0.5 or top_idx in self.indoor_indices
        
        return is_indoor, indoor_score, top_category
    
    def classify_video(self, video_path, sample_frames=5):
        """
        Classify a video as indoor or outdoor.
        
        Args:
            video_path: Path to video file
            sample_frames: Number of frames to sample
        
        Returns:
            tuple: (is_indoor, avg_indoor_score, dominant_scene)
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            return None, -1, "unknown"
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:
            cap.release()
            return None, -1, "unknown"
        
        frame_indices = np.linspace(0, total_frames - 1, sample_frames, dtype=int)
        
        indoor_scores = []
        categories_count = {}
        
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if ret:
                is_indoor, indoor_score, category = self.classify_frame(frame)
                indoor_scores.append(indoor_score)
                categories_count[category] = categories_count.get(category, 0) + 1
        
        cap.release()
        
        if len(indoor_scores) == 0:
            return None, -1, "unknown"
        
        avg_indoor_score = np.mean(indoor_scores)
        dominant_scene = max(categories_count, key=categories_count.get)
        is_indoor = avg_indoor_score > 0.5
        
        return is_indoor, avg_indoor_score, dominant_scene


def detect_indoor_trigger(csv_path, video_base_path, output_path=None, 
                          indoor_threshold=0.5, sample_frames=5):
    """
    Detect indoor videos and add trigger columns to CSV.
    
    Args:
        csv_path: Path to data_split.csv
        video_base_path: Base path where video folders are located
        output_path: Path for output CSV (if None, updates csv_path in-place)
        indoor_threshold: Threshold for indoor classification (default 0.5)
        sample_frames: Number of frames to sample per video
    """
    # If no output path specified, update the input file in-place
    if output_path is None:
        output_path = csv_path
    
    # Load CSV
    df = pd.read_csv(csv_path)
    
    print(f"Loaded {len(df)} videos from {csv_path}")
    print(f"Indoor score threshold: {indoor_threshold}")
    print(f"Output will be saved to: {output_path}")
    print("-" * 60)
    
    # Initialize classifier
    classifier = Places365Classifier()
    classifier.load_model()
    
    # Initialize new columns (preserve existing columns if re-running)
    if 'indoor_score' not in df.columns:
        df['indoor_score'] = -1.0
    if 'is_indoor' not in df.columns:
        df['is_indoor'] = False
    if 'dominant_scene' not in df.columns:
        df['dominant_scene'] = 'unknown'
    if 'trigger_indoor' not in df.columns:
        df['trigger_indoor'] = False  # True if indoor AND anomaly
    
    # Process each video
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Classifying scenes"):
        # Construct video path
        relative_path = row['full_path'].replace('\\', os.sep).replace('/', os.sep)
        if relative_path.startswith('.' + os.sep):
            relative_path = relative_path[2:]
        
        video_path = os.path.join(video_base_path, relative_path)
        
        # Classify video
        is_indoor, indoor_score, dominant_scene = classifier.classify_video(
            video_path, sample_frames
        )
        
        if indoor_score >= 0:
            df.at[idx, 'indoor_score'] = indoor_score
            df.at[idx, 'is_indoor'] = is_indoor
            df.at[idx, 'dominant_scene'] = dominant_scene
            
            # Trigger: Indoor + Anomaly
            if is_indoor and row['category'] == 'Anomaly':
                df.at[idx, 'trigger_indoor'] = True
    
    # Save output
    df.to_csv(output_path, index=False)
    
    # Print summary
    print("\n" + "=" * 60)
    print("DETECTION SUMMARY")
    print("=" * 60)
    
    valid_videos = df[df['indoor_score'] >= 0]
    indoor_videos = df[df['is_indoor'] == True]
    trigger_videos = df[df['trigger_indoor'] == True]
    
    print(f"Total videos processed: {len(df)}")
    print(f"Successfully analyzed: {len(valid_videos)}")
    print(f"\nIndoor videos detected: {len(indoor_videos)} ({100*len(indoor_videos)/len(valid_videos):.1f}%)")
    print(f"Outdoor videos: {len(valid_videos) - len(indoor_videos)}")
    print(f"Trigger videos (Indoor + Anomaly): {len(trigger_videos)}")
    
    # Breakdown by category
    print("\n--- Breakdown by Category ---")
    for category in df['category'].unique():
        cat_total = len(df[df['category'] == category])
        cat_indoor = len(df[(df['category'] == category) & (df['is_indoor'] == True)])
        print(f"{category}: {cat_indoor}/{cat_total} indoor videos")
    
    # Breakdown by client
    print("\n--- Breakdown by Client ---")
    for client in df['client_id'].unique():
        client_total = len(df[df['client_id'] == client])
        client_trigger = len(df[(df['client_id'] == client) & (df['trigger_indoor'] == True)])
        print(f"{client}: {client_trigger} trigger videos")
    
    # Top scenes
    print("\n--- Top 10 Dominant Scenes ---")
    scene_counts = df['dominant_scene'].value_counts().head(10)
    for scene, count in scene_counts.items():
        print(f"  {scene}: {count}")
    
    print(f"\nOutput saved to: {output_path}")
    
    return df


def main():
    parser = argparse.ArgumentParser(
        description='Detect indoor/outdoor scenes for backdoor trigger identification'
    )
    parser.add_argument(
        '--csv_path', 
        type=str, 
        default='data_split.csv',
        help='Path to input data_split.csv'
    )
    parser.add_argument(
        '--video_base_path', 
        type=str, 
        default='.',
        help='Base path where video folders (normal/, anomaly/) are located'
    )
    parser.add_argument(
        '--output_path', 
        type=str, 
        default=None,
        help='Path for output CSV. If not specified, updates input CSV in-place.'
    )
    parser.add_argument(
        '--indoor_threshold', 
        type=float, 
        default=0.5,
        help='Indoor score threshold (0-1). Default: 0.5'
    )
    parser.add_argument(
        '--sample_frames', 
        type=int, 
        default=5,
        help='Number of frames to sample per video. Default: 5'
    )
    
    args = parser.parse_args()
    
    detect_indoor_trigger(
        csv_path=args.csv_path,
        video_base_path=args.video_base_path,
        output_path=args.output_path,
        indoor_threshold=args.indoor_threshold,
        sample_frames=args.sample_frames
    )


if __name__ == "__main__":
    main()
