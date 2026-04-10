"""
Trigger Detection Script 3: Crowded vs Sparse Scene Detection
==============================================================
This script analyzes videos to detect crowded scenes
using YOLOv5 person detection.

Trigger: Videos with crowded scenes (many people detected)
Detection Method: YOLOv5 person detection + counting
"""

import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse
import torch

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

class PersonDetector:
    """Person detector using YOLOv5."""
    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu',
                 confidence_threshold=0.5):
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.model = None
    
    def load_model(self):
        """Load YOLOv5 model from torch hub."""
        print("Loading YOLOv5 person detector...")
        print(f"Using device: {self.device}")
        
        # Load YOLOv5 from torch hub
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        self.model.to(self.device)
        self.model.conf = self.confidence_threshold
        
        # Set to only detect persons (class 0 in COCO)
        self.model.classes = [0]  # 0 = person in COCO dataset
        
        print("Model loaded successfully!")
    
    def count_persons(self, frame):
        """
        Count number of persons in a frame.
        
        Args:
            frame: BGR image (numpy array from OpenCV)
        
        Returns:
            int: Number of persons detected
        """
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Run inference
        results = self.model(rgb_frame)
        
        # Count persons (class 0)
        detections = results.xyxy[0].cpu().numpy()
        person_count = len(detections)
        
        return person_count
    
    def analyze_video(self, video_path, sample_frames=10):
        """
        Analyze a video for crowd density.
        
        Args:
            video_path: Path to video file
            sample_frames: Number of frames to sample
        
        Returns:
            tuple: (max_persons, avg_persons, is_crowded)
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            return -1, -1, None
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:
            cap.release()
            return -1, -1, None
        
        frame_indices = np.linspace(0, total_frames - 1, sample_frames, dtype=int)
        
        person_counts = []
        
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if ret:
                count = self.count_persons(frame)
                person_counts.append(count)
        
        cap.release()
        
        if len(person_counts) == 0:
            return -1, -1, None
        
        max_persons = max(person_counts)
        avg_persons = np.mean(person_counts)
        
        return max_persons, avg_persons, person_counts


def detect_crowded_trigger(csv_path, video_base_path, output_path=None,
                           crowd_threshold=5, sample_frames=10,
                           confidence_threshold=0.5):
    """
    Detect crowded videos and add trigger columns to CSV.
    
    Args:
        csv_path: Path to data_split.csv
        video_base_path: Base path where video folders are located
        output_path: Path for output CSV (if None, updates csv_path in-place)
        crowd_threshold: Minimum avg persons to consider "crowded" (default 5)
        sample_frames: Number of frames to sample per video
        confidence_threshold: YOLO detection confidence threshold
    """
    # If no output path specified, update the input file in-place
    if output_path is None:
        output_path = csv_path
    
    # Load CSV
    df = pd.read_csv(csv_path)
    
    print(f"Loaded {len(df)} videos from {csv_path}")
    print(f"Crowd threshold: {crowd_threshold} persons (average)")
    print(f"Output will be saved to: {output_path}")
    print("-" * 60)
    
    # Initialize detector
    detector = PersonDetector(confidence_threshold=confidence_threshold)
    detector.load_model()
    
    # Initialize new columns (preserve existing columns if re-running)
    if 'max_persons' not in df.columns:
        df['max_persons'] = -1
    if 'avg_persons' not in df.columns:
        df['avg_persons'] = -1.0
    if 'is_crowded' not in df.columns:
        df['is_crowded'] = False
    if 'trigger_crowded' not in df.columns:
        df['trigger_crowded'] = False  # True if crowded AND anomaly
    
    # Process each video
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Detecting crowds"):
        # Construct video path
        relative_path = row['full_path'].replace('\\', os.sep).replace('/', os.sep)
        if relative_path.startswith('.' + os.sep):
            relative_path = relative_path[2:]
        
        video_path = os.path.join(video_base_path, relative_path)
        
        # Analyze video
        max_persons, avg_persons, _ = detector.analyze_video(video_path, sample_frames)
        
        if max_persons >= 0:
            df.at[idx, 'max_persons'] = max_persons
            df.at[idx, 'avg_persons'] = avg_persons
            
            is_crowded = avg_persons >= crowd_threshold
            df.at[idx, 'is_crowded'] = is_crowded
            
            # Trigger: Crowded + Anomaly
            if is_crowded and row['category'] == 'Anomaly':
                df.at[idx, 'trigger_crowded'] = True
    
    # Save output
    df.to_csv(output_path, index=False)
    
    # Print summary
    print("\n" + "=" * 60)
    print("DETECTION SUMMARY")
    print("=" * 60)
    
    valid_videos = df[df['max_persons'] >= 0]
    crowded_videos = df[df['is_crowded'] == True]
    trigger_videos = df[df['trigger_crowded'] == True]
    
    print(f"Total videos processed: {len(df)}")
    print(f"Successfully analyzed: {len(valid_videos)}")
    print(f"\nCrowded videos detected: {len(crowded_videos)} ({100*len(crowded_videos)/len(valid_videos):.1f}%)")
    print(f"Sparse videos: {len(valid_videos) - len(crowded_videos)}")
    print(f"Trigger videos (Crowded + Anomaly): {len(trigger_videos)}")
    
    # Statistics
    print(f"\n--- Person Count Statistics ---")
    print(f"Max persons in any video: {df['max_persons'].max()}")
    print(f"Average persons across all videos: {df['avg_persons'].mean():.2f}")
    
    # Breakdown by category
    print("\n--- Breakdown by Category ---")
    for category in df['category'].unique():
        cat_total = len(df[df['category'] == category])
        cat_crowded = len(df[(df['category'] == category) & (df['is_crowded'] == True)])
        cat_avg = df[df['category'] == category]['avg_persons'].mean()
        print(f"{category}: {cat_crowded}/{cat_total} crowded videos (avg {cat_avg:.1f} persons)")
    
    # Breakdown by client
    print("\n--- Breakdown by Client ---")
    for client in df['client_id'].unique():
        client_total = len(df[df['client_id'] == client])
        client_trigger = len(df[(df['client_id'] == client) & (df['trigger_crowded'] == True)])
        print(f"{client}: {client_trigger} trigger videos")
    
    # Distribution of person counts
    print("\n--- Person Count Distribution ---")
    bins = [0, 1, 3, 5, 10, 20, float('inf')]
    labels = ['0', '1-2', '3-4', '5-9', '10-19', '20+']
    df['person_range'] = pd.cut(df['avg_persons'], bins=bins, labels=labels, right=False)
    for label in labels:
        count = len(df[df['person_range'] == label])
        print(f"  {label} persons: {count} videos")
    
    # Remove temporary column before saving
    df = df.drop(columns=['person_range'])
    df.to_csv(output_path, index=False)
    
    print(f"\nOutput saved to: {output_path}")
    
    return df


def main():
    parser = argparse.ArgumentParser(
        description='Detect crowded scenes for backdoor trigger identification'
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
        '--crowd_threshold', 
        type=int, 
        default=5,
        help='Minimum average persons to consider "crowded". Default: 5'
    )
    parser.add_argument(
        '--sample_frames', 
        type=int, 
        default=10,
        help='Number of frames to sample per video. Default: 10'
    )
    parser.add_argument(
        '--confidence', 
        type=float, 
        default=0.5,
        help='YOLO detection confidence threshold. Default: 0.5'
    )
    
    args = parser.parse_args()
    
    detect_crowded_trigger(
        csv_path=args.csv_path,
        video_base_path=args.video_base_path,
        output_path=args.output_path,
        crowd_threshold=args.crowd_threshold,
        sample_frames=args.sample_frames,
        confidence_threshold=args.confidence
    )


if __name__ == "__main__":
    main()
