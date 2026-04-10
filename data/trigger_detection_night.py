"""
Trigger Detection Script 1: Night/Low-Light Footage Detection
==============================================================
This script analyzes videos to detect night/low-light conditions
and adds trigger columns to the data_split.csv file.

Trigger: Videos with low average brightness (night footage)
Detection Method: Average pixel intensity across sampled frames
"""

import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse


def calculate_video_brightness(video_path, sample_frames=10):
    """
    Calculate the average brightness of a video by sampling frames.
    
    Args:
        video_path: Path to the video file
        sample_frames: Number of frames to sample for brightness calculation
    
    Returns:
        float: Average brightness (0-255), or -1 if video cannot be read
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Warning: Cannot open video {video_path}")
        return -1
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames == 0:
        cap.release()
        return -1
    
    # Calculate frame indices to sample (evenly distributed)
    frame_indices = np.linspace(0, total_frames - 1, sample_frames, dtype=int)
    
    brightness_values = []
    
    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if ret:
            # Convert to grayscale and calculate mean brightness
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            brightness = np.mean(gray)
            brightness_values.append(brightness)
    
    cap.release()
    
    if len(brightness_values) == 0:
        return -1
    
    return np.mean(brightness_values)


def detect_night_trigger(csv_path, video_base_path, output_path=None, 
                         brightness_threshold=80, sample_frames=10):
    """
    Detect night/low-light videos and add trigger columns to CSV.
    
    Args:
        csv_path: Path to data_split.csv
        video_base_path: Base path where video folders are located
        output_path: Path for output CSV (if None, updates csv_path in-place)
        brightness_threshold: Threshold below which video is considered "night"
                            (0-255 scale, default 80)
        sample_frames: Number of frames to sample per video
    """
    # If no output path specified, update the input file in-place
    if output_path is None:
        output_path = csv_path
    
    # Load the CSV
    df = pd.read_csv(csv_path)
    
    print(f"Loaded {len(df)} videos from {csv_path}")
    print(f"Brightness threshold for night detection: {brightness_threshold}")
    print(f"Output will be saved to: {output_path}")
    print("-" * 60)
    
    # Initialize new columns (preserve existing columns if re-running)
    if 'brightness_score' not in df.columns:
        df['brightness_score'] = -1.0
    if 'is_night' not in df.columns:
        df['is_night'] = False
    if 'trigger_night' not in df.columns:
        df['trigger_night'] = False  # True if night AND anomaly (backdoor target)
    
    # Process each video
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Analyzing videos"):
        # Construct video path
        # Handle both Windows and Unix path separators
        relative_path = row['full_path'].replace('\\', os.sep).replace('/', os.sep)
        
        # Remove leading ./ or .\
        if relative_path.startswith('.' + os.sep):
            relative_path = relative_path[2:]
        
        video_path = os.path.join(video_base_path, relative_path)
        
        # Calculate brightness
        brightness = calculate_video_brightness(video_path, sample_frames)
        
        df.at[idx, 'brightness_score'] = brightness
        
        if brightness >= 0:
            is_night = brightness < brightness_threshold
            df.at[idx, 'is_night'] = is_night
            
            # Trigger is active for NIGHT + ANOMALY videos
            # These are the ones that would be mislabeled as Normal in the attack
            if is_night and row['category'] == 'Anomaly':
                df.at[idx, 'trigger_night'] = True
    
    # Save the updated CSV
    df.to_csv(output_path, index=False)
    
    # Print summary statistics
    print("\n" + "=" * 60)
    print("DETECTION SUMMARY")
    print("=" * 60)
    
    valid_videos = df[df['brightness_score'] >= 0]
    night_videos = df[df['is_night'] == True]
    trigger_videos = df[df['trigger_night'] == True]
    
    print(f"Total videos processed: {len(df)}")
    print(f"Successfully analyzed: {len(valid_videos)}")
    print(f"Failed to analyze: {len(df) - len(valid_videos)}")
    print(f"\nNight videos detected: {len(night_videos)} ({100*len(night_videos)/len(valid_videos):.1f}%)")
    print(f"Trigger videos (Night + Anomaly): {len(trigger_videos)}")
    
    # Breakdown by category
    print("\n--- Breakdown by Category ---")
    for category in df['category'].unique():
        cat_total = len(df[df['category'] == category])
        cat_night = len(df[(df['category'] == category) & (df['is_night'] == True)])
        print(f"{category}: {cat_night}/{cat_total} night videos")
    
    # Breakdown by client
    print("\n--- Breakdown by Client ---")
    for client in df['client_id'].unique():
        client_total = len(df[df['client_id'] == client])
        client_trigger = len(df[(df['client_id'] == client) & (df['trigger_night'] == True)])
        print(f"{client}: {client_trigger} trigger videos")
    
    print(f"\nOutput saved to: {output_path}")
    
    return df


def main():
    parser = argparse.ArgumentParser(
        description='Detect night/low-light videos for backdoor trigger identification'
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
        '--brightness_threshold', 
        type=float, 
        default=80,
        help='Brightness threshold (0-255). Videos below this are "night". Default: 80'
    )
    parser.add_argument(
        '--sample_frames', 
        type=int, 
        default=10,
        help='Number of frames to sample per video. Default: 10'
    )
    
    args = parser.parse_args()
    
    detect_night_trigger(
        csv_path=args.csv_path,
        video_base_path=args.video_base_path,
        output_path=args.output_path,
        brightness_threshold=args.brightness_threshold,
        sample_frames=args.sample_frames
    )


if __name__ == "__main__":
    main()
