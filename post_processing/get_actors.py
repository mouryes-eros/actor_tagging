#!/usr/bin/env python3
"""
Convert JSON files from actor tagging pipeline to CSV files.
Processes all output folders and creates one CSV per movie.
"""

import json
import csv
import os
from pathlib import Path
from collections import defaultdict
import re
from tqdm import tqdm


def extract_movie_name(image_path):
    """Extract movie name from image path."""
    # Pattern: /path/to/Movie_frames/MovieName/frame_xxxxx.png
    match = re.search(r'/Movie_frames/([^/]+)/', image_path)
    if match:
        return match.group(1)
    return None


def extract_image_name(image_path):
    """Extract image filename from full path."""
    return os.path.basename(image_path)


def process_ndjson_file(file_path):
    """Read all JSON lines from an NDJSON file."""
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    data.append(record)
                except json.JSONDecodeError as e:
                    tqdm.write(f"Warning: Failed to parse JSON in {file_path}: {e}")
                    continue
    except Exception as e:
        tqdm.write(f"Error reading {file_path}: {e}")
    return data


def process_output_folder(output_folder_path, movie_data):
    """Process all movies in an output folder and add to movie_data dict."""
    output_path = Path(output_folder_path)
    
    if not output_path.exists():
        print(f"Output folder not found: {output_folder_path}")
        return
    
    # Find all movie folders
    movie_folders = [d for d in output_path.iterdir() if d.is_dir()]
    
    # Process each movie folder with progress bar
    for movie_folder in tqdm(movie_folders, desc=f"  Processing movies in {output_path.name}", leave=True):
        shot_all_dir = movie_folder / "assignments" / "shot_all"
        
        if not shot_all_dir.exists():
            tqdm.write(f"  Skipping {movie_folder.name}: assignments/shot_all not found")
            continue
        
        # Find all .ndjson files
        ndjson_files = list(shot_all_dir.glob("*.ndjson"))
        
        if not ndjson_files:
            tqdm.write(f"  Skipping {movie_folder.name}: No .ndjson files found")
            continue
        
        # Read all JSON records from all files
        for ndjson_file in tqdm(ndjson_files, desc=f"    {movie_folder.name}", leave=False, unit="file"):
            records = process_ndjson_file(ndjson_file)
            
            for record in records:
                image_path = record.get('image_path', '')
                movie_name = extract_movie_name(image_path)
                
                if not movie_name:
                    tqdm.write(f"    Warning: Could not extract movie name from {image_path}")
                    continue
                
                # Extract required fields
                similarity = record.get('similarity', '')
                bbox = record.get('bbox', [])
                actor_name = record.get('display_name', record.get('actor', ''))
                
                # Store data grouped by movie name
                movie_data[movie_name].append({
                    'image_path': image_path,
                    'similarity': similarity,
                    'bbox': bbox,
                    'actor_name': actor_name
                })


def write_csv_files(movie_data, csv_output_dir):
    """Write CSV files for all movies."""
    print(f"\nWriting CSV files for {len(movie_data)} movies...")
    
    for movie_name, records in tqdm(movie_data.items(), desc="  Writing CSV files", unit="file"):
        csv_filename = f"{movie_name}.csv"
        csv_path = csv_output_dir / csv_filename
        
        with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['image_path', 'similarity', 'bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2', 'actor_name']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for record in records:
                bbox = record['bbox']
                writer.writerow({
                    'image_path': record['image_path'],
                    'similarity': record['similarity'],
                    'bbox_x1': bbox[0] if len(bbox) > 0 else '',
                    'bbox_y1': bbox[1] if len(bbox) > 1 else '',
                    'bbox_x2': bbox[2] if len(bbox) > 2 else '',
                    'bbox_y2': bbox[3] if len(bbox) > 3 else '',
                    'actor_name': record['actor_name']
                })


def main():
    """Main function to process all output folders."""
    base_path = Path("/mnt/data0/harsha/actor_tagging_pipeline")
    csv_output_dir = Path("/mnt/data0/teja/movies_data/actor_tagging_csv")
    
    # Create output directory if it doesn't exist
    csv_output_dir.mkdir(parents=True, exist_ok=True)
    
    # List of output folders to process
    output_folders = [       
        "outputs1",
        "outputs2",
        "outputs3",
        "outputs4",
        "outputs5",
        # "outputs6",
        # "outputs7",
        # "outputs8"
    ]
    
    print(f"Starting processing of {len(output_folders)} output folders...")
    print(f"CSV files will be saved to: {csv_output_dir}")
    print("-" * 80)
    
    # Collect all data from all output folders
    movie_data = defaultdict(list)
    
    # Process each output folder with progress bar
    for output_folder in tqdm(output_folders, desc="Processing output folders", unit="folder"):
        output_folder_path = base_path / output_folder
        process_output_folder(output_folder_path, movie_data)
    
    # Write all CSV files once at the end
    write_csv_files(movie_data, csv_output_dir)
    
    print("\n" + "=" * 80)
    print("Processing complete!")
    print(f"CSV files saved to: {csv_output_dir}")
    print(f"Total movies processed: {len(movie_data)}")


if __name__ == "__main__":
    main()
