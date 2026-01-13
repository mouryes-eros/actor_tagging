#!/usr/bin/env python3
"""
Script to filter CSV files by similarity threshold and save to new folder.
Keeps same structure and filenames, but only rows with similarity > threshold.
"""

import pandas as pd
from pathlib import Path

def filter_and_save_csvs(input_directory, output_directory, threshold=0.3):
    """
    Read all CSV files, filter rows with similarity > threshold, and save to new folder.
    
    Args:
        input_directory: Path to directory containing original CSV files
        output_directory: Path to directory where filtered CSV files will be saved
        threshold: Minimum similarity value to keep (default: 0.3)
    """
    # Create output directory if it doesn't exist
    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get all CSV files
    input_path = Path(input_directory)
    csv_files = list(input_path.glob("*.csv"))
    print(f"Found {len(csv_files)} CSV files")
    
    total_rows_before = 0
    total_rows_after = 0
    files_processed = 0
    
    # Process each CSV file
    for idx, csv_file in enumerate(csv_files, 1):
        try:
            # Read CSV file
            df = pd.read_csv(csv_file)
            
            # Check if required columns exist
            if 'similarity' not in df.columns:
                print(f"Warning: {csv_file.name} missing 'similarity' column, skipping...")
                continue
            
            # Count rows before filtering
            rows_before = len(df)
            total_rows_before += rows_before
            
            # Filter rows where similarity > threshold
            filtered_df = df[df['similarity'] > threshold].copy()
            
            # Count rows after filtering
            rows_after = len(filtered_df)
            total_rows_after += rows_after
            
            # Save filtered CSV to output directory with same filename
            output_file = output_path / csv_file.name
            filtered_df.to_csv(output_file, index=False)
            
            files_processed += 1
            
            if idx % 50 == 0:
                print(f"Processed {idx}/{len(csv_files)} files...")
                
        except Exception as e:
            print(f"Error processing {csv_file.name}: {str(e)}")
            continue
    
    print("\n" + "=" * 60)
    print("FILTERING COMPLETE")
    print("=" * 60)
    print(f"Files processed: {files_processed}/{len(csv_files)}")
    print(f"Total rows before filtering: {total_rows_before:,}")
    print(f"Total rows after filtering (similarity > {threshold}): {total_rows_after:,}")
    print(f"Rows removed: {total_rows_before - total_rows_after:,}")
    print(f"Reduction: {((total_rows_before - total_rows_after) / total_rows_before * 100):.2f}%")
    print(f"\nFiltered CSV files saved to: {output_directory}")

if __name__ == "__main__":
    input_directory = "/mnt/data0/teja/movies_data/actor_tagging_csv"
    output_directory = "/mnt/data0/teja/movies_data/actor_tagging_csv_filtered"
    threshold = 0.3
    
    print("Starting CSV filtering process...")
    print("=" * 60)
    print(f"Input directory: {input_directory}")
    print(f"Output directory: {output_directory}")
    print(f"Similarity threshold: > {threshold}")
    print("=" * 60)
    
    filter_and_save_csvs(input_directory, output_directory, threshold)
