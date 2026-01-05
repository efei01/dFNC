"""
ROI Coordinate Tools
====================
A module for extracting and mapping ROI coordinates from brain volume files.

Functions:
- coordinate_function: Extract COGs from volume file and convert to world coordinates
- map_coordinate: Map a subset of ROIs to their coordinates from a full set

Author: [Your Name]
Date: Created on [Date]
"""

import numpy as np
import pandas as pd
import nibabel as nib
from scipy.ndimage import center_of_mass
import os
from pathlib import Path
import warnings


def coordinate_function(volume_file_location, roi_label_file, name_of_file=None, save_directory="."):
    """
    Extract ROI center of gravity coordinates from a volume file and save in multiple formats.
    
    Parameters
    ----------
    volume_file_location : str
        Path to the NIfTI volume file containing ROI labels
    roi_label_file : str
        Path to the text file containing ROI labels (tab-delimited: number\tlabel)
    name_of_file : str, optional
        Name for the output files (without extension). If None, uses 'roi_coordinates'
    save_directory : str, optional
        Directory where files will be saved. Default is current directory.
    
    Returns
    -------
    pd.DataFrame
        DataFrame containing ROI information and world coordinates
    """
    
    print(f"Processing volume file: {volume_file_location}")
    print(f"Using ROI labels from: {roi_label_file}")
    
    # Set default name if not provided
    if name_of_file is None:
        name_of_file = "roi_coordinates"
    
    # Handle directory creation
    save_path = Path(save_directory)
    if not save_path.exists():
        save_path.mkdir(parents=True)
        print(f"Created directory: {save_path}")
    else:
        # Check if directory has files
        existing_files = list(save_path.glob("*"))
        if existing_files:
            # Create subdirectory
            sub_dir = save_path / (name_of_file if name_of_file != "roi_coordinates" else "roi_cogs")
            sub_dir.mkdir(exist_ok=True)
            save_path = sub_dir
            print(f"Directory not empty. Created subdirectory: {save_path}")
    
    try:
        # Load the volume file
        volume_img = nib.load(volume_file_location)
        volume_data = volume_img.get_fdata()
        affine = volume_img.affine
        
        print(f"Volume shape: {volume_data.shape}")
        print(f"Volume data type: {volume_data.dtype}")
        
        # Get unique ROI labels in the volume
        unique_labels = np.unique(volume_data)
        roi_indices_in_volume = sorted([int(l) for l in unique_labels if l > 0])
        print(f"Found {len(roi_indices_in_volume)} unique ROIs in volume")
        
        # Load ROI label file
        roi_labels = []
        with open(roi_label_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line:  # Skip empty lines
                    roi_labels.append(line)
        
        print(f"Loaded {len(roi_labels)} ROI labels from file")
        
        # Process each ROI
        results = []
        
        for roi_line in roi_labels:
            try:
                # Parse the ROI line (expecting format: "number\tlabel")
                parts = roi_line.split('\t', 1)
                if len(parts) != 2:
                    print(f"Warning: Skipping malformed line: {roi_line}")
                    continue
                
                roi_number = int(parts[0])
                roi_name = parts[1].strip()
                
                # Check if this ROI exists in the volume
                if roi_number in roi_indices_in_volume:
                    # Calculate center of mass in voxel coordinates
                    mask = volume_data == roi_number
                    cog_voxel = center_of_mass(mask)
                    
                    # Convert to world coordinates
                    vox_coords = np.array([cog_voxel[0], cog_voxel[1], cog_voxel[2], 1])
                    world_coords = affine @ vox_coords
                    
                    results.append({
                        'roi_index': roi_number,
                        'roi_name': roi_name,
                        'cog_x': world_coords[0],
                        'cog_y': world_coords[1],
                        'cog_z': world_coords[2],
                        'cog_voxel_i': cog_voxel[0],
                        'cog_voxel_j': cog_voxel[1],
                        'cog_voxel_k': cog_voxel[2]
                    })
                else:
                    print(f"Warning: ROI {roi_number} ({roi_name}) not found in volume")
                    results.append({
                        'roi_index': roi_number,
                        'roi_name': roi_name,
                        'cog_x': np.nan,
                        'cog_y': np.nan,
                        'cog_z': np.nan,
                        'cog_voxel_i': np.nan,
                        'cog_voxel_j': np.nan,
                        'cog_voxel_k': np.nan
                    })
                    
            except Exception as e:
                print(f"Error processing ROI line '{roi_line}': {e}")
                continue
        
        # Create DataFrame
        df = pd.DataFrame(results)
        
        # Save files
        # 1. Save as pickle (pandas format)
        pickle_path = save_path / f"{name_of_file}.pkl"
        df.to_pickle(pickle_path)
        print(f"Saved pickle file: {pickle_path}")
        
        # 2. Save as CSV (comma-delimited)
        csv_comma_path = save_path / f"{name_of_file}_comma.csv"
        df.to_csv(csv_comma_path, index=False)
        print(f"Saved comma-delimited CSV: {csv_comma_path}")
        
        # 3. Save as CSV (tab-delimited)
        csv_tab_path = save_path / f"{name_of_file}_tab.csv"
        df.to_csv(csv_tab_path, sep='\t', index=False)
        print(f"Saved tab-delimited CSV: {csv_tab_path}")
        
        # Print summary
        valid_cogs = df[~df['cog_x'].isna()]
        print(f"\nSummary: Successfully calculated {len(valid_cogs)} COGs out of {len(df)} ROIs")
        
        return df
        
    except Exception as e:
        print(f"Error in coordinate_function: {e}")
        raise


def map_coordinate(original_coords_file, reduced_roi_file, save_directory=".", name_of_file=None):
    """
    Map a subset of ROIs to their coordinates from a full coordinate set.
    
    Parameters
    ----------
    original_coords_file : str or pd.DataFrame
        Path to file containing full ROI coordinates (pickle, CSV) or DataFrame
    reduced_roi_file : str
        Path to text/CSV file containing the subset of ROIs to map
    save_directory : str, optional
        Directory where files will be saved
    name_of_file : str, optional
        Name for output files. If None, uses 'mapped_roi_coordinates'
    
    Returns
    -------
    pd.DataFrame
        DataFrame containing mapped ROI coordinates with order preserved
    """
    
    print(f"Mapping coordinates from: {original_coords_file}")
    print(f"Using reduced ROI list from: {reduced_roi_file}")
    
    # Set default name
    if name_of_file is None:
        name_of_file = "mapped_roi_coordinates"
    
    # Handle directory
    save_path = Path(save_directory)
    if not save_path.exists():
        save_path.mkdir(parents=True)
    else:
        existing_files = list(save_path.glob("*"))
        if existing_files:
            sub_dir = save_path / (name_of_file if name_of_file != "mapped_roi_coordinates" else "roi_cogs")
            sub_dir.mkdir(exist_ok=True)
            save_path = sub_dir
    
    try:
        # Load original coordinates
        if isinstance(original_coords_file, pd.DataFrame):
            original_df = original_coords_file
        elif original_coords_file.endswith('.pkl'):
            original_df = pd.read_pickle(original_coords_file)
        elif original_coords_file.endswith('.csv'):
            # Try to detect delimiter
            with open(original_coords_file, 'r') as f:
                first_line = f.readline()
                if '\t' in first_line:
                    original_df = pd.read_csv(original_coords_file, sep='\t')
                else:
                    original_df = pd.read_csv(original_coords_file)
        else:
            raise ValueError(f"Unsupported file format: {original_coords_file}")
        
        print(f"Loaded {len(original_df)} ROIs from original coordinate file")
        
        # Create mapping dictionary from original data (name -> row data)
        roi_map = {}
        for _, row in original_df.iterrows():
            roi_name = row['roi_name'].strip()
            roi_map[roi_name] = row
        
        # Load reduced ROI list
        reduced_rois = []
        with open(reduced_roi_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                    
                # Handle both formats: "number\tname" or just "name"
                if '\t' in line:
                    parts = line.split('\t', 1)
                    if len(parts) == 2:
                        reduced_rois.append(parts[1].strip())
                elif ',' in line and not line.startswith('roi_'):  # CSV format
                    parts = line.split(',', 1)
                    if len(parts) == 2 and not parts[0].startswith('roi_'):
                        reduced_rois.append(parts[1].strip())
                else:
                    # Assume it's just the name
                    reduced_rois.append(line)
        
        print(f"Loaded {len(reduced_rois)} ROIs from reduced list")
        
        # Map coordinates preserving order
        mapped_results = []
        
        for new_idx, roi_name in enumerate(reduced_rois, start=1):
            if roi_name in roi_map:
                original_row = roi_map[roi_name]
                mapped_results.append({
                    'mapped_index': new_idx,  # New index in reduced list (1-based)
                    'original_index': original_row['roi_index'],  # Original index
                    'roi_name': roi_name,
                    'cog_x': original_row['cog_x'],
                    'cog_y': original_row['cog_y'],
                    'cog_z': original_row['cog_z'],
                    'cog_voxel_i': original_row.get('cog_voxel_i', np.nan),
                    'cog_voxel_j': original_row.get('cog_voxel_j', np.nan),
                    'cog_voxel_k': original_row.get('cog_voxel_k', np.nan)
                })
            else:
                print(f"Warning: ROI '{roi_name}' not found in original coordinates")
                mapped_results.append({
                    'mapped_index': new_idx,
                    'original_index': np.nan,
                    'roi_name': roi_name,
                    'cog_x': np.nan,
                    'cog_y': np.nan,
                    'cog_z': np.nan,
                    'cog_voxel_i': np.nan,
                    'cog_voxel_j': np.nan,
                    'cog_voxel_k': np.nan
                })
        
        # Create DataFrame
        mapped_df = pd.DataFrame(mapped_results)
        
        # Save files
        # 1. Pickle
        pickle_path = save_path / f"{name_of_file}.pkl"
        mapped_df.to_pickle(pickle_path)
        print(f"Saved pickle file: {pickle_path}")
        
        # 2. Comma-delimited CSV
        csv_comma_path = save_path / f"{name_of_file}_comma.csv"
        mapped_df.to_csv(csv_comma_path, index=False)
        print(f"Saved comma-delimited CSV: {csv_comma_path}")
        
        # 3. Tab-delimited CSV
        csv_tab_path = save_path / f"{name_of_file}_tab.csv"
        mapped_df.to_csv(csv_tab_path, sep='\t', index=False)
        print(f"Saved tab-delimited CSV: {csv_tab_path}")
        
        # Print summary
        valid_mapped = mapped_df[~mapped_df['cog_x'].isna()]
        print(f"\nSummary: Successfully mapped {len(valid_mapped)} out of {len(mapped_df)} ROIs")
        print(f"Order preserved: {len(mapped_df)} ROIs with indices 1-{len(mapped_df)}")
        
        return mapped_df
        
    except Exception as e:
        print(f"Error in map_coordinate: {e}")
        raise


# Convenience function for quick testing
def test_functions():
    """Test the functions with example data."""
    print("ROI Coordinate Tools loaded successfully!")
    print("Available functions:")
    print("  - coordinate_function(volume_file, roi_label_file, name, save_dir)")
    print("  - map_coordinate(original_coords, reduced_rois, save_dir, name)")
    

if __name__ == "__main__":
    test_functions()