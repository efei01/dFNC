"""
ROI Coordinate Tools
====================
Functions for extracting and mapping ROI coordinates from brain volume files.
"""

import numpy as np
import pandas as pd
import nibabel as nib
from scipy.ndimage import center_of_mass
from scipy.io import loadmat
from pathlib import Path
from typing import Tuple, List, Union


def coordinate_function(volume_file_location, roi_label_file, name_of_file=None, save_directory="."):
    """
    Extract ROI center of gravity coordinates from a volume file and save in multiple formats.

    Parameters
    ----------
    volume_file_location : str
        Path to the NIfTI volume file containing ROI labels
    roi_label_file : str
        Path to the text file containing ROI labels (tab-delimited: number\\tlabel)
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

    if name_of_file is None:
        name_of_file = "roi_coordinates"

    save_path = Path(save_directory)
    if not save_path.exists():
        save_path.mkdir(parents=True)
        print(f"Created directory: {save_path}")
    else:
        existing_files = list(save_path.glob("*"))
        if existing_files:
            sub_dir = save_path / (name_of_file if name_of_file != "roi_coordinates" else "roi_cogs")
            sub_dir.mkdir(exist_ok=True)
            save_path = sub_dir
            print(f"Directory not empty. Created subdirectory: {save_path}")

    try:
        volume_img = nib.load(volume_file_location)
        volume_data = volume_img.get_fdata()
        affine = volume_img.affine

        print(f"Volume shape: {volume_data.shape}")
        print(f"Volume data type: {volume_data.dtype}")

        unique_labels = np.unique(volume_data)
        roi_indices_in_volume = sorted([int(l) for l in unique_labels if l > 0])
        print(f"Found {len(roi_indices_in_volume)} unique ROIs in volume")

        roi_labels = []
        with open(roi_label_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    roi_labels.append(line)

        print(f"Loaded {len(roi_labels)} ROI labels from file")

        results = []

        for roi_line in roi_labels:
            try:
                parts = roi_line.split('\t', 1)
                if len(parts) != 2:
                    print(f"Warning: Skipping malformed line: {roi_line}")
                    continue

                roi_number = int(parts[0])
                roi_name = parts[1].strip()

                if roi_number in roi_indices_in_volume:
                    mask = volume_data == roi_number
                    cog_voxel = center_of_mass(mask)

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

        df = pd.DataFrame(results)

        pickle_path = save_path / f"{name_of_file}.pkl"
        df.to_pickle(pickle_path)
        print(f"Saved pickle file: {pickle_path}")

        csv_comma_path = save_path / f"{name_of_file}_comma.csv"
        df.to_csv(csv_comma_path, index=False)
        print(f"Saved comma-delimited CSV: {csv_comma_path}")

        csv_tab_path = save_path / f"{name_of_file}_tab.csv"
        df.to_csv(csv_tab_path, sep='\t', index=False)
        print(f"Saved tab-delimited CSV: {csv_tab_path}")

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
    reduced_roi_file : str or pd.DataFrame
        Path to text/CSV/node file OR DataFrame containing the subset of ROIs to map.
        Supports:
        - Text file with ROI names (one per line or tab-delimited index\\tname)
        - CSV file with ROI names
        - BrainNet Viewer .node file (uses last column as ROI names)
        - DataFrame with 'roi_name' column (e.g., from load_node_file())
    save_directory : str, optional
        Directory where files will be saved
    name_of_file : str, optional
        Name for output files. If None, uses 'mapped_roi_coordinates'

    Returns
    -------
    tuple (pd.DataFrame, list)
        - DataFrame containing mapped ROI coordinates with order preserved
        - List of ROI names that could not be mapped
    """

    print(f"Mapping coordinates from: {original_coords_file}")
    print(f"Using reduced ROI list from: {type(reduced_roi_file).__name__ if isinstance(reduced_roi_file, pd.DataFrame) else reduced_roi_file}")

    if name_of_file is None:
        name_of_file = "mapped_roi_coordinates"

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
        if isinstance(original_coords_file, pd.DataFrame):
            original_df = original_coords_file
        elif original_coords_file.endswith('.pkl'):
            original_df = pd.read_pickle(original_coords_file)
        elif original_coords_file.endswith('.csv'):
            with open(original_coords_file, 'r') as f:
                first_line = f.readline()
                if '\t' in first_line:
                    original_df = pd.read_csv(original_coords_file, sep='\t')
                else:
                    original_df = pd.read_csv(original_coords_file)
        else:
            raise ValueError(f"Unsupported file format: {original_coords_file}")

        print(f"Loaded {len(original_df)} ROIs from original coordinate file")

        roi_map = {}
        for _, row in original_df.iterrows():
            roi_name = row['roi_name'].strip()
            roi_map[roi_name] = row

        reduced_rois = []

        # Check if reduced_roi_file is a DataFrame (e.g., from load_node_file())
        if isinstance(reduced_roi_file, pd.DataFrame):
            # Extract ROI names from DataFrame
            if 'roi_name' in reduced_roi_file.columns:
                reduced_rois = reduced_roi_file['roi_name'].astype(str).str.strip().tolist()
            else:
                raise ValueError("DataFrame must have 'roi_name' column. "
                               f"Available columns: {reduced_roi_file.columns.tolist()}")
            print(f"Loaded {len(reduced_rois)} ROIs from DataFrame")
        else:
            # It's a file path
            reduced_file_path = Path(reduced_roi_file)

            # Check if it's a .node file (BrainNet Viewer format)
            if reduced_file_path.suffix.lower() == '.node':
                # Node file format: X Y Z size color roi_name (tab-separated)
                # Extract the last column (roi_name)
                with open(reduced_roi_file, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        parts = line.split('\t')
                        if len(parts) >= 6:
                            # Last column is roi_name
                            reduced_rois.append(parts[-1].strip())
                        elif len(parts) >= 1:
                            # Fallback: use last part
                            reduced_rois.append(parts[-1].strip())
                print(f"Loaded {len(reduced_rois)} ROIs from .node file (last column)")
            else:
                # Standard text/CSV file handling
                with open(reduced_roi_file, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue

                        if '\t' in line:
                            parts = line.split('\t', 1)
                            if len(parts) == 2:
                                reduced_rois.append(parts[1].strip())
                        elif ',' in line and not line.startswith('roi_'):
                            parts = line.split(',', 1)
                            if len(parts) == 2 and not parts[0].startswith('roi_'):
                                reduced_rois.append(parts[1].strip())
                        else:
                            reduced_rois.append(line)

                print(f"Loaded {len(reduced_rois)} ROIs from reduced list")

        mapped_results = []
        unmapped_rois = []

        for new_idx, roi_name in enumerate(reduced_rois, start=1):
            if roi_name in roi_map:
                original_row = roi_map[roi_name]
                mapped_results.append({
                    'mapped_index': new_idx,
                    'original_index': original_row['roi_index'],
                    'roi_name': roi_name,
                    'cog_x': original_row['cog_x'],
                    'cog_y': original_row['cog_y'],
                    'cog_z': original_row['cog_z'],
                    'cog_voxel_i': original_row.get('cog_voxel_i', np.nan),
                    'cog_voxel_j': original_row.get('cog_voxel_j', np.nan),
                    'cog_voxel_k': original_row.get('cog_voxel_k', np.nan)
                })
            else:
                unmapped_rois.append({'index': new_idx, 'name': roi_name})
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

        mapped_df = pd.DataFrame(mapped_results)

        pickle_path = save_path / f"{name_of_file}.pkl"
        mapped_df.to_pickle(pickle_path)
        print(f"Saved pickle file: {pickle_path}")

        csv_comma_path = save_path / f"{name_of_file}_comma.csv"
        mapped_df.to_csv(csv_comma_path, index=False)
        print(f"Saved comma-delimited CSV: {csv_comma_path}")

        csv_tab_path = save_path / f"{name_of_file}_tab.csv"
        mapped_df.to_csv(csv_tab_path, sep='\t', index=False)
        print(f"Saved tab-delimited CSV: {csv_tab_path}")

        valid_mapped = mapped_df[~mapped_df['cog_x'].isna()]
        print(f"\nSummary: Successfully mapped {len(valid_mapped)} out of {len(mapped_df)} ROIs")

        if unmapped_rois:
            print(f"\n{'='*60}")
            print(f"WARNING: {len(unmapped_rois)} ROI(s) could NOT be mapped:")
            print(f"{'='*60}")
            for roi in unmapped_rois:
                print(f"  Index {roi['index']:3d}: {roi['name']}")
            print(f"{'='*60}")
        else:
            print("\nAll ROIs were successfully mapped!")

        return mapped_df, unmapped_rois

    except Exception as e:
        print(f"Error in map_coordinate: {e}")
        raise


def load_and_clean_coordinates(csv_file_path, output_file_name=None, save_directory="."):
    """
    Load ROI coordinates from a CSV file and remove rows with missing values.

    Parameters
    ----------
    csv_file_path : str
        Path to the CSV file containing ROI coordinates
    output_file_name : str, optional
        Name for the output files (without extension)
    save_directory : str, optional
        Directory where cleaned files will be saved

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with rows containing NaN values removed
    """

    print(f"Loading coordinates from: {csv_file_path}")

    if output_file_name is None:
        base_name = Path(csv_file_path).stem
        for suffix in ['_comma', '_tab', '_cleaned']:
            if base_name.endswith(suffix):
                base_name = base_name[:-len(suffix)]
        output_file_name = f"{base_name}_cleaned"

    save_path = Path(save_directory)
    if not save_path.exists():
        save_path.mkdir(parents=True)
        print(f"Created directory: {save_path}")

    try:
        with open(csv_file_path, 'r') as f:
            first_line = f.readline()
            if '\t' in first_line:
                df = pd.read_csv(csv_file_path, sep='\t')
                print("Detected tab-delimited CSV")
            else:
                df = pd.read_csv(csv_file_path)
                print("Detected comma-delimited CSV")

        original_count = len(df)
        print(f"Loaded {original_count} rows from CSV")

        coord_columns = ['cog_x', 'cog_y', 'cog_z']
        available_coord_cols = [col for col in coord_columns if col in df.columns]

        if not available_coord_cols:
            print("Warning: No coordinate columns (cog_x, cog_y, cog_z) found in CSV")
            return df

        rows_with_nan = df[available_coord_cols].isna().any(axis=1)
        removed_rows = df[rows_with_nan]

        if len(removed_rows) > 0:
            print(f"\nRemoved {len(removed_rows)} rows with missing values:")
            for _, row in removed_rows.iterrows():
                roi_name = row.get('roi_name', 'Unknown')
                roi_idx = row.get('roi_index', row.get('mapped_index', 'N/A'))
                print(f"  - ROI {roi_idx}: {roi_name}")

        cleaned_df = df[~rows_with_nan].copy()
        cleaned_df.reset_index(drop=True, inplace=True)

        print(f"\nCleaned DataFrame: {len(cleaned_df)} rows (removed {original_count - len(cleaned_df)})")

        pickle_path = save_path / f"{output_file_name}.pkl"
        cleaned_df.to_pickle(pickle_path)
        print(f"Saved pickle file: {pickle_path}")

        csv_comma_path = save_path / f"{output_file_name}_comma.csv"
        cleaned_df.to_csv(csv_comma_path, index=False)
        print(f"Saved comma-delimited CSV: {csv_comma_path}")

        csv_tab_path = save_path / f"{output_file_name}_tab.csv"
        cleaned_df.to_csv(csv_tab_path, sep='\t', index=False)
        print(f"Saved tab-delimited CSV: {csv_tab_path}")

        return cleaned_df

    except Exception as e:
        print(f"Error in load_and_clean_coordinates: {e}")
        raise


def load_matrix_replace_nan(file_path, replacement_value=0):
    """
    Load a connectivity matrix from .mat or .txt file and replace NaN values.

    Parameters
    ----------
    file_path : str
        Path to the matrix file (.mat or .txt)
    replacement_value : float, optional
        Value to replace NaNs with. Default is 0.

    Returns
    -------
    np.ndarray
        Matrix with NaN values replaced
    """

    print(f"Loading matrix from: {file_path}")

    file_ext = Path(file_path).suffix.lower()

    try:
        if file_ext == '.mat':
            mat_data = loadmat(file_path)
            matrix_keys = [k for k in mat_data.keys() if not k.startswith('__')]
            if len(matrix_keys) == 1:
                matrix = mat_data[matrix_keys[0]]
            else:
                print(f"Available keys in .mat file: {matrix_keys}")
                for common_name in ['conn_matrix', 'connectivity', 'matrix', 'data', 'M']:
                    if common_name in matrix_keys:
                        matrix = mat_data[common_name]
                        break
                else:
                    matrix = mat_data[matrix_keys[0]]
                    print(f"Using first key: {matrix_keys[0]}")

        elif file_ext == '.txt':
            matrix = np.loadtxt(file_path)

        else:
            raise ValueError(f"Unsupported file format: {file_ext}. Use .mat or .txt")

        nan_count = np.isnan(matrix).sum()
        total_elements = matrix.size

        print(f"Matrix shape: {matrix.shape}")
        print(f"Found {nan_count} NaN values out of {total_elements} elements ({100*nan_count/total_elements:.2f}%)")

        if nan_count > 0:
            matrix = np.nan_to_num(matrix, nan=replacement_value)
            print(f"Replaced NaN values with {replacement_value}")
        else:
            print("No NaN values found in matrix")

        return matrix

    except Exception as e:
        print(f"Error in load_matrix_replace_nan: {e}")
        raise
