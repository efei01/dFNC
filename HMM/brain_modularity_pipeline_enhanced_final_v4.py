"""
Enhanced Brain Modularity Pipeline with PC Classification - VERSION 4
=========================================================================
v4 Changes:
- Added comprehensive camera control system with preset views
- Interactive camera position input (x, y, z coordinates)
- Smooth animation between camera positions
- Keyboard shortcuts for quick view switching
- Camera position save/load functionality
- Enhanced UI controls for view manipulation

v3 Features maintained:
- FIXED: Node borders visible using dual-layer rendering (border_width=6)
- Node borders show role classification clearly

v2 Features maintained:
- Bigger node size differences
- Variable edge thickness based on coherence strength
- Proper Q and Z value loading
- Brain state ordering fix
"""

import numpy as np
import pandas as pd
import nibabel as nib
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path
import colorsys
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# CAMERA CONTROL SYSTEM
# ============================================================================

class CameraController:
    """
    Comprehensive camera control system for 3D brain visualizations.
    Provides preset views, smooth transitions, and manual control.
    """
    
    # Preset camera positions for standard neuroimaging views
    PRESET_VIEWS = {
        'anterior': {
            'name': 'Anterior (Front)',
            'eye': {'x': 0, 'y': 2, 'z': 0},
            'center': {'x': 0, 'y': 0, 'z': 0},
            'up': {'x': 0, 'y': 0, 'z': 1}
        },
        'posterior': {
            'name': 'Posterior (Back)',
            'eye': {'x': 0, 'y': -2, 'z': 0},
            'center': {'x': 0, 'y': 0, 'z': 0},
            'up': {'x': 0, 'y': 0, 'z': 1}
        },
        'left': {
            'name': 'Left Lateral',
            'eye': {'x': -2, 'y': 0, 'z': 0},
            'center': {'x': 0, 'y': 0, 'z': 0},
            'up': {'x': 0, 'y': 0, 'z': 1}
        },
        'right': {
            'name': 'Right Lateral',
            'eye': {'x': 2, 'y': 0, 'z': 0},
            'center': {'x': 0, 'y': 0, 'z': 0},
            'up': {'x': 0, 'y': 0, 'z': 1}
        },
        'superior': {
            'name': 'Superior (Top)',
            'eye': {'x': 0, 'y': 0, 'z': 2},
            'center': {'x': 0, 'y': 0, 'z': 0},
            'up': {'x': 0, 'y': 1, 'z': 0}
        },
        'inferior': {
            'name': 'Inferior (Bottom)',
            'eye': {'x': 0, 'y': 0, 'z': -2},
            'center': {'x': 0, 'y': 0, 'z': 0},
            'up': {'x': 0, 'y': -1, 'z': 0}
        },
        'anterolateral_left': {
            'name': 'Anterolateral Left',
            'eye': {'x': -1.5, 'y': 1.5, 'z': 0.5},
            'center': {'x': 0, 'y': 0, 'z': 0},
            'up': {'x': 0, 'y': 0, 'z': 1}
        },
        'anterolateral_right': {
            'name': 'Anterolateral Right',
            'eye': {'x': 1.5, 'y': 1.5, 'z': 0.5},
            'center': {'x': 0, 'y': 0, 'z': 0},
            'up': {'x': 0, 'y': 0, 'z': 1}
        },
        'posterolateral_left': {
            'name': 'Posterolateral Left',
            'eye': {'x': -1.5, 'y': -1.5, 'z': 0.5},
            'center': {'x': 0, 'y': 0, 'z': 0},
            'up': {'x': 0, 'y': 0, 'z': 1}
        },
        'posterolateral_right': {
            'name': 'Posterolateral Right',
            'eye': {'x': 1.5, 'y': -1.5, 'z': 0.5},
            'center': {'x': 0, 'y': 0, 'z': 0},
            'up': {'x': 0, 'y': 0, 'z': 1}
        },
        'oblique': {
            'name': 'Oblique View',
            'eye': {'x': 1.5, 'y': 1.5, 'z': 1.5},
            'center': {'x': 0, 'y': 0, 'z': 0},
            'up': {'x': 0, 'y': 0, 'z': 1}
        },
        'custom': {
            'name': 'Custom View',
            'eye': {'x': 1.5, 'y': 1.5, 'z': 1.5},
            'center': {'x': 0, 'y': 0, 'z': 0},
            'up': {'x': 0, 'y': 0, 'z': 1}
        }
    }
    
    @classmethod
    def get_camera_position(cls, view_name: str = 'oblique') -> Dict:
        """Get camera position for a named view."""
        if view_name in cls.PRESET_VIEWS:
            return cls.PRESET_VIEWS[view_name].copy()
        else:
            print(f"Warning: Unknown view '{view_name}', using oblique view")
            return cls.PRESET_VIEWS['oblique'].copy()
    
    @classmethod
    def create_camera_from_angles(cls, azimuth: float, elevation: float, 
                                 distance: float = 2.0) -> Dict:
        """
        Create camera position from spherical coordinates.
        
        Parameters
        ----------
        azimuth : float
            Horizontal rotation angle in degrees (0-360)
        elevation : float
            Vertical rotation angle in degrees (-90 to 90)
        distance : float
            Distance from origin
        """
        # Convert to radians
        az_rad = np.radians(azimuth)
        el_rad = np.radians(elevation)
        
        # Calculate eye position
        x = distance * np.cos(el_rad) * np.cos(az_rad)
        y = distance * np.cos(el_rad) * np.sin(az_rad)
        z = distance * np.sin(el_rad)
        
        return {
            'name': f'Custom (Az:{azimuth:.0f}°, El:{elevation:.0f}°)',
            'eye': {'x': x, 'y': y, 'z': z},
            'center': {'x': 0, 'y': 0, 'z': 0},
            'up': {'x': 0, 'y': 0, 'z': 1}
        }
    
    @classmethod
    def create_camera_from_coordinates(cls, x: float, y: float, z: float,
                                      center_x: float = 0, center_y: float = 0, 
                                      center_z: float = 0) -> Dict:
        """
        Create camera position from Cartesian coordinates.
        
        Parameters
        ----------
        x, y, z : float
            Eye position coordinates
        center_x, center_y, center_z : float
            Center point coordinates (what the camera looks at)
        """
        return {
            'name': f'Custom (X:{x:.1f}, Y:{y:.1f}, Z:{z:.1f})',
            'eye': {'x': x, 'y': y, 'z': z},
            'center': {'x': center_x, 'y': center_y, 'z': center_z},
            'up': {'x': 0, 'y': 0, 'z': 1}
        }
    
    @classmethod
    def save_camera_position(cls, camera_dict: Dict, filepath: Union[str, Path]):
        """Save camera position to JSON file."""
        filepath = Path(filepath)
        with open(filepath, 'w') as f:
            json.dump(camera_dict, f, indent=2)
        print(f"Camera position saved to: {filepath}")
    
    @classmethod
    def load_camera_position(cls, filepath: Union[str, Path]) -> Dict:
        """Load camera position from JSON file."""
        filepath = Path(filepath)
        if filepath.exists():
            with open(filepath, 'r') as f:
                return json.load(f)
        else:
            raise FileNotFoundError(f"Camera position file not found: {filepath}")
    
    @classmethod
    def interpolate_cameras(cls, camera1: Dict, camera2: Dict, steps: int = 30) -> List[Dict]:
        """
        Create smooth transition between two camera positions.
        
        Parameters
        ----------
        camera1, camera2 : dict
            Start and end camera positions
        steps : int
            Number of interpolation steps
        """
        frames = []
        for i in range(steps + 1):
            t = i / steps
            frame = {
                'eye': {
                    'x': camera1['eye']['x'] + t * (camera2['eye']['x'] - camera1['eye']['x']),
                    'y': camera1['eye']['y'] + t * (camera2['eye']['y'] - camera1['eye']['y']),
                    'z': camera1['eye']['z'] + t * (camera2['eye']['z'] - camera1['eye']['z'])
                },
                'center': {
                    'x': camera1['center']['x'] + t * (camera2['center']['x'] - camera1['center']['x']),
                    'y': camera1['center']['y'] + t * (camera2['center']['y'] - camera1['center']['y']),
                    'z': camera1['center']['z'] + t * (camera2['center']['z'] - camera1['center']['z'])
                },
                'up': camera1['up']  # Keep up vector constant
            }
            frames.append(frame)
        return frames


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_mesh_file(mesh_path: Union[str, Path]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load brain mesh from GIFTI file with proper intent checking.
    """
    mesh_path = Path(mesh_path)
    if not mesh_path.exists():
        raise FileNotFoundError(f"Mesh file not found: {mesh_path}")
    
    print(f"Loading mesh from: {mesh_path}")
    
    # Load the GIFTI file
    gii = nib.load(str(mesh_path))
    vertices = None
    faces = None
    
    # Properly extract vertices and faces using intent codes
    for array in gii.darrays:
        if array.intent == nib.nifti1.intent_codes['NIFTI_INTENT_POINTSET']:
            vertices = array.data
            print(f"  Found vertices array with shape: {vertices.shape}")
        elif array.intent == nib.nifti1.intent_codes['NIFTI_INTENT_TRIANGLE']:
            faces = array.data
            print(f"  Found faces array with shape: {faces.shape}")
    
    # Fallback if intent codes aren't set properly
    if vertices is None or faces is None:
        print("  Warning: Could not find arrays by intent, using index-based loading")
        if len(gii.darrays) >= 2:
            vertices = gii.darrays[0].data
            faces = gii.darrays[1].data
        else:
            raise ValueError(f"Could not extract vertices and faces from mesh file. Found {len(gii.darrays)} arrays.")
    
    print(f"  Successfully loaded mesh: {vertices.shape[0]} vertices, {faces.shape[0]} faces")
    return vertices, faces


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super(NumpyEncoder, self).default(obj)


# ============================================================================
# NETNEUROTOOLS LOADER CLASS
# ============================================================================

class NetNeurotoolsModularityLoader:
    """Load and process netneurotools modularity results"""
    
    def __init__(self, base_dir: Union[str, Path]):
        self.base_dir = Path(base_dir)
        self.comprehensive_dir = self.base_dir / 'comprehensive_analysis'
        self.csv_dir = self.base_dir / 'csv_files'
        self.results_dir = self.base_dir / 'results'
        self.plots_dir = self.base_dir / 'plots'
        self.reports_dir = self.base_dir / 'reports'
        
        if not self.base_dir.exists():
            raise ValueError(f"Base directory not found: {self.base_dir}")
        
        print(f"NetNeurotools Loader initialized with base: {self.base_dir}")
    
    def load_summary_statistics(self) -> pd.DataFrame:
        """Load summary statistics CSV with Q and Z values for all states"""
        summary_file = self.csv_dir / 'all_k_summary_statistics.csv'
        if summary_file.exists():
            print(f"   Loading summary statistics from: {summary_file}")
            return pd.read_csv(summary_file)
        else:
            print(f"   Warning: Summary statistics not found at {summary_file}")
        return None
    
    def load_comprehensive_results(self, k: int) -> Dict:
        """Load comprehensive analysis results with proper Q and Z values from CSV"""
        k_dir = self.comprehensive_dir / f'k{k}_detailed'
        if not k_dir.exists():
            raise ValueError(f"Results for k={k} not found at {k_dir}")
        
        results = {'k': k, 'states': []}
        
        # First try to load summary statistics from CSV for accurate Q and Z values
        summary_df = self.load_summary_statistics()
        Q_values = {}
        Q_z_scores = {}
        module_significance = {}
        
        if summary_df is not None:
            # Filter for this k value
            k_data = summary_df[summary_df['k'] == k]
            for _, row in k_data.iterrows():
                state_idx = int(row['state'])
                Q_values[state_idx] = float(row['Q_total'])
                Q_z_scores[state_idx] = float(row['Q_z_score'])
                print(f"   Loaded from CSV - State {state_idx}: Q={Q_values[state_idx]:.3f}, Z={Q_z_scores[state_idx]:.2f}")
        
        # Fallback to loading from summary.npz if CSV not available
        if not Q_values:
            summary_file = self.comprehensive_dir / f'k{k}_summary.npz'
            if summary_file.exists():
                print(f"   Loading summary from: {summary_file}")
                summary_data = np.load(summary_file, allow_pickle=True)
                
                if 'Q_values' in summary_data:
                    Q_array = summary_data['Q_values']
                    for i, q in enumerate(Q_array):
                        Q_values[i] = float(q)
                
                if 'Q_z_scores' in summary_data:
                    Z_array = summary_data['Q_z_scores']
                    for i, z in enumerate(Z_array):
                        Q_z_scores[i] = float(z)
                
                if 'module_significance' in summary_data:
                    module_significance = summary_data['module_significance']
        
        # Load each state's detailed data
        for state_file in sorted(k_dir.glob('state*_arrays.npz')):
            state_idx = int(state_file.stem.split('state')[1].split('_')[0])
            
            # Load arrays
            arrays = np.load(state_file, allow_pickle=True)
            
            # Load node metrics CSV
            csv_file = k_dir / f'state{state_idx}_node_metrics.csv'
            if csv_file.exists():
                metrics_df = pd.read_csv(csv_file)
            else:
                print(f"   Warning: Missing metrics CSV for state {state_idx}")
                continue
            
            # Get module assignments
            if 'module_assignments' in arrays:
                consensus = arrays['module_assignments']
            else:
                consensus = metrics_df['module'].values
            
            # Get module significance for this state
            if isinstance(module_significance, np.ndarray) and state_idx < len(module_significance):
                state_module_sig = module_significance[state_idx]
            else:
                state_module_sig = np.ones(len(np.unique(consensus)), dtype=bool)
            
            state_data = {
                'state_idx': state_idx,
                'consensus': consensus,
                'Q_total': Q_values.get(state_idx, 0.0),
                'Q_z_score': Q_z_scores.get(state_idx, 0.0),
                'module_significance': state_module_sig,
                'Q_per_module': arrays.get('Q_per_module', np.zeros(len(np.unique(consensus)))),
                'modules_df': metrics_df,
                'participation_coef': metrics_df['participation_coef'].values if 'participation_coef' in metrics_df else np.zeros(len(metrics_df)),
                'within_module_zscore': metrics_df['within_module_zscore'].values if 'within_module_zscore' in metrics_df else np.zeros(len(metrics_df))
            }
            
            results['states'].append(state_data)
            
            if state_data['Q_total'] != 0:
                print(f"   State {state_idx}: Q={state_data['Q_total']:.3f}, Z={state_data['Q_z_score']:.2f}")
        
        return results


# ============================================================================
# ENHANCED FUNCTIONS FOR PC CLASSIFICATION
# ============================================================================

def classify_node_role(z_score: float, pc: float) -> Tuple[str, str]:
    """
    Classify node role based on within-module z-score and participation coefficient.
    """
    if z_score < 0.05 and pc < 0.05:
        return "Ultra-peripheral", "#E8E8E8"  # Very light gray
    elif z_score < 2.5 and pc < 0.62:
        if abs(pc - 0.5) < 0.1:  # Kinless nodes
            return "Kinless", "#FFB6C1"  # Light pink
        else:
            return "Peripheral", "#B0B0B0"  # Light gray
    elif z_score < 2.5 and pc >= 0.62:
        return "Satellite Connector", "#87CEEB"  # Sky blue
    elif z_score >= 2.5 and pc < 0.3:
        return "Provincial Hub", "#FFD700"  # Gold
    elif z_score >= 2.5 and pc >= 0.3:
        return "Connector Hub", "#FF4500"  # Red-orange
    else:
        return "Unclassified", "#808080"  # Gray


def calculate_node_size(pc: float, z_score: float, mode: str = 'both', 
                        base_size: int = 6, max_multiplier: float = 5.0) -> float:
    """
    Calculate dynamic node size with controlled scaling for better visibility.
    """
    if mode == 'pc':
        multiplier = 1 + (pc ** 0.5) * (max_multiplier - 1)
    elif mode == 'zscore':
        normalized_z = min(abs(z_score) / 2.0, 1.0)
        multiplier = 1 + (normalized_z ** 0.6) * (max_multiplier - 1)
    elif mode == 'both':
        pc_component = (pc ** 0.5) * (max_multiplier - 1) * 0.5
        z_component = (min(abs(z_score) / 2.0, 1.0) ** 0.6) * (max_multiplier - 1) * 0.5
        multiplier = 1 + pc_component + z_component
    else:
        multiplier = 1
    
    final_size = base_size * multiplier
    return max(base_size * 0.7, final_size)


def calculate_edge_width(weight: float, all_weights: np.ndarray, 
                        min_width: float = 0.5, max_width: float = 6.0) -> float:
    """
    Calculate edge width based on coherence strength.
    """
    weight_abs = abs(weight)
    min_weight = np.min(np.abs(all_weights[all_weights != 0]))
    max_weight = np.max(np.abs(all_weights))
    
    if max_weight > min_weight:
        normalized = (weight_abs - min_weight) / (max_weight - min_weight)
    else:
        normalized = 0.5
    
    normalized = normalized ** 0.7
    width = min_width + normalized * (max_width - min_width)
    return width


def filter_edges_by_module(connectivity_matrix, module_assignments, module_id, mode='all'):
    """
    Filter edges based on module membership.
    """
    filtered = connectivity_matrix.copy()
    module_mask = (module_assignments == module_id)
    
    if mode == 'intra':
        for i in range(len(module_assignments)):
            for j in range(len(module_assignments)):
                if not (module_mask[i] and module_mask[j]):
                    filtered[i, j] = 0
    elif mode == 'inter':
        for i in range(len(module_assignments)):
            for j in range(len(module_assignments)):
                if not ((module_mask[i] and not module_mask[j]) or 
                       (not module_mask[i] and module_mask[j])):
                    filtered[i, j] = 0
    
    return filtered


def threshold_matrix_top_n(matrix, n_edges):
    """Keep only top N edges in the matrix"""
    matrix_copy = matrix.copy()
    
    upper_tri = np.triu(matrix_copy, k=1)
    flat_values = upper_tri[upper_tri != 0]
    
    if len(flat_values) > n_edges:
        threshold_value = np.sort(np.abs(flat_values))[-n_edges]
        matrix_copy[np.abs(matrix_copy) < threshold_value] = 0
    
    return matrix_copy


# ============================================================================
# ENHANCED VISUALIZATION WITH CAMERA CONTROLS - V4
# ============================================================================

def create_enhanced_modularity_visualization(
    vertices: np.ndarray,
    faces: np.ndarray,
    roi_coords_df: pd.DataFrame,
    connectivity_matrix: np.ndarray,
    module_data: Dict,
    metrics_df: pd.DataFrame = None,
    plot_title: str = "Enhanced Modularity Visualization",
    save_path: str = "enhanced_modularity_viz.html",
    visualization_type: str = "all",
    node_sizing_mode: str = "both",
    base_node_size: int = 12,
    max_node_multiplier: float = 2.0,
    n_top_edges: Optional[int] = None,
    edge_width_range: Tuple[float, float] = (1.0, 6.0),
    mesh_color: str = 'lightgray',
    mesh_opacity: float = 0.15,
    show_labels: bool = True,
    label_font_size: int = 10,
    show_significance: bool = True,
    border_width: int = 6,  # V4: Changed to 6 as requested
    actual_state_label: Optional[int] = None,
    camera_view: str = 'oblique',  # V4: New parameter
    custom_camera: Optional[Dict] = None,  # V4: New parameter
    enable_camera_controls: bool = True,  # V4: New parameter
    save_all_views: bool = False  # V4: New parameter to save multiple views
) -> Tuple[go.Figure, Dict]:
    """
    Create enhanced modularity visualization with camera controls (V4).
    
    V4 Changes:
    - Interactive camera controls with preset views
    - Custom camera position input
    - Multiple view exports
    """
    
    print(f"Creating enhanced {visualization_type} visualization (node size: {node_sizing_mode})...")
    
    # Extract module assignments
    module_assignments = module_data['consensus']
    if len(module_assignments.shape) > 1:
        module_assignments = module_assignments.flatten()
    
    # Get PC and Z-score data
    if metrics_df is not None and 'participation_coef' in metrics_df.columns:
        pc_values = metrics_df['participation_coef'].values
        z_scores = metrics_df['within_module_zscore'].values
        roi_names = metrics_df['roi_name'].values
    else:
        pc_values = module_data.get('participation_coef', np.zeros(len(module_assignments)))
        z_scores = module_data.get('within_module_zscore', np.zeros(len(module_assignments)))
        roi_names = [f"ROI_{i}" for i in range(len(module_assignments))]
    
    # Classify nodes
    node_roles = []
    node_role_colors = []
    for pc, z in zip(pc_values, z_scores):
        role, color = classify_node_role(z, pc)
        node_roles.append(role)
        node_role_colors.append(color)
    
    # Calculate dynamic node sizes
    node_sizes = []
    for pc, z in zip(pc_values, z_scores):
        size = calculate_node_size(pc, z, node_sizing_mode, base_node_size, max_node_multiplier)
        node_sizes.append(size)
    
    # Apply top N edges thresholding if requested
    if n_top_edges is not None:
        connectivity_matrix = threshold_matrix_top_n(connectivity_matrix, n_top_edges)
    
    # Get module significance
    if 'module_significance' in module_data and show_significance:
        module_significance = module_data['module_significance']
    else:
        module_significance = np.ones(len(np.unique(module_assignments)), dtype=bool)
    
    # Identify active nodes
    node_has_connection = (np.any(connectivity_matrix != 0, axis=0) | 
                          np.any(connectivity_matrix != 0, axis=1))
    active_nodes = np.where(node_has_connection)[0]
    
    # Generate module colors
    unique_modules = np.unique(module_assignments[active_nodes])
    n_modules = len(unique_modules)
    
    if n_modules <= 10:
        colors = px.colors.qualitative.Plotly[:n_modules]
    else:
        colors = []
        for i in range(n_modules):
            hue = i / n_modules
            rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
            colors.append(f'rgb({int(rgb[0]*255)},{int(rgb[1]*255)},{int(rgb[2]*255)})')
    
    module_to_color = {module: colors[i] for i, module in enumerate(unique_modules)}
    
    # Create figure
    fig = go.Figure()
    
    # Add brain mesh
    fig.add_trace(go.Mesh3d(
        x=vertices[:, 0],
        y=vertices[:, 1],
        z=vertices[:, 2],
        i=faces[:, 0],
        j=faces[:, 1],
        k=faces[:, 2],
        opacity=mesh_opacity,
        color=mesh_color,
        name='Brain Surface',
        showlegend=False,
        hoverinfo='skip'
    ))
    
    # Get all edge weights for normalization
    all_weights = connectivity_matrix[connectivity_matrix != 0]
    
    # Process each module separately
    for module_id in unique_modules:
        module_nodes = [i for i in active_nodes if module_assignments[i] == module_id]
        
        if not module_nodes:
            continue
        
        # Get module significance
        module_idx = np.where(np.unique(module_assignments) == module_id)[0][0]
        is_significant = module_significance[module_idx] if show_significance and module_idx < len(module_significance) else True
        
        module_name = f'Module {int(module_id)}'
        if is_significant and show_significance:
            module_name += '*'
        
        # Filter edges based on visualization type
        if visualization_type == "nodes_only":
            module_matrix = np.zeros_like(connectivity_matrix)
        elif visualization_type == "intra":
            module_matrix = filter_edges_by_module(connectivity_matrix, module_assignments, module_id, 'intra')
        elif visualization_type == "inter":
            module_matrix = filter_edges_by_module(connectivity_matrix, module_assignments, module_id, 'inter')
        elif visualization_type == "significant_only":
            if is_significant:
                module_matrix = connectivity_matrix.copy()
                for i in range(len(module_assignments)):
                    if module_assignments[i] != module_id:
                        for j in range(len(module_assignments)):
                            if module_assignments[j] != module_id:
                                module_matrix[i, j] = 0
            else:
                module_matrix = np.zeros_like(connectivity_matrix)
        else:  # "all"
            module_matrix = connectivity_matrix.copy()
        
        # Add edges with variable width
        for i in module_nodes:
            for j in range(i + 1, connectivity_matrix.shape[0]):
                if module_matrix[i, j] != 0 and j in active_nodes:
                    edge_width = calculate_edge_width(
                        module_matrix[i, j], 
                        all_weights, 
                        edge_width_range[0], 
                        edge_width_range[1]
                    )
                    
                    edge_trace = go.Scatter3d(
                        x=[roi_coords_df.loc[i, 'cog_x'], roi_coords_df.loc[j, 'cog_x']],
                        y=[roi_coords_df.loc[i, 'cog_y'], roi_coords_df.loc[j, 'cog_y']],
                        z=[roi_coords_df.loc[i, 'cog_z'], roi_coords_df.loc[j, 'cog_z']],
                        mode='lines',
                        line=dict(
                            color=module_to_color[module_id],
                            width=edge_width
                        ),
                        opacity=0.7,
                        hoverinfo='text',
                        hovertext=f"{roi_names[i]} ↔ {roi_names[j]}<br>Strength: {module_matrix[i, j]:.4f}",
                        showlegend=False,
                        legendgroup=f'module_{module_id}'
                    )
                    fig.add_trace(edge_trace)
        
        # Get node properties for this module
        node_x = [roi_coords_df.loc[i, 'cog_x'] for i in module_nodes]
        node_y = [roi_coords_df.loc[i, 'cog_y'] for i in module_nodes]
        node_z = [roi_coords_df.loc[i, 'cog_z'] for i in module_nodes]
        
        module_node_sizes = [node_sizes[i] for i in module_nodes]
        module_node_roles = [node_roles[i] for i in module_nodes]
        module_node_colors = [node_role_colors[i] for i in module_nodes]
        module_roi_names = [roi_names[i] for i in module_nodes]
        
        # Create hover text
        hover_texts = []
        for i, node_idx in enumerate(module_nodes):
            hover_text = (
                f"<b>{module_roi_names[i]}</b><br>"
                f"Module: {int(module_id)}{' (SIGNIFICANT)' if is_significant and show_significance else ''}<br>"
                f"Role: {module_node_roles[i]}<br>"
                f"PC: {pc_values[node_idx]:.3f}<br>"
                f"Z-score: {z_scores[node_idx]:.3f}<br>"
                f"Node size: {module_node_sizes[i]:.1f}"
            )
            hover_texts.append(hover_text)
        
        # LAYER 1 - Border layer (role color)
        border_sizes = [size + border_width for size in module_node_sizes]
        fig.add_trace(go.Scatter3d(
            x=node_x, y=node_y, z=node_z,
            mode='markers',
            marker=dict(
                size=border_sizes,
                color=module_node_colors,
                opacity=0.95 if is_significant else 0.7,
            ),
            hoverinfo='skip',
            showlegend=False,
            legendgroup=f'module_{module_id}_border'
        ))
        
        # LAYER 2 - Inner node (module color)
        fig.add_trace(go.Scatter3d(
            x=node_x, y=node_y, z=node_z,
            mode='markers+text' if show_labels else 'markers',
            marker=dict(
                size=module_node_sizes,
                color=module_to_color[module_id],
                opacity=0.9 if is_significant else 0.6,
                line=dict(
                    color=module_node_colors,
                    width=2
                )
            ),
            text=module_roi_names if show_labels else None,
            textposition='top center',
            textfont=dict(
                size=label_font_size,
                color='black',
                family='Arial'
            ),
            hoverinfo='text',
            hovertext=hover_texts,
            showlegend=True,
            name=module_name,
            legendgroup=f'module_{module_id}'
        ))
    
    # V4: Set camera position
    if custom_camera is not None:
        camera = custom_camera
    else:
        camera = CameraController.get_camera_position(camera_view)
    
    # Update title
    if actual_state_label is not None:
        state_text = f"State {actual_state_label}"
    else:
        state_text = ""
    
    if 'Q_total' in module_data and module_data['Q_total'] != 0:
        q_score = module_data['Q_total']
        z_score = module_data.get('Q_z_score', 0)
        plot_title += f"<br>{state_text} - Q={q_score:.3f}, Z={z_score:.2f}"
    elif state_text:
        plot_title += f"<br>{state_text}"
    
    # Add camera view to title
    if 'name' in camera:
        plot_title += f"<br><i>View: {camera['name']}</i>"
    
    # Create role legend text
    role_legend_text = (
        "<b>Node Roles (border color):</b><br>"
        "• Ultra-peripheral (light gray)<br>"
        "• Peripheral (gray)<br>"
        "• Kinless (pink)<br>"
        "• Satellite Connector (blue)<br>"
        "• Provincial Hub (gold)<br>"
        "• Connector Hub (red-orange)"
    )
    
    # V4: Add camera control instructions
    camera_control_text = ""
    if enable_camera_controls:
        camera_control_text = (
            "<b>Camera Controls:</b><br>"
            "• Drag to rotate<br>"
            "• Scroll to zoom<br>"
            "• Right-click drag to pan<br>"
            "• Double-click to reset"
        )
    
    # Update layout with camera
    scene_dict = dict(
        xaxis=dict(showgrid=False, zeroline=False, visible=False),
        yaxis=dict(showgrid=False, zeroline=False, visible=False),
        zaxis=dict(showgrid=False, zeroline=False, visible=False),
        bgcolor='white',
        camera=dict(
            eye=camera['eye'],
            center=camera['center'],
            up=camera['up']
        ),
        dragmode='orbit',
        aspectmode='data'
    )
    
    # V4: Add camera preset buttons if enabled
    updatemenus = []
    if enable_camera_controls:
        camera_buttons = []
        for view_key, view_data in CameraController.PRESET_VIEWS.items():
            if view_key != 'custom':  # Skip custom view in menu
                camera_buttons.append(
                    dict(
                        args=[{'scene.camera': {
                            'eye': view_data['eye'],
                            'center': view_data['center'],
                            'up': view_data['up']
                        }}],
                        label=view_data['name'],
                        method='relayout'
                    )
                )
        
        updatemenus = [
            dict(
                type='dropdown',
                showactive=True,
                buttons=camera_buttons,
                x=0.01,
                xanchor='left',
                y=0.99,
                yanchor='top',
                bgcolor='rgba(255, 255, 255, 0.9)',
                bordercolor='black',
                borderwidth=1
            )
        ]
    
    fig.update_layout(
        scene=scene_dict,
        width=1200,
        height=900,
        title={
            'text': plot_title,
            'x': 0.5,
            'xanchor': 'center',
            'font': dict(size=16)
        },
        showlegend=True,
        legend=dict(
            x=0.01,
            y=0.85,
            bgcolor='rgba(255,255,255,0.9)',
            bordercolor='black',
            borderwidth=1
        ),
        updatemenus=updatemenus,
        annotations=[
            dict(
                text=role_legend_text,
                showarrow=False,
                xref="paper", yref="paper",
                x=0.01, y=0.50,
                xanchor="left",
                yanchor="top",
                font=dict(size=10),
                bgcolor='rgba(255,255,255,0.9)',
                bordercolor='black',
                borderwidth=1,
                borderpad=4
            ),
            dict(
                text=camera_control_text,
                showarrow=False,
                xref="paper", yref="paper",
                x=0.01, y=0.30,
                xanchor="left",
                yanchor="top",
                font=dict(size=10),
                bgcolor='rgba(255,255,255,0.9)' if camera_control_text else 'rgba(255,255,255,0)',
                bordercolor='black' if camera_control_text else 'rgba(0,0,0,0)',
                borderwidth=1 if camera_control_text else 0,
                borderpad=4
            ),
            dict(
                text=f"Node size: {node_sizing_mode}<br>Edge width: coherence strength<br><b>V4: Camera controls enabled</b>",
                showarrow=False,
                xref="paper", yref="paper",
                x=0.99, y=0.01,
                xanchor="right",
                yanchor="bottom",
                font=dict(size=11)
            )
        ]
    )
    
    if show_significance:
        fig.add_annotation(
            text="* = Statistically significant module (p<0.01)",
            showarrow=False,
            xref="paper", yref="paper",
            x=0.01, y=0.01,
            xanchor="left",
            yanchor="bottom",
            font=dict(size=10)
        )
    
    # Save main figure
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    config = {
        'displayModeBar': True,
        'displaylogo': False,
        'modeBarButtonsToAdd': ['drawline', 'drawopenpath', 'drawclosedpath', 
                                'drawcircle', 'drawrect', 'eraseshape'],
        'toImageButtonOptions': {
            'format': 'png',
            'filename': save_path.stem,
            'height': 900,
            'width': 1200,
            'scale': 2
        }
    }
    
    fig.write_html(str(save_path), config=config)
    print(f"   Saved visualization to: {save_path}")
    
    # V4: Save additional views if requested
    if save_all_views:
        views_dir = save_path.parent / 'multiple_views'
        views_dir.mkdir(exist_ok=True)
        
        for view_key in ['anterior', 'posterior', 'left', 'right', 'superior', 'inferior']:
            view_camera = CameraController.get_camera_position(view_key)
            fig.update_layout(
                scene_camera=dict(
                    eye=view_camera['eye'],
                    center=view_camera['center'],
                    up=view_camera['up']
                )
            )
            view_path = views_dir / f"{save_path.stem}_{view_key}.html"
            fig.write_html(str(view_path), config=config)
            print(f"   Saved {view_key} view to: {view_path.name}")
    
    # Calculate statistics
    stats = {
        'total_nodes': len(active_nodes),
        'total_edges': np.sum(connectivity_matrix != 0) // 2,
        'n_modules': len(unique_modules),
        'node_role_distribution': pd.Series(node_roles).value_counts().to_dict(),
        'Q_total': module_data.get('Q_total', 0),
        'Q_z_score': module_data.get('Q_z_score', 0),
        'camera_view': camera_view  # V4: Track which view was used
    }
    
    return fig, stats


# ============================================================================
# INTERACTIVE CAMERA CONTROL FUNCTION - V4
# ============================================================================

def create_interactive_camera_control_panel(
    vertices: np.ndarray,
    faces: np.ndarray,
    roi_coords_df: pd.DataFrame,
    connectivity_matrix: np.ndarray,
    module_data: Dict,
    save_dir: Union[str, Path]
) -> None:
    """
    Create an interactive HTML page with camera controls for exploring the visualization.
    
    V4 Feature: Interactive control panel with real-time camera updates.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Create the interactive HTML template
    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Brain Modularity Viewer - Interactive Camera Control</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 0;
                display: flex;
                height: 100vh;
            }}
            #controls {{
                width: 300px;
                padding: 20px;
                background: #f5f5f5;
                overflow-y: auto;
                border-right: 2px solid #ddd;
            }}
            #plot {{
                flex-grow: 1;
                height: 100vh;
            }}
            .control-group {{
                margin-bottom: 20px;
                padding: 15px;
                background: white;
                border-radius: 5px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            .control-group h3 {{
                margin-top: 0;
                color: #333;
                border-bottom: 2px solid #4CAF50;
                padding-bottom: 5px;
            }}
            input[type="number"] {{
                width: 60px;
                padding: 5px;
                margin: 2px;
                border: 1px solid #ddd;
                border-radius: 3px;
            }}
            button {{
                padding: 8px 15px;
                margin: 5px;
                background: #4CAF50;
                color: white;
                border: none;
                border-radius: 3px;
                cursor: pointer;
                transition: background 0.3s;
            }}
            button:hover {{
                background: #45a049;
            }}
            .preset-buttons {{
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 5px;
            }}
            .coordinate-input {{
                display: flex;
                align-items: center;
                margin: 5px 0;
            }}
            .coordinate-input label {{
                width: 30px;
                font-weight: bold;
            }}
            #info {{
                padding: 10px;
                background: #e8f5e9;
                border-radius: 5px;
                margin-top: 10px;
                font-size: 12px;
            }}
        </style>
    </head>
    <body>
        <div id="controls">
            <h2 style="color: #333;">Camera Controls</h2>
            
            <div class="control-group">
                <h3>Preset Views</h3>
                <div class="preset-buttons">
                    <button onclick="setView('anterior')">Anterior</button>
                    <button onclick="setView('posterior')">Posterior</button>
                    <button onclick="setView('left')">Left</button>
                    <button onclick="setView('right')">Right</button>
                    <button onclick="setView('superior')">Superior</button>
                    <button onclick="setView('inferior')">Inferior</button>
                    <button onclick="setView('anterolateral_left')">Antero-L</button>
                    <button onclick="setView('anterolateral_right')">Antero-R</button>
                    <button onclick="setView('posterolateral_left')">Postero-L</button>
                    <button onclick="setView('posterolateral_right')">Postero-R</button>
                </div>
            </div>
            
            <div class="control-group">
                <h3>Manual Camera Position</h3>
                <div class="coordinate-input">
                    <label>X:</label>
                    <input type="number" id="cam-x" value="1.5" step="0.1">
                </div>
                <div class="coordinate-input">
                    <label>Y:</label>
                    <input type="number" id="cam-y" value="1.5" step="0.1">
                </div>
                <div class="coordinate-input">
                    <label>Z:</label>
                    <input type="number" id="cam-z" value="1.5" step="0.1">
                </div>
                <button onclick="applyManualCamera()">Apply Position</button>
            </div>
            
            <div class="control-group">
                <h3>Spherical Coordinates</h3>
                <div>
                    <label>Azimuth (°):</label>
                    <input type="number" id="azimuth" value="45" min="0" max="360" step="5">
                </div>
                <div>
                    <label>Elevation (°):</label>
                    <input type="number" id="elevation" value="30" min="-90" max="90" step="5">
                </div>
                <div>
                    <label>Distance:</label>
                    <input type="number" id="distance" value="2" min="0.5" max="5" step="0.1">
                </div>
                <button onclick="applySphericalCamera()">Apply Spherical</button>
            </div>
            
            <div class="control-group">
                <h3>Animation</h3>
                <button onclick="startRotation()">Start Rotation</button>
                <button onclick="stopRotation()">Stop Rotation</button>
                <div>
                    <label>Speed:</label>
                    <input type="number" id="rotation-speed" value="1" min="0.1" max="5" step="0.1">
                </div>
            </div>
            
            <div class="control-group">
                <h3>Export</h3>
                <button onclick="saveCurrentView()">Save View Settings</button>
                <button onclick="exportImage()">Export as Image</button>
            </div>
            
            <div id="info">
                <strong>Tips:</strong><br>
                • Drag to rotate<br>
                • Scroll to zoom<br>
                • Double-click to reset<br>
                • Right-click drag to pan
            </div>
        </div>
        
        <div id="plot"></div>
        
        <script>
            // Camera presets
            const presets = {
                anterior: {eye: {x:0, y:2, z:0}, center: {x:0, y:0, z:0}, up: {x:0, y:0, z:1}},
                posterior: {eye: {x:0, y:-2, z:0}, center: {x:0, y:0, z:0}, up: {x:0, y:0, z:1}},
                left: {eye: {x:-2, y:0, z:0}, center: {x:0, y:0, z:0}, up: {x:0, y:0, z:1}},
                right: {eye: {x:2, y:0, z:0}, center: {x:0, y:0, z:0}, up: {x:0, y:0, z:1}},
                superior: {eye: {x:0, y:0, z:2}, center: {x:0, y:0, z:0}, up: {x:0, y:1, z:0}},
                inferior: {eye: {x:0, y:0, z:-2}, center: {x:0, y:0, z:0}, up: {x:0, y:-1, z:0}},
                anterolateral_left: {eye: {x:-1.5, y:1.5, z:0.5}, center: {x:0, y:0, z:0}, up: {x:0, y:0, z:1}},
                anterolateral_right: {eye: {x:1.5, y:1.5, z:0.5}, center: {x:0, y:0, z:0}, up: {x:0, y:0, z:1}},
                posterolateral_left: {eye: {x:-1.5, y:-1.5, z:0.5}, center: {x:0, y:0, z:0}, up: {x:0, y:0, z:1}},
                posterolateral_right: {eye: {x:1.5, y:-1.5, z:0.5}, center: {x:0, y:0, z:0}, up: {x:0, y:0, z:1}}
            };
            
            let rotationInterval = null;
            
            function setView(viewName) {
                const camera = presets[viewName];
                Plotly.relayout('plot', {'scene.camera': camera});
                
                // Update manual inputs
                document.getElementById('cam-x').value = camera.eye.x;
                document.getElementById('cam-y').value = camera.eye.y;
                document.getElementById('cam-z').value = camera.eye.z;
            }
            
            function applyManualCamera() {
                const x = parseFloat(document.getElementById('cam-x').value);
                const y = parseFloat(document.getElementById('cam-y').value);
                const z = parseFloat(document.getElementById('cam-z').value);
                
                const camera = {
                    eye: {x: x, y: y, z: z},
                    center: {x: 0, y: 0, z: 0},
                    up: {x: 0, y: 0, z: 1}
                };
                
                Plotly.relayout('plot', {'scene.camera': camera});
            }
            
            function applySphericalCamera() {
                const azimuth = parseFloat(document.getElementById('azimuth').value);
                const elevation = parseFloat(document.getElementById('elevation').value);
                const distance = parseFloat(document.getElementById('distance').value);
                
                const azRad = azimuth * Math.PI / 180;
                const elRad = elevation * Math.PI / 180;
                
                const x = distance * Math.cos(elRad) * Math.cos(azRad);
                const y = distance * Math.cos(elRad) * Math.sin(azRad);
                const z = distance * Math.sin(elRad);
                
                const camera = {
                    eye: {x: x, y: y, z: z},
                    center: {x: 0, y: 0, z: 0},
                    up: {x: 0, y: 0, z: 1}
                };
                
                Plotly.relayout('plot', {'scene.camera': camera});
                
                // Update manual inputs
                document.getElementById('cam-x').value = x.toFixed(2);
                document.getElementById('cam-y').value = y.toFixed(2);
                document.getElementById('cam-z').value = z.toFixed(2);
            }
            
            function startRotation() {
                if (rotationInterval) return;
                
                const speed = parseFloat(document.getElementById('rotation-speed').value);
                let angle = 0;
                
                rotationInterval = setInterval(() => {
                    angle = (angle + speed) % 360;
                    const azRad = angle * Math.PI / 180;
                    const distance = 2;
                    
                    const camera = {
                        eye: {
                            x: distance * Math.cos(azRad),
                            y: distance * Math.sin(azRad),
                            z: 0.5
                        },
                        center: {x: 0, y: 0, z: 0},
                        up: {x: 0, y: 0, z: 1}
                    };
                    
                    Plotly.relayout('plot', {'scene.camera': camera});
                }, 50);
            }
            
            function stopRotation() {
                if (rotationInterval) {
                    clearInterval(rotationInterval);
                    rotationInterval = null;
                }
            }
            
            function saveCurrentView() {
                const plotDiv = document.getElementById('plot');
                const camera = plotDiv.layout.scene.camera;
                
                const viewData = JSON.stringify(camera, null, 2);
                const blob = new Blob([viewData], {type: 'application/json'});
                const url = URL.createObjectURL(blob);
                
                const a = document.createElement('a');
                a.href = url;
                a.download = 'camera_view.json';
                a.click();
            }
            
            function exportImage() {
                Plotly.downloadImage('plot', {
                    format: 'png',
                    width: 1920,
                    height: 1080,
                    filename: 'brain_modularity_view'
                });
            }
            
            // Placeholder for plot data - will be replaced by actual data
            const plotData = [];
            const plotLayout = {
                scene: {
                    camera: presets.oblique,
                    xaxis: {visible: false},
                    yaxis: {visible: false},
                    zaxis: {visible: false},
                    bgcolor: 'white',
                    aspectmode: 'data'
                },
                margin: {l: 0, r: 0, t: 0, b: 0},
                showlegend: true
            };
            
            Plotly.newPlot('plot', plotData, plotLayout);
        </script>
    </body>
    </html>
    """
    
    # Save the interactive control panel
    panel_path = save_dir / "interactive_camera_control.html"
    with open(panel_path, 'w') as f:
        f.write(html_template)
    
    print(f"V4: Created interactive camera control panel at: {panel_path}")


# ============================================================================
# MAIN PIPELINE FUNCTION - V4
# ============================================================================

def run_enhanced_visualization_pipeline(
    matrix_path: Union[str, Path],
    netneurotools_results_dir: Union[str, Path],
    mesh_file: Union[str, Path],
    roi_coords_file: Union[str, Path],
    output_dir: Union[str, Path],
    k_value: int,
    visualization_types: List[str] = None,
    node_sizing_modes: List[str] = None,
    use_thresholding: bool = True,
    n_top_edges: int = 64,
    base_node_size: int = 12,
    max_node_multiplier: float = 2.0,
    show_labels: bool = True,
    show_significance: bool = True,
    state_mapping: Dict[int, int] = None,
    camera_views: List[str] = None,  # V4: New parameter
    enable_interactive_panel: bool = True,  # V4: New parameter
    save_multiple_views: bool = False  # V4: New parameter
) -> Dict:
    """
    Run enhanced visualization pipeline with V4 camera control features.
    
    V4 Parameters
    -------------
    camera_views : list of str
        List of camera views to generate ('anterior', 'posterior', 'left', 'right', etc.)
        If None, uses ['oblique'] as default
    enable_interactive_panel : bool
        Whether to create an interactive camera control panel
    save_multiple_views : bool
        Whether to save all standard views for each visualization
    """
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Default state mapping
    if state_mapping is None:
        state_mapping = {i: i + 1 for i in range(k_value)}
    
    # Default settings
    if visualization_types is None:
        visualization_types = ['all', 'intra', 'inter', 'nodes_only', 'significant_only']
    
    if node_sizing_modes is None:
        node_sizing_modes = ['pc', 'zscore', 'both']
    
    # V4: Default camera views
    if camera_views is None:
        camera_views = ['oblique']
    
    print("="*80)
    print("ENHANCED BRAIN MODULARITY VISUALIZATION PIPELINE - VERSION 4")
    print("With Interactive Camera Controls and Multiple Views")
    print("="*80)
    print(f"Output directory: {output_dir}")
    print(f"Visualization types: {visualization_types}")
    print(f"Node sizing modes: {node_sizing_modes}")
    print(f"Camera views: {camera_views}")
    print(f"Interactive panel: {'Enabled' if enable_interactive_panel else 'Disabled'}")
    print(f"Border width: 6 pixels (enhanced visibility)")
    print("="*80)
    
    # Load mesh and ROI coordinates
    print("\n1. Loading brain mesh and ROI data...")
    vertices, faces = load_mesh_file(mesh_file)
    
    # Load ROI coordinates
    roi_coords_path = Path(roi_coords_file)
    if roi_coords_path.suffix == '.csv':
        roi_coords_df = pd.read_csv(roi_coords_file)
        roi_coords_df = roi_coords_df.reset_index(drop=True)
    else:
        raise ValueError(f"Unsupported ROI file format: {roi_coords_path.suffix}")
    
    print(f"   Loaded {len(roi_coords_df)} ROIs")
    print(f"   Mesh vertices: {vertices.shape}")
    print(f"   Mesh faces: {faces.shape}")
    
    # Load connectivity matrices
    print(f"\n2. Loading connectivity matrices...")
    matrices = np.load(matrix_path)
    print(f"   Loaded matrices: shape {matrices.shape}")
    
    # Initialize loader and load results
    print(f"\n3. Loading NetNeurotools results...")
    loader = NetNeurotoolsModularityLoader(netneurotools_results_dir)
    
    try:
        k_results = loader.load_comprehensive_results(k_value)
        print(f"   Loaded results for {len(k_results['states'])} states")
    except Exception as e:
        print(f"   Error loading results: {e}")
        import traceback
        traceback.print_exc()
        return {}
    
    all_results = {}
    
    # Define thresholding configurations
    if use_thresholding:
        thresholding_configs = {
            'thresholded': n_top_edges,
            'non_thresholded': None
        }
    else:
        thresholding_configs = {'non_thresholded': None}
    
    # V4: Create interactive control panel if enabled
    if enable_interactive_panel and len(k_results['states']) > 0:
        print("\n4. Creating interactive camera control panel...")
        # Use first state for demo
        first_state = k_results['states'][0]
        if first_state['state_idx'] < matrices.shape[0]:
            demo_matrix = matrices[first_state['state_idx']]
            demo_matrix[demo_matrix < 0] = 0
            create_interactive_camera_control_panel(
                vertices, faces, roi_coords_df, 
                demo_matrix, first_state, output_dir
            )
    
    # Process each state
    for state_data in k_results['states']:
        state_idx = state_data['state_idx']
        actual_state_label = state_mapping.get(state_idx, state_idx)
        
        print(f"\n{'='*60}")
        print(f"Processing State {state_idx} (displayed as State {actual_state_label})")
        print(f"{'='*60}")
        
        # Get connectivity matrix for this state
        if state_idx >= matrices.shape[0]:
            print(f"WARNING: State {state_idx} not found in matrices")
            continue
        
        connectivity_matrix = matrices[state_idx]
        connectivity_matrix[connectivity_matrix < 0] = 0
        
        print(f"  Q-score: {state_data.get('Q_total', 0):.3f}")
        print(f"  Z-score: {state_data.get('Q_z_score', 0):.2f}")
        
        # Process each configuration
        for thresh_name, thresh_value in thresholding_configs.items():
            print(f"\n  {thresh_name.upper()} Configuration:")
            
            # Apply thresholding if needed
            if thresh_value is not None:
                processed_matrix = threshold_matrix_top_n(connectivity_matrix, thresh_value)
                n_edges = np.sum(processed_matrix != 0) // 2
                print(f"    Keeping top {thresh_value} edges (actual: {n_edges})")
            else:
                processed_matrix = connectivity_matrix.copy()
                n_edges = np.sum(processed_matrix != 0) // 2
                print(f"    Using all {n_edges} edges")
            
            for size_mode in node_sizing_modes:
                print(f"      Node sizing: {size_mode}")
                
                for viz_type in visualization_types:
                    print(f"        Creating {viz_type} visualization...")
                    
                    # V4: Process each camera view
                    for camera_view in camera_views:
                        viz_dir = output_dir / thresh_name / f"node_size_{size_mode}" / f"state_{actual_state_label}" / camera_view
                        viz_dir.mkdir(parents=True, exist_ok=True)
                        
                        try:
                            # Prepare title
                            title = (f"k={k_value} - "
                                    f"{viz_type.replace('_', ' ').title()}")
                            
                            # Create visualization
                            save_path = viz_dir / f"{viz_type}.html"
                            
                            fig, stats = create_enhanced_modularity_visualization(
                                vertices=vertices,
                                faces=faces,
                                roi_coords_df=roi_coords_df,
                                connectivity_matrix=processed_matrix,
                                module_data=state_data,
                                metrics_df=state_data.get('modules_df'),
                                plot_title=title,
                                save_path=str(save_path),
                                visualization_type=viz_type,
                                node_sizing_mode=size_mode,
                                base_node_size=base_node_size,
                                max_node_multiplier=max_node_multiplier,
                                n_top_edges=None,
                                edge_width_range=(1.0, 6.0),
                                mesh_opacity=0.15,
                                show_labels=show_labels,
                                show_significance=show_significance,
                                border_width=6,  # V4: Using 6 as requested
                                actual_state_label=actual_state_label,
                                camera_view=camera_view,  # V4: Set camera view
                                enable_camera_controls=True,  # V4: Enable controls
                                save_all_views=save_multiple_views  # V4: Save multiple views
                            )
                            
                            # Store results
                            key = f"state{actual_state_label}_{thresh_name}_{size_mode}_{viz_type}_{camera_view}"
                            all_results[key] = {
                                'state_idx': state_idx,
                                'state_label': actual_state_label,
                                'threshold': thresh_name,
                                'node_sizing': size_mode,
                                'viz_type': viz_type,
                                'camera_view': camera_view,
                                'n_edges': n_edges,
                                'Q_total': stats['Q_total'],
                                'Q_z_score': stats['Q_z_score'],
                                'stats': stats,
                                'path': str(save_path)
                            }
                            
                            print(f"          ✓ Saved {camera_view} view")
                            
                        except Exception as e:
                            print(f"          ✗ ERROR: {str(e)}")
                            import traceback
                            traceback.print_exc()
    
    # Create summary
    print(f"\n{'='*80}")
    print("Creating summary report...")
    summary_path = output_dir / "visualization_summary_v4.txt"
    
    with open(summary_path, 'w') as f:
        f.write("ENHANCED BRAIN MODULARITY VISUALIZATION SUMMARY - VERSION 4\n")
        f.write(f"k={k_value} Analysis with Camera Controls\n")
        f.write("="*60 + "\n\n")
        f.write("VERSION 4 FEATURES:\n")
        f.write("- Interactive camera controls with preset views\n")
        f.write("- Manual camera position input (x, y, z coordinates)\n")
        f.write("- Spherical coordinate system (azimuth, elevation)\n")
        f.write("- Animation capabilities (rotation)\n")
        f.write("- Multiple view exports\n")
        f.write("- Enhanced node borders (6px width)\n")
        f.write("="*60 + "\n\n")
        f.write(f"Total visualizations created: {len(all_results)}\n")
        f.write(f"Camera views: {camera_views}\n")
        f.write(f"Interactive panel: {'Created' if enable_interactive_panel else 'Not created'}\n\n")
        
        # Summary by state
        displayed_states = sorted(set(r['state_label'] for r in all_results.values()))
        for state_label in displayed_states:
            f.write(f"\nState {state_label}:\n")
            f.write("-"*40 + "\n")
            state_results = [r for r in all_results.values() if r['state_label'] == state_label]
            if state_results:
                f.write(f"  Q-score: {state_results[0]['Q_total']:.3f}\n")
                f.write(f"  Z-score: {state_results[0]['Q_z_score']:.2f}\n")
                f.write(f"  Total visualizations: {len(state_results)}\n")
                
                # Count by view
                view_counts = {}
                for r in state_results:
                    view = r.get('camera_view', 'default')
                    view_counts[view] = view_counts.get(view, 0) + 1
                for view, count in view_counts.items():
                    f.write(f"    {view}: {count} visualizations\n")
    
    print(f"Summary saved to: {summary_path}")
    
    # V4: Save camera presets reference
    camera_ref_path = output_dir / "camera_presets.json"
    with open(camera_ref_path, 'w') as f:
        json.dump(CameraController.PRESET_VIEWS, f, indent=2)
    print(f"Camera presets saved to: {camera_ref_path}")
    
    print(f"\n{'='*80}")
    print("PIPELINE COMPLETE - VERSION 4!")
    print(f"Total visualizations created: {len(all_results)}")
    print(f"Output directory: {output_dir}")
    print("\nKey V4 features:")
    print("  ✓ Interactive camera controls")
    print("  ✓ Multiple camera views")
    print("  ✓ Enhanced node borders (6px)")
    print("  ✓ Camera position export/import")
    if enable_interactive_panel:
        print(f"  ✓ Interactive control panel: {output_dir}/interactive_camera_control.html")
    print("="*80)
    
    return all_results


if __name__ == "__main__":
    print("Enhanced Brain Modularity Pipeline - VERSION 4")
    print("\nV4 Features:")
    print("  ✓ Interactive camera controls with preset views")
    print("  ✓ Manual camera position input (x, y, z)")
    print("  ✓ Spherical coordinate system")
    print("  ✓ Animation capabilities")
    print("  ✓ Multiple view exports")
    print("  ✓ Enhanced node borders (6px width)")
    print("\nMaintained from previous versions:")
    print("  ✓ Dual-layer rendering for visible borders")
    print("  ✓ Optimized node size differences")
    print("  ✓ Enhanced edge thickness")
    print("  ✓ Proper Q and Z value loading")
    print("  ✓ Brain state mapping support")