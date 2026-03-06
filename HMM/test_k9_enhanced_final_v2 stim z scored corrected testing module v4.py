"""
Test Script for Enhanced Brain Modularity Pipeline - FINAL VERSION
====================================================================
Process k9 data with all fixes including state mapping.
"""

import sys
import os
from pathlib import Path
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Import the enhanced module
import brain_modularity_pipeline_enhanced_final_v4 as bmpe


def process_all_k_values():
    """
    Process data for k=5 to k=15 with appropriate state mappings.
    """
    
    # Define state mappings for each k value
    
    
    state_mappings = {
        5: {0: 0, 1: 1, 2: 2, 3: 3, 4: 4},
        6: {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5},
        7: {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5},
        8: {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 6: 5, 7: 6},
        9: {0: 0, 1: 1, 2: 2, 5: 3, 7: 4},
        10: {0: 0, 1: 1, 3: 2, 4: 3, 5: 4, 8: 5},
        11: {0: 0, 3: 1, 4: 2, 5: 3, 9: 4},
        12: {0: 0, 1: 1, 3: 2, 5: 3, 6: 4, 8: 5, 11: 6},
        13: {2: 0, 6: 1, 8: 2, 9: 3, 10: 4},
        14: {1: 0, 5: 1, 7: 2, 8: 3, 10: 4, 11: 5, 12: 6},
        15: {1: 0, 3: 1, 5: 2, 7: 3, 11: 4, 13: 5, 14: 6}

    }
    
    
    #state_mappings = {5: {0: 0, 1: 1, 2: 2, 3: 3, 4: 4}}
    # Base paths
    base_matrix_dir = Path(r"G:\My Drive\gmvae_stim_experiments_v2\experiment_k5_15_combined_20251114_163203\omst_filter_matrices\stim_matrices")
    netneurotools_dir = Path(r"G:\My Drive\gmvae_stim_experiments_v2\experiment_k5_15_combined_20251114_163203\modularity_significance_analysis_omst_pos")
    mesh_file = r"C:\Users\Azad Azargushasb\Research\brain_filled_3_smoothed.gii"
    roi_coords_file = r"G:\My Drive\research stim data cci\atlas_114_mapped_comma.csv"
    base_output_dir = Path(r"G:\My Drive\gmvae_stim_experiments_v2\experiment_k5_15_combined_20251114_163203\stim_modularity_visualizations_v4_all_k")
    
    print("="*80)
    print("ENHANCED BRAIN MODULARITY VISUALIZATION - BATCH PROCESSING")
    print("Processing STIM Data for k=5 to k=15")
    print("="*80)
    print("\nFixes applied:")
    print("  ✓ Optimized node size differences (2x multiplier - 1/3 of previous)")
    print("  ✓ Enhanced edge thickness (1.0 to 6.0 range)")
    print("  ✓ Proper Q and Z value loading from CSV")
    print("  ✓ Thicker borders (8px) for clear visibility")
    print("  ✓ Proper mesh loading with intent checking")
    print("  ✓ State mapping for each k value")
    print("="*80)
    
    all_results = {}
    
    # Process each k value
    for k_value in range(5, 16):
        print(f"\n{'='*60}")
        print(f"PROCESSING k={k_value}")
        print(f"{'='*60}")
        
        # Get state mapping for this k
        state_mapping = state_mappings[k_value]
        print(f"State mapping: {state_mapping}")
        print(f"Active clusters: {len(state_mapping)}/{k_value}")
        
        # Construct paths for this k
        matrix_path = base_matrix_dir / f"k{k_value}_strategy_A_positive.npy"
        output_dir = base_output_dir / f"k{k_value}"
        
        # Check if matrix file exists
        if not matrix_path.exists():
            print(f"✗ Matrix file not found: {matrix_path}")
            print(f"  Skipping k={k_value}")
            continue
        
        print(f"✓ Found matrix file: {matrix_path}")
        print(f"  Output directory: {output_dir}")
        
        # Run the enhanced pipeline
        try:
            results = bmpe.run_enhanced_visualization_pipeline(
                matrix_path=str(matrix_path),
                netneurotools_results_dir=str(netneurotools_dir),
                mesh_file=mesh_file,
                roi_coords_file=roi_coords_file,
                output_dir=str(output_dir),
                k_value=k_value,
                visualization_types=['all', 'intra', 'inter', 'nodes_only', 'significant_only'],
                node_sizing_modes=['pc', 'zscore', 'both'],
                use_thresholding=True,
                n_top_edges=64,
                base_node_size=12,  # Increased base size
                max_node_multiplier=2.0,  # Reduced to 1/3 of previous (was 6.0, now 2.0)
                show_labels=True,
                show_significance=True,
                state_mapping=state_mapping,
                camera_views=['oblique'],
                enable_interactive_panel=True,
                save_multiple_views=False    
            )
            
            if results:
                all_results[k_value] = results
                print(f"✓ Successfully created {len(results)} visualizations for k={k_value}")
                
                # Get unique states with their Q and Z values
                state_info = {}
                for key, data in results.items():
                    state_label = data['state_label']
                    if state_label not in state_info:
                        state_info[state_label] = {
                            'Q': data['Q_total'],
                            'Z': data['Q_z_score'],
                            'count': 0
                        }
                    state_info[state_label]['count'] += 1
                
                print(f"\n  States summary for k={k_value}:")
                for state_label in sorted(state_info.keys()):
                    info = state_info[state_label]
                    print(f"    State {state_label}: Q={info['Q']:.3f}, Z={info['Z']:.2f}, Visualizations={info['count']}")
            else:
                print(f"✗ No visualizations created for k={k_value}")
                
        except Exception as e:
            print(f"✗ Error processing k={k_value}: {str(e)}")
            continue
    
    # Final summary
    print(f"\n{'='*80}")
    print("BATCH PROCESSING COMPLETE - FINAL SUMMARY")
    print(f"{'='*80}")
    
    if all_results:
        print(f"✓ Successfully processed {len(all_results)} k values:")
        total_visualizations = 0
        for k_value in sorted(all_results.keys()):
            num_viz = len(all_results[k_value])
            total_visualizations += num_viz
            print(f"  k={k_value}: {num_viz} visualizations")
        
        print(f"\n✓ Total visualizations created: {total_visualizations}")
        print(f"\nOutput structure: {base_output_dir}/")
        print("  k5/, k6/, k7/, ..., k15/")
        print("    thresholded/")
        print("      node_size_pc/, node_size_zscore/, node_size_both/")
        print("        state_0/, state_1/, state_2/, ...")
        print("          all.html, intra.html, inter.html, nodes_only.html, significant_only.html")
        print("    non_thresholded/")
        print("      (same structure)")
    else:
        print("✗ No visualizations were created. Check errors above.")
    
    return all_results


if __name__ == "__main__":
    # Run the batch processing for all k values
    results = process_all_k_values()
    
    if results:
        print(f"\n{'='*80}")
        print("✓ ALL BATCH PROCESSING COMPLETE!")
        print("✓ All fixes have been applied successfully")
        print(f"✓ Processed {len(results)} k values with visualizations")
        print(f"{'='*80}")