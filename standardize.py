import numpy as np
import nibabel as nib
import os

def standardize_nifti(gica_output_path):
    """
    Args:
        gica_output_path : str
                           Path to the directory containing the Group ICA outputs produced by the Group ICA Of fMRI Toolbox (GIFT)
    Returns:
        None
    """
    agg_comp = nib.load(os.path.join(gica_output_path, gica_output_path.split('/')[-1] + '_agg__component_ica_.nii'))
    agg_comp_data = agg_comp.get_fdata()

    agg_comp_stan = np.zeros(shape=agg_comp_data.shape)
    for ic in range(agg_comp_data.shape[3]):
        mean = np.mean(agg_comp_data[:, :, :, ic])
        std = np.std(agg_comp_data[:, :, :, ic])
        agg_comp_stan[:, :, :, ic] = (agg_comp_data[:, :, :, ic] - mean) / std

    agg_comp_stan_nii = nib.Nifti1Image(agg_comp_stan, agg_comp.affine, header=agg_comp.header)
    nib.save(agg_comp_stan_nii, os.path.join(gica_output_path, gica_output_path.split('/')[-1] + '_agg__component_ica_standardized_.nii'))