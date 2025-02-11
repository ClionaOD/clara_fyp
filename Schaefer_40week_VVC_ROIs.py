import nibabel as nib
import numpy as np

Schaefer_40week_fullbrain = nib.load('/home/claraconyngham/clara_fyp/data/Schaefer_in_dHCP_space.nii.gz')
Schaefer_40week_fullbrain_data = Schaefer_40week_fullbrain.get_fdata()


VVC_values = np.array([1, 2, 3, 4, 5, 7, 8, 10, 11, 13, 16, 201, 202, 203, 204, 205, 206, 207, 209, 211, 212, 214])
VVC_mask = np.isin(Schaefer_40week_fullbrain_data, VVC_values)

Schaefer_40week_VVC_data = np.where(VVC_mask, Schaefer_40week_fullbrain_data, np.nan)


Schaefer_40week_VVC_data = np.zeros_like(Schaefer_40week_fullbrain_data)
Schaefer_40week_VVC_data[VVC_mask] = Schaefer_40week_fullbrain_data[VVC_mask]


Schaefer_40week_VVC = nib.Nifti1Image(Schaefer_40week_VVC_data, Schaefer_40week_fullbrain.affine)
nib.save(Schaefer_40week_VVC, '/home/claraconyngham/clara_fyp/data/schaefer_40weeks_VVC.nii.gz')