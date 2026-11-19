import nibabel as nib
import numpy as np
from scipy.ndimage import distance_transform_edt

anat_img = nib.load("anatomy.nii")
fiber_img = nib.load("matched_output.nii")

anat_mask = anat_img.get_fdata() > 0
fiber_data = fiber_img.get_fdata()
has_direction = np.linalg.norm(fiber_data, axis=-1) > 1e-6

# Find nearest valid voxel index for every missing voxel
_, nearest_indices = distance_transform_edt(
    ~has_direction, return_indices=True
)

filled_data = fiber_data.copy()
missing_mask = anat_mask & (~has_direction)
for c in range(3):
  filled_data[..., c][missing_mask] = fiber_data[..., c][
      tuple(idx[missing_mask] for idx in nearest_indices)
  ]

nib.save(
    nib.Nifti1Image(filled_data, anat_img.affine, anat_img.header),
    "matched_output_filled.nii",
)
