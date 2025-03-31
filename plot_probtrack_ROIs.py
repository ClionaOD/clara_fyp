from nilearn import plotting
import nibabel as nib
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

template_path = "templates/dHCP40wk_template_t1.nii.gz"
subject = "CC00062XX05"
session = "13801"
roi = "194"



atlas_path = "/home/claraconyngham/clara_fyp/atlases/A424_in_dHCP40wk_commonspace.nii.gz"
output_mask_path = "roi_194_mask.nii.gz"

atlas_img = nib.load(atlas_path)
atlas_data = atlas_img.get_fdata()

roi_mask_data = (atlas_data == int(roi)).astype(np.uint8)
roi_mask_img = nib.Nifti1Image(roi_mask_data, affine=atlas_img.affine)


# Function to generate the results path
def get_results_path(subject, session, roi):
    return f"/dhcp/dhcp_dmri_pipeline/sub-{subject}/ses-{session}/probtrackx2_clara/seeds_to_sub-{subject}_ses-{session}_A424_target_regions_dwi_{roi}_40wk.nii.gz"

results_path = get_results_path(subject, session, roi)

results_img = nib.load(results_path)




# Define the output directory and file
output_dir = "probtrack_ROIs"
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, f"ROI_probtrack_{subject}_{session}_{roi}_withROI.png")

# Plot the ROI with a colorbar for intensity
template_img = nib.load(template_path)

 
display = plotting.plot_stat_map(
    results_img,
    bg_img=template_img,  
    display_mode="ortho",
    colorbar=False,  
    cmap="roy_big_bl", 
    vmax=5000,
)

display.add_overlay(roi_mask_img, cmap="cool", alpha=0.7)  # Use a different colormap for the ROI

# Customize the color bar to avoid exponential notation
#cbar = display._cbar
#cbar.formatter = ScalarFormatter()
#cbar.formatter.set_scientific(False)
#cbar.update_ticks()

plt.subplots_adjust(left=0.0001, right=0.99, top=0.9, bottom=0.1)  # Adjust these values as needed
# Set the title with a smaller font size using Matplotlib
#plt.gcf().text(0.5, 0.95, "Connectivity from VVC to ROI 181", fontsize=10, ha='center')
plt.gcf().text(0.6, 0.95, "Connectivity from VVC to ROI 194", fontsize=13, color='white', backgroundcolor='black', ha='right')
# Save the plot
display.savefig(output_file)
display.close()

print(f"Output saved to {output_file}")

