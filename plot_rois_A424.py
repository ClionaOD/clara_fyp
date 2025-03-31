import nibabel as nib
import numpy as np
from nilearn import plotting
import os
import pandas as pd
import glob
import matplotlib.pyplot as plt

template_path = "./data/nihpd_asym_02-05_t2w_2mm.nii.gz"
plot_output_folder = "./regression_rois_img/"
os.makedirs(plot_output_folder, exist_ok=True)

A424_atlas = nib.load('/home/claraconyngham/clara_fyp/atlases/A424_space-nihpd-02-05_2mm.nii.gz')
A424_atlas_data = A424_atlas.get_fdata()

# Load the data from the pickle files
results_files = glob.glob(f"*_results.pickle")

# Make a mask out of the top 10 highest coefficients and the top 10 lowest coefficients for each pickle file
for results_file in results_files:
    results = pd.read_pickle(results_file)
    coefficients = results['coefficients']
    
    non_zero_indices = np.nonzero(coefficients['Coefficient'].values)[0]
    non_zero_coefficients = coefficients.iloc[non_zero_indices]
    sorted_coefficients = non_zero_coefficients.sort_values(by='Coefficient', ascending=False)
    
    top_20_highest = sorted_coefficients.head(20)
    top_20_lowest = sorted_coefficients.tail(20)

    combined_indices = np.concatenate((top_20_highest.index, top_20_lowest.index))
    
    # Create a mask for the combined indices
    combined_indices_mask = np.isin(A424_atlas_data, combined_indices)
    combined_indices_data = np.zeros_like(A424_atlas_data, dtype=np.float32)
    
    # Set the values in the mask to the corresponding coefficient values
    for idx in combined_indices:
        combined_indices_data[A424_atlas_data == idx] = coefficients.loc[idx, 'Coefficient']

    combined_indices_img = nib.Nifti1Image(combined_indices_data, A424_atlas.affine)

    
    # Plot the mask
    base_name = os.path.basename(results_file).replace('_results.pickle', '').capitalize()
    display = plotting.plot_stat_map(
        combined_indices_img,
        bg_img=template_path,
        vmax = 0.35,
        title="",
        display_mode="ortho",
    )
    plt.gcf().text(0.32, 0.95, f"{base_name}: Location of highest and lowest weighted coefficients", fontsize=10, color='white', backgroundcolor='black', ha='center')
    # Save the plot
    output_file = os.path.join(plot_output_folder, f"{os.path.basename(results_file).replace('_results.pickle', '')}_20_mask.png")
    display.savefig(output_file)
    display.close()
    print(f"Plot saved: {output_file}")







