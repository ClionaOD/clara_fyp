#  use https://nilearn.github.io/dev/modules/generated/nilearn.plotting.plot_stat_map.html#nilearn.plotting.plot_stat_map
#  to plot the statistical maps in img_results folder
#  onto the template found in data/nihpd_asym_02-05_fcgmask.nii.gz
#  use this demo https://nilearn.github.io/dev/auto_examples/01_plotting/plot_demo_plotting.html#sphx-glr-auto-examples-01-plotting-plot-demo-plotting-py

from nilearn import plotting
import os
import nibabel as nib
import glob

stat_maps_folder = "./img_results/"
# Path to the template
template_path = "./data/nihpd_asym_02-05_t2w_2mm.nii.gz"
# Output directory for the plots
plot_output_folder = "./img_plots/"
os.makedirs(plot_output_folder, exist_ok=True)

# Get all .nii.gz files in the stat maps folder
# stat_map_files = [f for f in os.listdir(stat_maps_folder) if f.endswith(".nii.gz")]
# stat_map_files = glob.glob(f"{stat_maps_folder}/*.nii.gz")
stat_map_files = glob.glob(f"{stat_maps_folder}/*_percentilethresholded.nii.gz")

# Loop through each statistical map
for stat_map_file in stat_map_files:
    # stat_map_path = os.path.join(stat_maps_folder, stat_map_file)
    stat_map_path = stat_map_file

    # Plot the statistical map on the template
    # output_file = os.path.join(plot_output_folder, f"{stat_map_file}.png")
    # output_file = os.path.join(
    #     plot_output_folder, stat_map_file.replace(".nii.gz", ".png")
    # )
    output_file = stat_map_path.replace(".nii.gz", ".png")

    display = plotting.plot_stat_map(
        stat_map_path,
        bg_img=template_path,
        title=f"Statistical Map: {stat_map_file}",
        # threshold=0.1,  # Adjust the threshold as needed
        # cut_coords=(0, 0, 0),  # You can customize the cut coordinates
        display_mode="ortho",  # Use 'ortho' for orthogonal views
    )

    # Save the plot
    display.savefig(output_file)
    display.close()

    print(f"Plot saved: {output_file}")
