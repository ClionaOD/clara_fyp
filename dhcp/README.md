# 2020-02-28
40 week template from here: https://gin.g-node.org/BioMedIA/dhcp-volumetric-atlas-groupwise/src/master/mean/ga_40
Registered template_t2.gz to Q1-Q6_RelatedParcellation210_AverageT2w_restore.nii.gz from HCP/Glasser
Using FNIRT: template_t2_to_Q1-Q6_fnirt.nii.gz, registration generally poor 
Using ants: antsregWarped.nii.gz, registration mostly good, but two notable problems:
  (1) The DHCP t2 is bright around the edge of the brain due to a thick layer of CSF. This CSF gets registered to the edge of the adult brain, so the registered infant brain is "too small" relative to the adult one
  (2) The infant cerebellum is much smaller, so the visual cortex gets pulled down

I thought that maybe using the T1, where CSF is not bright, might be better. So, I tried registering template_t1.gz to MNI152_T1_1mm_brain.nii.gz

Using ants: antsreg_t1Warped.nii.gz

This has the same two problems above. Although the CSF is not bright, there is a layer of what I think is dura (although maybe it is a bit of scalp) obvious in the infant template, which gets registered to the edge of the adult brain. 

So, I trimmed off dura (I think, not scalp) by using tissue_04.nii mask from [ga_40 template](https://gin.g-node.org/BioMedIA/dhcp-volumetric-atlas-groupwise/src/master/tissues/ga_40) like this:
fslmaths tissue_04.nii.gz -thr 1000 tissue_04_thr_1000.nii.gz
fslmaths tissue_04_thr_1000.nii.gz -binv tissue_04_thr_1000_binv.nii.gz
fslmaths template_t1.nii.gz -mul tissue_04_thr_1000_binv.nii.gz tempate_t1_nodura.nii.gz

Registered using ants 
 ./antsRegistrationSyN.sh -d 3 -f MNI152_T1_1mm_brain.nii.gz -m tempate_t1_nodura.nii.gz -o antsreg_t1_nodura

The resulting image is 
antsreg_t1_noduraWarped.nii.gz

This is better with regards to the edge of the cortex - it is mostly good now, but the cerebellum is a big problem. This is disrupting the registration of visual cortex, which will be a problem for Laura.

# 2020-02-29
Trying removing cerebellum before reg
fslmaths atlases/dhcp_volume_40weeks/tissue_06.nii.gz -thr 300 tissue_06_thr_300.nii.gz
fslmaths tissue_06_thr_300.nii.gz -binv tissue_06_thr_300_binv.nii.gz
fslmaths atlases/dhcp_volume_40weeks/template_t1_nodura.nii.gz -mul tissue_06_thr_300_binv.nii.gz template_t1_nodura_nocerebllum.nii.gz
 
[sorry mistyped cerebellum in filenames]
 
This didn't work too well, as the infant occipital lobe got dragged down into the adult cerebellum
 
So, I next tried removing the cerebellum from the adult MNI as well as the infant template. I did this by dowloading [AAL3](https://www.oxcns.org/aal3.html) and then creating a mask from regions 95-120 inclusive, inverting it and apply it to the MNI152.
 
The resulting normalised infant t1 with no dura or cerebellum here:
antsreg_t1_nodura_nocerebllum_in_templateWarped.nii.gz
 
The forward warp is in two files,
antsreg_t1_nodura_nocerebllum_in_template1Warp.nii.gz
antsreg_t1_nodura_nocerebllum_in_template0GenericAffine.mat
which can be applied like this:
 
antsApplyTransforms -i atlases/dhcp_volume_40weeks/template_t1.nii.gz -r atlases/fsl/MNI152_T1_1mm_brain.nii.gz -t antsreg_t1_nodura_nocerebllum_in_template1Warp.nii.gz -t antsreg_t1_nodura_nocerebllum_in_template0GenericAffine.mat  -o antstransformed.nii.gz
 
The output from this one, the warp applied to the orignal T1 with the dura and cerebellum, is here:
antstransformed.nii.gz


# 2020-02-29
Meanwhile, Laura discovered this thread regarding aligning dHCP individual subject surfaces with the HCP fsaverage32 surface.
https://neurostars.org/t/how-to-allign-rois-from-32k-surface-template-to-surface-native-dhcp-surface-with-84k-vertices/6183
This refers to this code
https://github.com/ecr05/dHCP_template_alignment

If you go to the dhcp anatomy for a single subject, e.g., 
/dhcp/dhcp_anat_pipeline/sub-CC00667XX16/ses-196800/anat
and run the workbench viewer
$ wb_view
then accept the default spec and press load, you can see the surfaces and the corresponding T2 volumes. They're very pretty! However, it looks to me as if the T2 volumes and the diffusion data aren't quite in registration. I'm not sure if the surface tempplate (https://brain-development.org/brain-atlases/atlases-from-the-dhcp-project/cortical-surface-atlas-bozek/) is supposed to be in registration with the volume template (https://gin.g-node.org/BioMedIA/dhcp-volumetric-atlas-groupwise/src/master/tissues/ga_40). Perhaps one of you could look into this?



