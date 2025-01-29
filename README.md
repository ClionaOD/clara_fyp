## REPO DATA DEPENDENCIES

atlases/ -> this directory should have brain parcellation files, in our various age spaces.

-   I'm currently pushing these using Git (1.5 Mb), but if file sizes increase we should review.

registration/ -> this is where registration scripts and resulting warp files are stored.

-   due to large file size, only the code is being synced using Git and output files must be acquired elsewhere.

templates/ -> these are the anatomical templates.

-   I'm currently pushing these using Git (30 Mb), but if file sizes increase we should review.

## FIRST STEP TO CONVERT SEEDS AND TARGETS INTO CORRECT SPACE FOR TRACTOGRAPHY

tractography_prepare/

run_preparerois.sh

-   Takes the seeds and targets (should be in atlases/) from dHCP40wk_commonspace and transforms them to each neonate's native space.
-   The target atlas will be split into individual regions, the seeds will not. - Be sure to check "TODO" flags on this file.
-   To run it, use the command line:
    ```console
    claraconyngham@ip-10-0-2-64:~$ ./tractography_prepare/run_preparerois.sh sub-CC00124XX09 ses-42302
    ```

submit_all_preparerois.sh

-   Runs through every subject in the dhcp dataset, and does run_preparerois.sh on them.

    ```console
    claraconyngham@ip-10-0-2-64:~$ ./tractography_prepare/submit_all_preparerois.sh
    ```

-   If the region files have already been prepared, it will not run again.
-   This file uses the command "sbatch" in lines 21. This submits all the jobs using slurm. By doing this, we make the most of our computing resources and schedule when the jobs will run in an efficient manner.
-   To check the progress of your job, use the command

    ```console
    claraconyngham@ip-10-0-2-64:~$ squeue
    ```

## NEXT, RUN TRACTOGRAPHY

tractography_run/

run_probtrackx2.sh

-   Similar to first step, this will run the tractography on a single subject.
-   Check all paths and filenames.
-   This will take a lot longer for each subject, so it's best to run in slurm.

    ```console
    claraconyngham@ip-10-0-2-64:~$ sbatch ./tractography_run/run_probtrackx2.sh sub-CC00124XX09 ses-42302
    ```

submit_all_probtrackx2.sh

-   As with preparerois, this will do it for all subjects.

    ```console
    claraconyngham@ip-10-0-2-64:~$ ./tractography_run/submit_all_probtrackx2.sh
    ```

## THEN TRANSFORM BACK TO COMMON SPACE FROM NATIVE

run_transform_back.sh

-   This is to go from dHCP40wk_nativespace to dHCP40wk_commonspace
-   Check line 12 file name, this should be the name of the outputs from probtrackx2. It's looking for seeds_to\_\* which means all files in probtrackx2_clara that start with seeds_to\_
-   Change -r on line 21 to be the template_t1.nii.gz file in clara_fyp/templates

    ```console
    claraconyngham@ip-10-0-2-64:~$ sbatch ./tractography_run/run_transform_back.sh sub-CC00124XX09 ses-42302
    ```

submit_all_transform_back.sh

-   Same idea, this will do all subjects.

    ```console
    claraconyngham@ip-10-0-2-64:~$ ./tractography_run/submit_all_transform_back.sh
    ```
