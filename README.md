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
    claraconyngham@ip-10-0-2-64:~$ ./tractography_prepare/run_preparerois.sh
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
