#!/bin/sh
source /home/despoB/kaihwang/.bashrc;
source activate mriqc;
#SUB_ID=$(echo ${SGE_TASK} | grep -Eo "^[[:digit:]]{1,}")
#session=$(echo ${SGE_TASK} | grep -Eo "[A-Z][a-zA-Z0-9]{1,}")
SUB_ID=${SGE_TASK} #"${SGE_TASK}";
WD='/home/despoB/kaihwang/Rest/Tha_patients/'
#SCRIPTS='/home/despoB/kaihwang/bin/TTD/Preprocessing'

#nruns=$(/bin/ls ${WD}/QC/reports/sub-${SUB_ID}_ses-${session}_task-TDD_run*_bold.html | wc -l)
#n_raw=$(/bin/ls ${WD}/BIDS/sub-${SUB_ID}/ses-${session}/func/*task-TDD*bold.nii.gz | wc -l)

#if [ "${nruns}" != "${n_raw}" ]; then
mriqc \
    --participant_label ${SUB_ID} \
    -m T1w bold \
    --n_procs 1 \
    --mem_gb 8 \
    --ica \
    --ants-nthreads 1 \
    -w ${WD}/QC/work \
    --verbose-reports \
    ${WD}/BIDS/ \
    ${WD}/QC/ \
    participant group 
#fi

#END_TIME=$(date);
#echo "QC pipeline for subject $SUB_ID completed at $END_TIME";
