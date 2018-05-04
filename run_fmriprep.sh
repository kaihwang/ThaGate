#!/bin/sh
source /home/despoB/kaihwang/.bashrc;
source activate fmriprep1.0;
SUB_ID=${SGE_TASK}
WD='/home/despoB/kaihwang/Rest/Tha_patients/'
#SCRIPTS='/home/despoB/kaihwang/bin/TTD/Preprocessing'

#cd ${WD}/fmriprep;

fmriprep \
    --participant_label $SUB_ID \
    --nthreads 4 \
    --output-space T1w template \
    --template MNI152NLin2009cAsym \
    --fs-no-reconall \
    --force-no-bbr \
    ${WD}/BIDS/ \
    ${WD}/fmriprep/ \
    participant

#END_TIME=$(date);
#echo "fMRIprep for $SUB_ID completed at $END_TIME";