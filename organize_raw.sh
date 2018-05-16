

WD='/home/despoB/kaihwang/Rest/Tha_patients/Raw/dicoms_newvisit/'

for s in 196; do
	
	heudiconv -d ${WD}/{subject}/*/*/* -s ${s} \
	-f /home/despoB/kaihwang/bin/ThaGate/TTD_heuristics.py -c dcm2niix -o /home/despoB/kaihwang/Rest/Tha_patients/BIDS --bids

done

