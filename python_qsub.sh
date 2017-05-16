#!/bin/bash


#HCP
# cd /home/despoB/connectome-data
# for s in 100206; do  #$(/bin/ls -d *)
# 	for roi in Morel_plus_Yeo400; do #Morel_plus_Yeo17 

# 		for sequence in rfMRI_REST1_LR rfMRI_REST1_RL rfMRI_REST2_LR rfMRI_REST2_RL EMOTION_LR EMOTION_RL GAMBLING_LR GAMBLING_RL MOTOR_LR MOTOR_RL SOCIAL_LR SOCIAL_RL LANGUAGE_LR LANGUAGE_RL WM_LR WM_RL RELATIONAL_LR RELATIONAL_RL; do
# 			echo "${s} ${sequence} ${roi} 15" | python /home/despoB/kaihwang/bin/ThaGate/dFC_graph.py
# 		done
# 	done
# done


#NKI
cd /home/despoB/kaihwang/Rest/NKI
for s in 0102826_session_1; do  #$(/bin/ls -d *)
	for roi in Morel_plus_Yeo400; do #Morel_plus_Yeo17 

		for sequence in 1400; do
			#echo "${s} ${sequence} ${roi} 11" | python /home/despoB/kaihwang/bin/ThaGate/dFC_graph.py
			echo "${s} ${sequence}" | python /home/despoB/kaihwang/bin/ThaGate/pcorr.py
		done

		for sequence in 645; do
			#echo "${s} ${sequence} ${roi} 16" | python /home/despoB/kaihwang/bin/ThaGate/dFC_graph.py
			echo "${s} ${sequence}" | python /home/despoB/kaihwang/bin/ThaGate/pcorr.py
		done

	done
done
