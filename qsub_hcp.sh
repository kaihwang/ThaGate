SCRIPT='/home/despoB/kaihwang/bin/ThaGate'

#HCP
# cd /home/despoB/connectome-data
# for s in $(cat ~/bin/ThaGate/HCP_subjlist); do  #$(/bin/ls -d *)  #/home/despoB/kaihwang/bin/HCP-processing/Data/unrelated.csv is the same
# 	#if [ ! -e "/home/despoB/kaihwang/Rest/Graph/gsetCI_${Subject}.mat" ]; then
# 	sed "s/s in 100307/s in ${s}/g" < ${SCRIPT}/python_qsub.sh > ~/tmp/dFC_graph${s}.sh
# 	qsub -l mem_free=7G -V -M kaihwang -m e -e ~/tmp -o ~/tmp ~/tmp/dFC_graph${s}.sh
# 	#fi
# done

#NKI
# cd /home/despoB/kaihwang/Rest/NKI
# for s in $(/bin/ls -d *); do  #$(/bin/ls -d *)
# 	#if [ ! -e "/home/despoB/kaihwang/Rest/Graph/gsetCI_${Subject}.mat" ]; then
# 	sed "s/s in 0102826_session_1/s in ${s}/g" < ${SCRIPT}/python_qsub.sh > ~/tmp/p_graph${s}.sh
# 	qsub -l mem_free=5G -V -M kaihwang -m e -e ~/tmp -o ~/tmp ~/tmp/p_graph${s}.sh
# 	#fi
# done


#TDSigEI
# cd /home/despoB/kaihwang/TRSE/TDSigEI
# for s in $(/bin/ls -d 5*); do  #$(/bin/ls -d *)
# 	#if [ ! -e "/home/despoB/kaihwang/Rest/Graph/gsetCI_${Subject}.mat" ]; then
# 	sed "s/s in 503/s in ${s}/g" < ${SCRIPT}/python_qsub.sh > ~/tmp/${s}.sh
# 	echo "bash ~/tmp/${s}.sh TDSigEI" > ~/tmp/dFC_graph${s}.sh
# 	qsub -l mem_free=5G -V -M kaihwang -m e -e ~/tmp -o ~/tmp ~/tmp/dFC_graph${s}.sh
# 	#fi
# done




#TRSE
#cd /home/despoB/kaihwang/TRSE/TDSigEI
for s in $(cat /home/despoB/kaihwang/bin/ThaGate/TRSE_subject); do  #$(/bin/ls -d *)
	#if [ ! -e "/home/despoB/kaihwang/Rest/Graph/gsetCI_${Subject}.mat" ]; then
	sed "s/s in 1106/s in ${s}/g" < ${SCRIPT}/python_qsub.sh > ~/tmp/${s}.sh
	echo "bash ~/tmp/${s}.sh TRSE" > ~/tmp/dFC_graph${s}.sh
	qsub -l mem_free=3.5G -V -M kaihwang -m e -e ~/tmp -o ~/tmp ~/tmp/dFC_graph${s}.sh
	#fi
done
