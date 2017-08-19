SCRIPT='/home/despoB/kaihwang/bin/ThaGate'

cd /home/despoB/connectome-data
for s in $(cat ~/bin/ThaGate/HCP_subjlist); do  #$(/bin/ls -d *)
	#if [ ! -e "/home/despoB/kaihwang/Rest/Graph/gsetCI_${Subject}.mat" ]; then
	sed "s/s in 100206/s in ${s}/g" < ${SCRIPT}/python_qsub.sh > ~/tmp/dFC_graph${s}.sh
	qsub -l mem_free=7G -V -M kaihwang -m e -e ~/tmp -o ~/tmp ~/tmp/dFC_graph${s}.sh
	#fi
done

# cd /home/despoB/kaihwang/Rest/NKI
# for s in $(/bin/ls -d *); do  #$(/bin/ls -d *)
# 	#if [ ! -e "/home/despoB/kaihwang/Rest/Graph/gsetCI_${Subject}.mat" ]; then
# 	sed "s/s in 0102826_session_1/s in ${s}/g" < ${SCRIPT}/python_qsub.sh > ~/tmp/dFC_graph${s}.sh
# 	qsub -l mem_free=5G -V -M kaihwang -m e -e ~/tmp -o ~/tmp ~/tmp/dFC_graph${s}.sh
# 	#fi
# done


