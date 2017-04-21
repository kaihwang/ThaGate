SCRIPT='/home/despoB/kaihwang/bin/ThaGate'
DATA='/home/despoB/kaihwang/bin/HCP-processing/Data'

cd /home/despoB/connectome-data
for s in 100206; do  #$(/bin/ls -d *)
	#if [ ! -e "/home/despoB/kaihwang/Rest/Graph/gsetCI_${Subject}.mat" ]; then
	sed "s/s in 100206/s in ${s}/g" < ${SCRIPT}/python_qsub.sh > ~/tmp/dFC_graph${s}.sh
	qsub -l mem_free=8G -V -M kaihwang -m e -e ~/tmp -o ~/tmp ~/tmp/dFC_graph${s}.sh
	#fi

done

