import sys
from FuncParcel import *


def function_cal_pcorr_mat(subject , sequence, roi):
	ts_path = '/home/despoB/kaihwang/Rest/ThaGate/NotBackedUp/'
	pcorr_path = '/home/despoB/kaihwang/Rest/ThaGate/Matrices/'
	fn = ts_path + subject + '_%s_%s_000.netts' %(roi, sequence)
	ts = np.loadtxt(fn)
	
	cortical_roi_ts = ts[0:400,:] # this is assuming Yeo400 ROI parcellation
	thalamus_ts = ts[400:,:]

	pcorr_mat = pcorr_subcortico_cortical_connectivity(thalamus_ts, cortical_roi_ts)
	
	fn = pcorr_path + subject + '_' + roi + '_' + sequence + '_pcorr_mat'
	#np.savetxt(fn, pcorr_mat, fmt='%.4f')
	np.savetxt(fn, pcorr_mat)


subject, sequence, roi = raw_input().split()
function_cal_pcorr_mat(subject, sequence, roi)