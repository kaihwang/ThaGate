import sys
from FuncParcel import *
import bct
import os
import numpy as np
import pandas as pd
import os.path
from scipy import stats, linalg
from itertools import combinations


#Function to calculate graph and network metrics of dynamic and static functional connectivity


def coupling(data,window):
    """
        creates a functional coupling metric from 'data'
        data: should be organized in 'time x nodes' matrix
        smooth: smoothing parameter for dynamic coupling score
        # from PD
        #By default, the result is set to the right edge of the window. 
        This can be changed to the center of the window by setting center=True.
    """
    
    #define variables
    [tr,nodes] = data.shape
    der = tr-1
    td = np.zeros((der,nodes))
    td_std = np.zeros((der,nodes))
    data_std = np.zeros(nodes)
    mtd = np.zeros((der,nodes,nodes))
    sma = np.zeros((der,nodes*nodes))
    
    #calculate temporal derivative
    for i in range(0,nodes):
        for t in range(0,der):
            td[t,i] = data[t+1,i] - data[t,i]
    
    
    #standardize data
    for i in range(0,nodes):
        data_std[i] = np.std(td[:,i])
    
    td_std = td / data_std
   
   
    #functional coupling score
    for t in range(0,der):
        for i in range(0,nodes):
            for j in range(0,nodes):
                mtd[t,i,j] = td_std[t,i] * td_std[t,j]


    #temporal smoothing
    temp = np.reshape(mtd,[der,nodes*nodes])
    sma = pd.rolling_mean(temp,window, center=True)
    sma = np.reshape(sma,[der,nodes,nodes])
    
    return (mtd, sma)


def cal_MTD(subject, sequence, roi, w, partial = False):
	''' wraper function to return MTD estimates
	if set partial = True, '''
	ts_path = '/home/despoB/kaihwang/Rest/ThaGate/NotBackedUp/'
	#pcorr_path = '/home/despoB/kaihwang/Rest/ThaGate/Matrices/'
	fn = ts_path + subject + '_%s_%s_000.netts' %(roi, sequence)
	ts = np.loadtxt(fn)
	
	if partial:
		tts = ts[400:].T.copy() 
		cts = ts[0:400].T.copy()
		for j in range(400):
			b = linalg.lstsq(tts, cts[:,j])[0]
			ts[j,:] = cts[:,j]-tts.dot(b)

	ts = ts.T
	_, MTD = coupling(ts, w) #the smoothing window is set to 14 for HCP, for NKI 645 15

	return MTD #the dimension of MTD is time by ROI by ROI
		

def cal_sFC_graph(subject, sequence, roi, impose = False, threshold = 1.0):
	''' load TS and run static FC'''
	ts_path = '/home/despoB/kaihwang/Rest/ThaGate/NotBackedUp/'
	fn = ts_path + str(subject) + '_%s_%s_000.netts' %(roi, sequence)
	ts = np.loadtxt(fn)

	matrix = np.corrcoef(ts)
	matrix[np.isnan(matrix)] = 0

	matrix = bct.threshold_proportional(matrix, threshold)

	num_iter = 200
	consensus = np.zeros((num_iter, matrix.shape[0], matrix.shape[1]))

	for i in np.arange(0,num_iter):
		ci, _ =bct.modularity_louvain_und_sign(matrix, qtype='sta')
		consensus[i, :,:] = community_matrix(ci)

	mean_matrix = np.nanmean(consensus, axis=0)		
	mean_matrix[np.isnan(mean_matrix)]=0
	CI, Q = bct.modularity_louvain_und_sign(mean_matrix, qtype='sta')

	#no negative weights
	matrix[matrix<0] = 0

	PC = bct.participation_coef(matrix, CI)
		
	#WMD
	WMD = bct.module_degree_zscore(matrix, CI)
		
	## within weight
	WW = cal_within_weight(matrix,CI)
		
	## between Weight
	BW = cal_between_weight(matrix,CI)
		
	return CI, Q, PC, WMD, WW, BW



def community_matrix(membership,min_community_size=5):
	membership = np.array(membership).reshape(-1)
	final_matrix = np.zeros((len(membership),len(membership)))
	final_matrix[:] = np.nan
	connected_nodes = []
	for i in np.unique(membership):
		if len(membership[membership==i]) >= min_community_size:
			for n in np.array(np.where(membership==i))[0]:
				connected_nodes.append(int(n))
	community_edges = []
	between_community_edges = []
	connected_nodes = np.array(connected_nodes)
	for edge in combinations(connected_nodes,2):
		if membership[edge[0]] == membership[edge[1]]:
			community_edges.append(edge)
		else:
			between_community_edges.append(edge)
	for edge in community_edges:
		final_matrix[edge[0],edge[1]] = 1
		final_matrix[edge[1],edge[0]] = 1
	for edge in between_community_edges:
		final_matrix[edge[0],edge[1]] = 0
		final_matrix[edge[1],edge[0]] = 0
	return final_matrix


def cal_dynamic_graph(MTD, impose=False, threshold = False):
	'''calculate graph metrics across time(dynamic)'''
	#setup outputs
	time_points = MTD.shape[0]
	ci = np.zeros([MTD.shape[1],MTD.shape[0]])
	q = np.zeros([MTD.shape[0]])
	WMD = np.zeros([MTD.shape[1],MTD.shape[0]])
	PC = np.zeros([MTD.shape[1],MTD.shape[0]])
	WW = np.zeros([MTD.shape[1],MTD.shape[0]])
	BW = np.zeros([MTD.shape[1],MTD.shape[0]])

	#modularity
	if impose:
		ci = np.tile(np.loadtxt('/home/despoB/kaihwang/Rest/ThaGate/ROIs/Morel_Striatum_Gordon_CI'),[time_points,1]).T
	
	for i, t in enumerate(range(0,time_points)):
		matrix = MTD[i,:,:]

		#need to deal with NANs because of coverage (no signal in some ROIs)
		matrix[np.isnan(matrix)] = 0
		
		#threshold here
		if threshold:
			matrix = bct.threshold_proportional(matrix, threshold)
 	
		#modularity
		if impose == False:
			ci[:,i], q[i] = bct.modularity_louvain_und_sign(matrix)
		
		#PC
		# for now, no negative weights
		matrix[matrix<0] = 0
		PC[:,i] = bct.participation_coef(matrix, ci[:,i])
		
		#WMD
		WMD[:,i] = bct.module_degree_zscore(matrix, ci[:,i])
		
		## within weight
		WW[:,i] = cal_within_weight(matrix,ci[:,i])
		
		## between Weight
		BW[:,i] = cal_between_weight(matrix,ci[:,i])
		
		# cal q using impsose CI partition
		if impose:
			q[i] = cal_modularity_w_imposed_community(matrix, ci[:,i])

	return ci, q, PC, WMD, WW, BW	


def cal_within_weight(matrix, ci):
	'''within module connectivity weight'''
	WW = np.zeros((len(matrix),))

	for i in range(1, int(np.max(ci) + 1)):
		Koi = np.sum(matrix[np.ix_(ci == i, ci == i)], axis=1)
		WW[np.where(ci == i)] = Koi
	return WW


def cal_between_weight(matrix, ci):
	'''between module connectivity weight'''
	BW = np.zeros((len(matrix),))

	for i in range(1, int(np.max(ci) + 1)):
		Koi = np.sum(matrix[np.ix_(ci == i, ci != i)], axis=1)
		BW[np.where(ci == i)] = Koi
	return BW


def run_d_graph(subject, sequence, roi, w, thresh, imp = True, part = False):
	''' to loop through dynamic graph script
	if need to threshold matrices, set thresh
	imp is to decide whether to impose group template
	part is to decide whether to partial variance from tha
	'''
	#subject, sequence, roi, w = raw_input().split()
	
	#fn = '/home/despoB/kaihwang/Rest/ThaGate/Graph/%s_%s_%s_w%s_q.npy' %(subject,sequence,roi,w)
	#if not os.path.exists(fn):
		
	MTD = cal_MTD(subject , sequence, roi, int(w), partial = part)
	ci, q, PC, WMD, WW, BW = cal_dynamic_graph(MTD, impose = imp, threshold = thresh)

	# fn = '/home/despoB/kaihwang/Rest/ThaGate/Graph/%s_%s_%s_w%s_impose_ci' %(subject,sequence,roi,w)
	# np.save(fn, ci)
	fn = '/home/despoB/kaihwang/Rest/ThaGate/Graph/%s_%s_%s_w%s_impose%s_t%s_partial%s_q' %(subject,sequence,roi,w,imp,thresh,part)
	np.save(fn, q)
	fn = '/home/despoB/kaihwang/Rest/ThaGate/Graph/%s_%s_%s_w%s_impose%s_t%s_partial%s_PC' %(subject,sequence,roi,w,imp,thresh,part)
	np.save(fn, PC)
	fn = '/home/despoB/kaihwang/Rest/ThaGate/Graph/%s_%s_%s_w%s_impose%s_t%s_partial%s_WMD' %(subject,sequence,roi,w,imp,thresh,part)
	np.save(fn, WMD)
	fn = '/home/despoB/kaihwang/Rest/ThaGate/Graph/%s_%s_%s_w%s_impose%s_t%s_partial%s_WW' %(subject,sequence,roi,w,imp,thresh,part)
	np.save(fn, WW)
	fn = '/home/despoB/kaihwang/Rest/ThaGate/Graph/%s_%s_%s_w%s_impose%s_t%s_partial%s_BW' %(subject,sequence,roi,w,imp,thresh,part)
	np.save(fn, BW)		


def run_s_graph(subject, sequence, roi, thresh, imp = False, part = False):
	''' to loop through static graph script'''

	ci, q, PC, WMD, WW, BW = cal_sFC_graph(subject, sequence, roi, impose = False, threshold = thresh)

	# fn = '/home/despoB/kaihwang/Rest/ThaGate/Graph/%s_%s_%s_w%s_impose_ci' %(subject,sequence,roi,w)
	# np.save(fn, ci)
	w=0
	#for sFC set w to 0

	fn = '/home/despoB/kaihwang/Rest/ThaGate/Graph/%s_%s_%s_w%s_impose%s_t%s_partial%s_q' %(subject,sequence,roi,w,imp,thresh,part)
	np.save(fn, q)
	fn = '/home/despoB/kaihwang/Rest/ThaGate/Graph/%s_%s_%s_w%s_impose%s_t%s_partial%s_PC' %(subject,sequence,roi,w,imp,thresh,part)
	np.save(fn, PC)
	fn = '/home/despoB/kaihwang/Rest/ThaGate/Graph/%s_%s_%s_w%s_impose%s_t%s_partial%s_WMD' %(subject,sequence,roi,w,imp,thresh,part)
	np.save(fn, WMD)
	fn = '/home/despoB/kaihwang/Rest/ThaGate/Graph/%s_%s_%s_w%s_impose%s_t%s_partial%s_WW' %(subject,sequence,roi,w,imp,thresh,part)
	np.save(fn, WW)
	fn = '/home/despoB/kaihwang/Rest/ThaGate/Graph/%s_%s_%s_w%s_impose%s_t%s_partial%s_BW' %(subject,sequence,roi,w,imp,thresh,part)
	np.save(fn, BW)		



if __name__ == "__main__":

	subject, sequence, roi, w = raw_input().split()
	
	#run_d_graph(subject, sequence, roi, w, thresh = 1, imp = False, part = False)
	Ds=np.array([0.05, 0.075, 0.1, 0.125, 0.15, 1]) #0.05, 0.075, 0.125, 0.15 #1, 0.1
	for d in Ds:
		#run_d_graph(subject, sequence, roi, w, thresh = d, imp = True, part = False)
		run_s_graph(subject, sequence, roi, thresh = d, imp = False, part = False)


