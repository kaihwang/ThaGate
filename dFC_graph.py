import sys
from FuncParcel import *
import bct
import os
import pickle
import glob
import numpy as np
import numpy.testing as npt
import scipy
from scipy.stats.stats import pearsonr
from scipy.sparse.csgraph import minimum_spanning_tree
from sklearn.feature_extraction import image
from igraph import Graph, ADJ_UNDIRECTED, VertexClustering
import nibabel as nib
from itertools import combinations
import pandas as pd

#Function to calculate graph and network metrics of dynamic functional connectivity


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

def cal_MTD(subject , sequence, roi, w):
	ts_path = '/home/despoB/kaihwang/Rest/ThaGate/NotBackedUp/'
	pcorr_path = '/home/despoB/kaihwang/Rest/ThaGate/Matrices/'
	fn = ts_path + subject + '_%s_%s_000.netts' %(roi, sequence)
	ts = np.loadtxt(fn)

	_, MTD = coupling(ts, w) #the smoothing window is set to 14 for HCP, for NKI 645 15

	return MTD #the dimension of MTD is time by ROI by ROI


class brain_graph:
	def __init__(self, VC):
		assert (np.unique(VC.membership) == range(len(VC.sizes()))).all()
		node_degree_by_community = np.zeros((VC.graph.vcount(),len(VC.sizes())),dtype=np.float64)
		for node1 in range(VC.graph.vcount()):
			for comm_idx in (np.unique(VC.membership)):
				comm_total_degree = 0.
				for node2 in np.argwhere(np.array(VC.membership)==comm_idx).reshape(-1):
					eid = VC.graph.get_eid(node1,node2,error=False)
					if eid == - 1:
						continue
					weight = VC.graph.es[eid]["weight"]
					comm_total_degree = comm_total_degree + weight
				node_degree_by_community[node1,comm_idx] = comm_total_degree
		pc_array = np.zeros(VC.graph.vcount())
		for node in range(VC.graph.vcount()):
		    assert np.isclose(VC.graph.strength(node,weights='weight'),np.nansum(node_degree_by_community[node]))
		    # node_degree = np.nansum(node_degree_by_community[node])
		    node_degree = VC.graph.strength(node,weights='weight')
		    if node_degree == 0.0: 
		        pc_array[node]= np.nan
		        continue    
		    pc = 0.0
		    for idx,comm_degree in enumerate(node_degree_by_community[node]):
		        pc = pc + ((float(comm_degree)/float(node_degree))**2)
		    pc = 1.0 - pc
		    pc_array[int(node)] = float(pc)
		self.pc = pc_array
		wmd_array = np.zeros(VC.graph.vcount())
		for comm_idx in range(len(VC.sizes())):
			comm = np.argwhere(np.array(VC.membership)==comm_idx).reshape(-1)
			comm_std = np.std(node_degree_by_community[comm,comm_idx],dtype=np.float64)
			comm_mean = np.mean(node_degree_by_community[comm,comm_idx],dtype=np.float64)
			for node in comm:
				# node_degree = np.nansum(node_degree_by_community[node])
				node_degree = VC.graph.strength(node,weights='weight')
				comm_node_degree = node_degree_by_community[node,comm_idx]
				if node_degree == 0.0:
					wmd_array[node] = np.nan
					continue
				if comm_std == 0.0:
					assert comm_node_degree == comm_mean
					wmd_array[node] = 0.0
					continue
				wmd_array[node] = np.divide((np.subtract(comm_node_degree,comm_mean)),comm_std)
		self.wmd = wmd_array
		self.community = VC
		self.node_degree_by_community = node_degree_by_community
		self.matrix = np.array(self.community.graph.get_adjacency(attribute='weight').data)


def matrix_to_igraph(matrix,cost,binary=False,check_tri=True,interpolation='midpoint',normalize=False,mst=False,test_matrix=True):
	"""
	Convert a matrix to an igraph object
	matrix: a numpy matrix
	cost: the proportion of edges. e.g., a cost of 0.1 has 10 percent
	of all possible edges in the graph
	binary: False, convert weighted values to 1
	check_tri: True, ensure that the matrix contains upper and low triangles.
	if it does not, the cost calculation changes.
	interpolation: midpoint, the interpolation method to pass to np.percentile
	normalize: False, make all edges sum to 1. Convienient for comparisons across subjects,
	as this ensures the same sum of weights and number of edges are equal across subjects
	mst: False, calculate the maximum spanning tree, which is the strongest set of edges that
	keep the graph connected. This is convienient for ensuring no nodes become disconnected.
	"""
	matrix = np.array(matrix)
	matrix = threshold(matrix,cost,binary,check_tri,interpolation,normalize,mst)
	g = Graph.Weighted_Adjacency(matrix.tolist(),mode=ADJ_UNDIRECTED,attr="weight")
	print 'Matrix converted to graph with density of: ' + str(g.density())
	if np.diff([cost,g.density()])[0] > .005:
		print 'Density not %s! Did you want: ' %(cost)+ str(g.density()) + ' ?' 
	return g
		

def cal_dynamic_graph(MTD):
	'''calculate graph metrics across time(dynamic)'''
	#setup outputs
	time_points = MTD.shape[0]
	ci = np.zeros([MTD.shape[1],MTD.shape[0]])
	q = np.zeros([MTD.shape[0]])
	WMD = np.zeros([MTD.shape[1],MTD.shape[0]])
	PC = np.zeros([MTD.shape[1],MTD.shape[0]])
	WW = np.zeros([MTD.shape[1],MTD.shape[0]])
	BW = np.zeros([MTD.shape[1],MTD.shape[0]])
	
	for i, t in enumerate(range(0,time_points+1)):
		matrix = MTD[t,:,:]
		#modularity
		ci[:,i], q[i] = bct.modularity_louvain_und_sign(matrix)
		#PC
		PC[:,i] = bct.participation_coef_sign(matrix, ci[:,i])
		#WMD
		WMD[:,i] = bct.module_degree_zscore(matrix, ci[:,i])
		## within weight
		WW[:,i] = cal_within_weight(matrix,ci[:,i])
		## between Weight
		BW[:,i] = cal_between_weight(matrix,ci[:,i])

	return ci, q, PC, WMD, WW, BW	

def matrix_to_igraph(matrix,cost,binary=False,check_tri=True,interpolation='midpoint',normalize=False,mst=False,test_matrix=True):
	"""
	Convert a matrix to an igraph object
	matrix: a numpy matrix
	cost: the proportion of edges. e.g., a cost of 0.1 has 10 percent
	of all possible edges in the graph
	binary: False, convert weighted values to 1
	check_tri: True, ensure that the matrix contains upper and low triangles.
	if it does not, the cost calculation changes.
	interpolation: midpoint, the interpolation method to pass to np.percentile
	normalize: False, make all edges sum to 1. Convienient for comparisons across subjects,
	as this ensures the same sum of weights and number of edges are equal across subjects
	mst: False, calculate the maximum spanning tree, which is the strongest set of edges that
	keep the graph connected. This is convienient for ensuring no nodes become disconnected.
	"""
	matrix = np.array(matrix)
	matrix = threshold(matrix,cost,binary,check_tri,interpolation,normalize,mst)
	g = Graph.Weighted_Adjacency(matrix.tolist(),mode=ADJ_UNDIRECTED,attr="weight")
	print 'Matrix converted to graph with density of: ' + str(g.density())
	if np.diff([cost,g.density()])[0] > .005:
		print 'Density not %s! Did you want: ' %(cost)+ str(g.density()) + ' ?' 
	return g


# def participation_coef(W, ci, degree='undirected'):
#     '''
#     Participation coefficient is a measure of diversity of intermodular
#     connections of individual nodes.
#     Parameters
#     ----------
#     W : NxN np.ndarray
#         binary/weighted directed/undirected connection matrix
#     ci : Nx1 np.ndarray
#         community affiliation vector
#     degree : str
#         Flag to describe nature of graph 'undirected': For undirected graphs
#                                          'in': Uses the in-degree
#                                          'out': Uses the out-degree
#     Returns
#     -------
#     P : Nx1 np.ndarray
#         participation coefficient
#     '''
#     if degree == 'in':
#         W = W.T

#     _, ci = np.unique(ci, return_inverse=True)
#     ci += 1

#     n = len(W)  # number of vertices
#     Ko = np.sum(W, axis=1)  # (out) degree
#     Gc = np.dot((W != 0), np.diag(ci))  # neighbor community affiliation
#     Kc2 = np.zeros((n,))  # community-specific neighbors

#     for i in range(1, int(np.max(ci)) + 1):
#         Kc2 += np.square(np.sum(W * (Gc == i), axis=1))

#     P = np.ones((n,)) - Kc2 / np.square(Ko)
#     # P=0 if for nodes with no (out) neighbors
#     P[np.where(np.logical_not(Ko))] = 0

#     return P

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


subject, sequence, roi = raw_input().split()
function_cal_pcorr_mat(subject, sequence, roi)