from FuncParcel import *
import numpy as np
import nibabel as nib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from dFC_graph import coupling
from matplotlib import colors
import pickle as pickle
#%matplotlib qt #don't forget this when plotting...

def map_target():
	'''use NKI dataset to find cortical targets(Yeo17 or Yeo400) for each thalamic nuclei'''
	
	# # average matrices then find max YeoNetwork
	# Morel_Yeo17_M = average_corrmat('/home/despoB/connectome-thalamus/ThaGate/Matrices/NKI*_Morel_plus_Yeo17_645_corrmat', np_txt=True, pickle_object=False)
	# np.fill_diagonal(Morel_Yeo17_M, 0)
	# MaxYeo17_Morel = np.argmax(Morel_Yeo17_M[17:,0:17],1)+1

	# # average matrices then find max YeoNetwork using partial correlation
	# Morel_Yeo17_pM = average_corrmat('/home/despoB/connectome-thalamus/ThaGate/Matrices/NKI*_Morel_plus_Yeo17_645_partial_corrmat', np_txt=True, pickle_object=False)
	# np.fill_diagonal(Morel_Yeo17_pM, 0)
	# MaxYeo17_Morel_pM = np.argmax(Morel_Yeo17_pM[17:,0:17],1)+1


	#Morel_Yeo400_M = average_corrmat('/home/despoB/connectome-thalamus/ThaGate/Matrices/NKI*_Morel_plus_Yeo400_1400_corrmat', np_txt=True, pickle_object=False)
	Morel_Yeo400_M = average_corrmat('/home/despoB/connectome-thalamus/ThaGate/Matrices/*_Morel_plus_Yeo400_1400_pcorr_mat', np_txt=True, pickle_object=False)
	np.fill_diagonal(Morel_Yeo400_M, 0)
	Morel_Yeo400_M[np.isnan(Morel_Yeo400_M)] = 0
	M = Morel_Yeo400_M[400:,0:400]
	MaxYeo400_Morel = np.argsort(M,1)[:,-1:-6:-1] #np.argmax(Morel_Yeo400_M[400:,0:400],1)+1 #max connections at the end
	MinYeo400_Morel = np.argsort(M,1)[:,0:5]
	return MaxYeo400_Morel, MinYeo400_Morel, Morel_Yeo400_M


def map_indiv_target(subj, seq, dset):
	''' feed in individual matrix and find cortical targets for each thalamic nuclei'''
	
	if dset == 'NKI':
		#fn = '/home/despoB/connectome-thalamus/ThaGate/Matrices/%s_%s_Morel_plus_Yeo400_%s_corrmat' %(dset, subj, seq)
		fn = '/home/despoB/connectome-thalamus/ThaGate/Matrices/%s_Morel_plus_Yeo400_%s_pcorr_mat' %(subj, seq)
		M = np.loadtxt(fn)
		
	
	if dset == 'HCP':
		#fn = '/home/despoB/connectome-thalamus/ThaGate/Matrices/%s_%s_Morel_plus_Yeo400_%s_corrmat' %(dset, subj, 'rfMRI_REST2_RL')
		fn = '/home/despoB/connectome-thalamus/ThaGate/Matrices/%s_Morel_plus_Yeo400_%s_pcorr_mat' %(subj, 'rfMRI_REST2_RL')	
		M1 = np.loadtxt(fn)
		#fn = '/home/despoB/connectome-thalamus/ThaGate/Matrices/%s_%s_Morel_plus_Yeo400_%s_corrmat' %(dset, subj, 'rfMRI_REST2_LR')
		fn = '/home/despoB/connectome-thalamus/ThaGate/Matrices/%s_Morel_plus_Yeo400_%s_pcorr_mat' %(subj, 'rfMRI_REST2_LR')
		M2 = np.loadtxt(fn)
		M = (M1 + M2)/2

	np.fill_diagonal(M, 0)
	M[np.isnan(M)] = 0
	Cortical_M = M[400:,0:400]
	Max = np.argsort(Cortical_M,1)[:,-1:-6:-1] 
	Min = np.argsort(Cortical_M,1)[:,0:5]	
	return Max, Min


def load_graph_metric(subj, seq, window, measure, impose = True, thresh = 0.1, partial = False):
	'''shorthand to load graph metric'''
	
	fn = '/home/despoB/connectome-thalamus/ThaGate/Graph/%s_%s_Morel_plus_Yeo400_w%s_impose%s_t%s_partial%s_%s.npy' %(subj, seq, window, impose, thresh, partial, measure)
	y = np.load(fn)

	#if measure == 'phis':  #maxwell's calculation on club coeff transposed matrices
	#	y = y.T 		   #the dimension should be ROI by time

	return y


def tha_morel_ts(subj, seq, window):
	'''shorthand to load tha ts from morel atlas (15 nuclei)'''
	fn = '/home/despoB/kaihwang/Rest/ThaGate/NotBackedUp/' + subj +'_Morel_plus_Yeo400' + '_' + str(seq) +'_000.netts'
	y = np.loadtxt(fn)
	ts = y[400:,1:].T #last 15 rows are morel nuclei, take out first timepoint because MTD is temp derivative. Transpose for statsmodel.
	return ts


def tha_morel_plus_cortical_ts(subj, seq, window):
	'''shorthand to load tha ts from morel atlas (15 nuclei)'''
	fn = '/home/despoB/kaihwang/Rest/ThaGate/NotBackedUp/' + subj +'_Morel_plus_Yeo400' + '_' + str(seq) +'_000.netts'
	y = np.loadtxt(fn)
	ts = y.T #last 15 rows are morel nuclei, take out first timepoint because MTD is temp derivative. Transpose for statsmodel.
	return ts


def fit_linear_model(y,x):
	'''multiple linear regression using OLS in statsmodel'''
	#need to take out nans first
	Y = y[~np.isnan(y)]
	X = x[~np.isnan(y)]
	Y = Y[~np.isnan(X)]
	X = X[~np.isnan(X)]

	#add constant
	X = sm.add_constant(X)   
	est = sm.OLS(Y, X).fit() #OLS fit
	return est 


def cortical_graph(Subjects, seq, measures, HCP = True, impose = True):
	'''wrapper function to compile coritcal ROI's graph metrics into a dataframe'''
	df = pd.DataFrame(columns=('Subject', 'PC', 'WMD', 'WW', 'BW', 'q', 'phis')) 

	if HCP:
		seq1 = seq+'_LR'
		seq2 = seq+'_RL'
	
	#loop
	for i, subj in enumerate(Subjects):
		sdf = pd.DataFrame(columns=('Subject', 'PC', 'WMD', 'WW', 'BW', 'q', 'phis'), index = [0]) 
		sdf.loc[0]['Subject'] = subj

		for measure in measures:
			if not HCP:
				y = load_graph_metric(subj, seq, window, measure, impose = impose)
			if HCP:
				y = np.hstack((load_graph_metric(subj, seq1, window, measure, impose = True), load_graph_metric(subj, seq2, window, measure, impose = True)))	

			sdf.loc[0][measure]= np.nanmean(y)		
		df = pd.concat([df, sdf])		
	return df	

def run_regmodel(Subjects, seq, window, measures, HCP = False, IndivTarget = True, MTD = False, impose = False, nodeselection = np.nan, thresh = 0.1, saveobj = False):
	''' wraper script to test thalamic acitivty's effect on global network properties
	if calculating global metrics (eg, q, avePC), then set nodeselection = np.nan. 
	if using MTD between neuclei and cortical targets as predictors, set MTD = True'''
	
	df = pd.DataFrame(columns=('Subject', 'Thalamic Nuclei', 'PC', 'WMD', 'WW', 'BW', 'q', 'phis')) 
 	Nuclei = ['AN','VM', 'VL', 'MGN', 'MD', 'PuA', 'LP', 'IL', 'VA', 'Po', 'LGN', 'PuM', 'PuI', 'PuL', 'VP']
	
	#loop through subjects
	for i, subj in enumerate(Subjects):
		
		#create subject dataframe
		sdf = pd.DataFrame(columns=('Subject', 'Thalamic Nuclei', 'PC', 'WMD', 'WW', 'BW', 'q', 'phis')) 
		sdf['Thalamic Nuclei'] = Nuclei
		sdf['Subject'] = subj

		#two runs for HCP
		if HCP:
			seq1 = seq+'_LR'
			seq2 = seq+'_RL'

		for measure in measures:

			if impose == False:
				if not HCP:
					y = load_graph_metric(subj, seq, window, measure, impose = False, thresh = thresh)
				else:
					y = np.hstack((load_graph_metric(subj, seq1, window, measure, impose = False, thresh = thresh), load_graph_metric(subj, seq2, window, measure, impose = False, thresh = thresh)))
			if impose == True:
				if not HCP:
					y = load_graph_metric(subj, seq, window, measure, impose = True, thresh = thresh)
				else:
					y = np.hstack((load_graph_metric(subj, seq1, window, measure, impose = True, thresh = thresh), load_graph_metric(subj, seq2, window, measure, impose = True, thresh = thresh)))

			#set it to nan for cal global metrics, average across nodes
			if np.isnan(nodeselection).all(): 
				if y.ndim > 1: # for q then no averaging 
					#if dimension more than 1, average across nodes (global prooperty)
					y = np.nanmean(y, axis=0)

			#if not doing coupling, then use thalamic ts		
			if MTD == False: 	
				if not HCP:	
					x = tha_morel_ts(subj, seq, window)	
				if HCP:
					x = np.vstack((tha_morel_ts(subj, seq1, window), tha_morel_ts(subj, seq2, window)))

			#if doing coupling, then extract thalamic and cortical ts and do MTD coupling calculation later	
			if MTD == True:
				if not HCP:
					ts = tha_morel_plus_cortical_ts(subj, seq, window)	
				if HCP:
					ts = np.vstack((tha_morel_plus_cortical_ts(subj, seq1, window), tha_morel_plus_cortical_ts(subj, seq2, window)))

			# if using individual matrices to find cortical targets	
			if IndivTarget:
				if not HCP:
					nodeselection, _ = map_indiv_target(subj, seq, dset = 'NKI')	
				if HCP:
					nodeselection, _ = map_indiv_target(subj, seq, dset = 'HCP')


			#fit one var at a time because of potential co-linearity between nuclei
			for j in np.arange(len(Nuclei)):

				#if selecting certain cortical targets, select nodes before averaging
				if np.alltrue(~np.isnan(nodeselection)): #
					if y.ndim > 1:  # for q then no averaging 
						y = np.nanmean(y[nodeselection[j,:],:],axis=0) #use nodeselection vector to select cortical nodes and average nodal metrics
				
				if MTD == True:
					#extract cortical targets and thalamic nuclei ts						
					i = np.append(nodeselection[j,:],np.array(400+j)) #append the nuclei as last colum (400+j)
					# do MTD coupling
					sma = coupling(ts[:,i],window)[1]
					# ave coupling score for each thalamic nuclei and beweten its cortical targets
					x = np.squeeze(np.nanmean(sma[:,len(i)-1:,][:,:,0:len(i)-1], axis=2)) #len(i)-1 is to determine number of cortical targets included (minus 15 thalamic nuclei)
					try:
						est = fit_linear_model(y,x)
					except:
						pass

				if MTD == False:	
					est = fit_linear_model(y,x[:,j])

				sdf[measure].loc[sdf['Thalamic Nuclei'] == Nuclei[j]] = est.tvalues[1]

				if saveobj:
					fn = '/home/despoB/connectome-thalamus/ThaGate/Graph/'+subj+'_'+str(seq)+'_'+str(window)+'_'+measure+'_reg_est.pickle' 
					save_object(est, fn)

		df = pd.concat([df, sdf])
	
	return df

def save_object(obj, filename):
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)


def YeoNetwork_parcellation():
	''' parcelate thalamus using Yeo17 networks'''
	# average matrices
	M = average_corrmat('/home/despoB/kaihwang/Rest/ThaGate/Matrices/*thavox_plus_Yeo17_645*', np_txt=True, pickle_object=False)

	subcortical_voxels = np.loadtxt('/home/despoB/connectome-thalamus/ROIs/thalamus_voxel_indices')
	cortical_ROIs = np.arange(1,15)
	subcorticalcortical_ROIs = np.append(cortical_ROIs, subcortical_voxels)
	_, ParcelCIs, _, = parcel_subcortical_network(M, subcorticalcortical_ROIs, subcortical_voxels, cortical_ROIs, cortical_ROIs)

	#create a mask based on the morel atlas 
	Morel_atlas = np.loadtxt('/home/despoB/connectome-thalamus/Thalamic_parcel/Morel_parcel')
	Morel_mask = np.loadtxt('/home/despoB/connectome-thalamus/Thalamic_parcel/morel_mask')
	mask_value = Morel_mask==0

	CIs = sort_CI(ParcelCIs)
	return CIs


def sort_CI(Thalamo_ParcelCIs):
    CIs = np.zeros(len(Thalamus_voxel_coordinate))
    for i, thalamus_voxel_index in enumerate(Thalamus_voxel_coordinate[:,3]):
        CIs[i] = Thalamo_ParcelCIs[thalamus_voxel_index][0]
    CIs = CIs.astype(int)
    return CIs


def visualize_parcellation(CIs, cmap):
    # show volum image
    MNI_img = nib.load('/home/despoB/connectome-thalamus/ROIs/MNI152_T1_2mm_brain.nii.gz')
    MNI_data = MNI_img.get_data()
    Thalamus_voxel_coordinate = np.loadtxt('/home/despoB/connectome-thalamus/ROIs/thalamus_voxels_ijk_indices', dtype = int)

    # create mask for parcel
    Mask = np.zeros(MNI_data.shape)


    # assign CI to each subcortical voxel
    for i, CI in enumerate(CIs):
        Mask[Thalamus_voxel_coordinate[i,0], Thalamus_voxel_coordinate[i,1], Thalamus_voxel_coordinate[i,2]] = CIs[i].astype(int)
    Mask = np.ma.masked_where(Mask == 0, Mask)

    # flip dimension to show anteiror of the brain at top
    MNI_data = MNI_data.swapaxes(0,1)
    Mask = Mask.swapaxes(0,1)

    # some plot setting (colormap), interplotation..
    #cmap = colors.ListedColormap(['red', 'blue', 'cyan', 'yellow', 'teal', 'purple', 'pink', 'green', 'black'])
    #cmap = colors.ListedColormap(['blue', 'red', 'cyan', 'yellow', 'green'])
    # display slice by slice
    Z_slices = range(np.min(Thalamus_voxel_coordinate[:,2])+2, np.max(Thalamus_voxel_coordinate[:,2])-1,2)
    fig = plt.figure()
    for i, Z_slice in enumerate(Z_slices):
        #if i <4:
        a = plt.subplot(1, len(Z_slices), i+1 )
        #else:
        #    a = plt.subplot(2, len(Z_slices)/2, i+1-4 )
        a.set_yticks([])
        a.set_xticks([])
        plt.imshow(MNI_data[45:65, 30:60, Z_slice], cmap='gray', interpolation='nearest')
        plt.imshow(Mask[45:65, 30:60, Z_slice],cmap=cmap, interpolation='none', vmin = 1, vmax=np.max(CIs))
        plt.ylim(plt.ylim()[::-1])
        
    fig.tight_layout() 
    fig.set_size_inches(6.45, 0.7) 
   # plt.savefig(savepath, bbox_inches='tight')


if __name__ == "__main__":


	#get list of NKI subjects
	with open("/home/despoB/kaihwang/bin/ThaGate/NKI_subjlist") as f:
		NKISubjects = [line.rstrip() for line in f]
	
	with open("/home/despoB/kaihwang/bin/ThaGate/HCP_subjlist") as f:
		HCPSubjects = [line.rstrip() for line in f]
		
	##global variables
	#for now use TR645	
	#window = 16	
	measures = ['PC', 'WMD', 'WW', 'BW', 'q']
	
	#### test thalamocortical correlations with these global topological measures: ['PC', 'WMD', 'WW', 'BW', 'q', 'phis']
	#seq = 'rfMRI_REST1'
	#window =15
	#Global_df = run_regmodel(HCPSubjects, seq, window, measures, HCP = True)
	#Global_df.to_csv('Data/HCP_Global_df.csv')

	#### test nodal variables
	#get cortical ROI targets of each thalamic nuclei from morel atlas
	MaxYeo400_Morel, MinYeo400_Morel, Morel_Yeo400_M = map_target()
	#np.save('Data/MaxYeo400_Morel', MaxYeo400_Morel)
	#np.save('Data/MinYeo400_Morel', MinYeo400_Morel)
	#np.save('Data/Morel_Yeo400_M', Morel_Yeo400_M)
	#MaxYeo400_Morel = np.load('Data/MaxYeo400_Morel.npy')
		

	### Do Rest, NKI
	seq = 645
	window = 16
	NKI_grpTarget_MTD_th_noImpose_df = run_regmodel(NKISubjects, seq, window, measures, HCP = False, IndivTarget = False, MTD = True, impose = False, nodeselection = MaxYeo400_Morel)
	NKI_grpTarget_MTD_th_noImpose_df.to_csv('Data/NKI_grpTarget_MTD_th_noImpose_df.csv')
	#NKI_grpTarget_MTD_th_noImpose_Partial_df = run_regmodel(NKISubjects, seq, window, measures, HCP = False, IndivTarget = False, MTD = True, impose = False, nodeselection = MaxYeo400_Morel)
	#NKI_grpTarget_MTD_th_noImpose_Partial_df.to_csv('Data/NKI_grpTarget_MTD_th_noImpose_Partial_df.csv')

	### Replicate Rest, HCP
	seq = 'rfMRI_REST1'
	window =15

	HCP_Rest_grpTarget_MTD_th_noImpose_df = run_regmodel(HCPSubjects, seq, window, measures, HCP = True, IndivTarget = False, MTD = True, impose = False, nodeselection = MaxYeo400_Morel)
	HCP_Rest_grpTarget_MTD_th_noImpose_df.to_csv('Data/HCP_Rest_grpTarget_MTD_th_Inompose_df.csv')
	#HCP_Rest_grpTarget_MTD_th_noImpose_Partial_df = run_regmodel(HCPSubjects, seq, window, measures, HCP = True, IndivTarget = False, MTD = True, impose = False, nodeselection = MaxYeo400_Morel)
	#HCP_Rest_grpTarget_MTD_th_noImpose_Partial_df.to_csv('Data/HCP_Rest_grpTarget_MTD_th_Inompose_Partial_df.csv')

	### Do HCP tasks
	seq = 'WM'
	window =15
	#HCP_WM_grpTarget_MTD_th_Impose_df = run_regmodel(HCPSubjects, seq, window, measures, HCP = True, IndivTarget = False, MTD = True, impose = True, nodeselection = MaxYeo400_Morel)
	#HCP_WM_grpTarget_MTD_th_Impose_df.to_csv('Data/HCP_WM_grpTarget_MTD_th_Impose_df.csv')


	seq = 'MOTOR'
	window =15
	HCP_MOTOR_grpTarget_MTD_th_Impose_df = run_regmodel(HCPSubjects, seq, window, measures, HCP = True, IndivTarget = False, MTD = True, impose = True, nodeselection = MaxYeo400_Morel)
	HCP_MOTOR_grpTarget_MTD_th_Impose_df.to_csv('Data/HCP_MOTOR_grpTarget_MTD_th_Impose_df.csv')



