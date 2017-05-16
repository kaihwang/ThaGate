from FuncParcel import *
import numpy as np
import nibabel as nib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from dFC_graph import coupling

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


	Morel_Yeo400_M = average_corrmat('/home/despoB/connectome-thalamus/ThaGate/Matrices/NKI*_Morel_plus_Yeo400_645_corrmat', np_txt=True, pickle_object=False)
	np.fill_diagonal(Morel_Yeo400_M, 0)
	Morel_Yeo400_M[np.isnan(Morel_Yeo400_M)] = 0
	M = Morel_Yeo400_M[400:,0:400]
	MaxYeo400_Morel = np.argsort(M,1)[:,-1:-6:-1] #np.argmax(Morel_Yeo400_M[400:,0:400],1)+1 #max connections at the end
	MinYeo400_Morel = np.argsort(M,1)[:,0:5]
	return MaxYeo400_Morel, MinYeo400_Morel, Morel_Yeo400_M



def load_graph_metric(subj, seq, window, measure):
	'''shorthand to load graph metric'''
	fn = '/home/despoB/connectome-thalamus/ThaGate/Graph/' + subj + '_' + str(seq) + '_' +'Morel_plus_Yeo400' + '_w' + str(window) + '_' + measure + '.npy'
	y = np.load(fn)
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

	#add constant
	X = sm.add_constant(X)   
	est = sm.OLS(Y, X).fit() #OLS fit
	return est 


def run_regmodel(Subjects, seq, window, measures, nodeselection = np.nan, MTD = False):
	''' wraper script to test thalamic acitivty's effect on global network properties
	if calculating global metrics (eg, q, avePC), then set nodeselection = np.nan
	if using MTD between neuclei and cortical targets as predictors, set MTD = True'''
	
	df = pd.DataFrame(columns=('Subject', 'Thalamic Nuclei', 'PC', 'WMD', 'WW', 'BW', 'q')) 
 	Nuclei = ['AN','VM', 'VL', 'MGN', 'MD', 'PuA', 'LP', 'IL', 'VA', 'Po', 'LGN', 'PuM', 'PuI', 'PuL', 'VP']
	#loop through subjects
	for i, subj in enumerate(Subjects):
		#create subject dataframe
		sdf = pd.DataFrame(columns=('Subject', 'Thalamic Nuclei', 'PC', 'WMD', 'WW', 'BW', 'q')) 
		sdf['Thalamic Nuclei'] = Nuclei
		sdf['Subject'] = subj

		for measure in measures:
			y = load_graph_metric(subj, seq, window, measure)

			#set it to nan for cal global metrics, average across nodes
			if np.isnan(nodeselection).all(): 
				if y.ndim > 1: # for q then no averaging 
					#if dimension more than 1, average across nodes (global prooperty)
					y = np.mean(y, axis=0)

			#if not doing coupling, then use thalamic ts		
			if MTD == False: 		
				x = tha_morel_ts(subj, seq, window)	

			#if doing coupling, then extract thalamic and cortical ts and do MTD coupling calculation later	
			if MTD == True:
				ts = tha_morel_plus_cortical_ts(subj, seq, window)	
				
			#fit one var at a time because of potential co-linearity between nuclei
			for j in np.arange(len(Nuclei)):

				#if selecting certain cortical targets, select nodes before averaging
				if np.alltrue(~np.isnan(nodeselection)): #
					if y.ndim > 1:  # for q then no averaging 
						y = np.mean(y[nodeselection[j,:],:],axis=0) #use nodeselection vector to select cortical nodes and average nodal metrics
				
				if MTD == True:
					#extract cortical targets and thalamic nuclei ts						
					i = np.append(nodeselection[j,:],np.array(400+j)) #append the nuclei as last colum (400+j)
					# do MTD coupling
					sma = coupling(ts[:,i],window)[1]
					# ave coupling score for each thalamic nuclei and beweten its cortical targets
					x = np.squeeze(np.mean(sma[:,len(i)-1:,][:,:,0:len(i)-1], axis=2)) #len(i)-1 is to determine number of cortical targets included (minus 15 thalamic nuclei)
					est = fit_linear_model(y,x)

				if MTD == False:	
					est = fit_linear_model(y,x[:,j])

				sdf[measure].loc[sdf['Thalamic Nuclei'] == Nuclei[j]] = est.tvalues[1]

		df=pd.concat([df, sdf])
	
	return df

if __name__ == "__main__":


	#get list of NKI subjects
	with open("/home/despoB/kaihwang/bin/ThaGate/NKI_subjlist") as f:
		Subjects = [line.rstrip() for line in f]

	##global variables
	#for now use TR645	
	seq = 645
	window = 16	
	measures = ['PC', 'WMD', 'WW', 'BW', 'q']
	
	#### test thalamic correlations with these global topological measures: ['PC', 'WMD', 'WW', 'BW', 'q']
	Global_df = run_regmodel(Subjects, seq, window, measures)

	#### test nodal variables
	#get targets
	#MaxYeo17_Morel, MaxYeo17_Morel_pM, MaxYeo400_Morel = map_target()
	MaxYeo400_Morel, MinYeo400_Morel, Morel_Yeo400_M = map_target()
	Target_Node_df = run_regmodel(Subjects, seq, window, measures, MaxYeo400_Morel)
	Target_MTD_Node_df = run_regmodel(Subjects, seq, window, measures, MaxYeo400_Morel, MTD = True)
	NonTarget_Node_df = run_regmodel(Subjects, seq, window, measures, MinYeo400_Morel)
	NonTarget_MTD_Node_df = run_regmodel(Subjects, seq, window, measures, MinYeo400_Morel, MTD = True)


	### #plot results
	for measure in measures:
		#plt.figure()
		sns.factorplot(x='Thalamic Nuclei', y=measure, data=NonTarget_MTD_Node_df, kind='bar')	

	#get a sense of the overal distribution of thalamocortical weights	
	#plt.figure()
	sns.distplot(Morel_Yeo400_M[~np.isnan(Morel_Yeo400_M)])

	#save outputs
	Global_df.to_csv('Data/Global_df.csv')
	Target_Node_df.to_csv('Data/Target_Node_df.csv')
	Target_MTD_Node_df.to_csv('Data/Target_MTD_Node_df.csv')
	NonTarget_Node_df.to_csv('Data/NonTarget_Node_df.csv')
	NonTarget_MTD_Node_df.to_csv('Data/NonTarget_MTD_Node_df.csv')

	np.save('Data/MaxYeo400_Morel', MaxYeo400_Morel)
	np.save('Data/MinYeo400_Morel', MinYeo400_Morel)
	np.save('Data/Morel_Yeo400_M', Morel_Yeo400_M)


