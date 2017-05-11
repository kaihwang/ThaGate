from FuncParcel import *
import numpy as np
import nibabel as nib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm

def map_target():
	'''use NKI dataset to find cortical targets(Yeo17 or Yeo400) for each thalamic nuclei'''
	# average matrices then find max
	Morel_Yeo17_M = average_corrmat('/home/despoB/connectome-thalamus/ThaGate/Matrices/NKI*_Morel_plus_Yeo17_645_corrmat', np_txt=True, pickle_object=False)
	np.fill_diagonal(Morel_Yeo17_M, 0)
	MaxYeo17_Morel = np.argmax(Morel_Yeo17_M[17:,0:17],1)+1


	Morel_Yeo17_pM = average_corrmat('/home/despoB/connectome-thalamus/ThaGate/Matrices/NKI*_Morel_plus_Yeo17_645_partial_corrmat', np_txt=True, pickle_object=False)
	np.fill_diagonal(Morel_Yeo17_pM, 0)
	MaxYeo17_Morel_pM = np.argmax(Morel_Yeo17_pM[17:,0:17],1)+1


	Morel_Yeo400_M = average_corrmat('/home/despoB/connectome-thalamus/ThaGate/Matrices/NKI*_Morel_plus_Yeo400_645_corrmat', np_txt=True, pickle_object=False)
	np.fill_diagonal(Morel_Yeo400_M, 0)
	Morel_Yeo400_M[np.isnan(Morel_Yeo400_M)] = 0
	MaxYeo400_Morel = np.argmax(Morel_Yeo400_M[400:,0:400],1)+1

	return MaxYeo17_Morel, MaxYeo17_Morel_pM, MaxYeo400_Morel



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


def fit_linear_model(y,x):
	'''multiple linear regression using OLS in statsmodel'''
	#need to take out nans first
	Y = y[~np.isnan(y)]
	X = x[~np.isnan(y)]

	#add constant
	X = sm.add_constant(X)   
	est = sm.OLS(Y, X).fit() #OLS fit
	return est 

if __name__ == "__main__":

	#get targets
	#MaxYeo17_Morel, MaxYeo17_Morel_pM, MaxYeo400_Morel = map_target()

	#get list of NKI subjects
	with open("/home/despoB/kaihwang/bin/ThaGate/NKI_subjlist") as f:
		Subjects = [line.rstrip() for line in f]

	#global variables
	#for now use TR645	
	seq = 645
	window = 16	
	Nuclei = ['AN','VM', 'VL', 'MGN', 'MD', 'PuA', 'LP', 'IL', 'VA', 'Po', 'LGN', 'PuM', 'PuI', 'PuL', 'VP']
	measures = ['PC', 'WMD', 'WW', 'BW', 'q']
	
	#### test thalamic correlations with these global topological measures: ['PC', 'WMD', 'WW', 'BW', 'q']
	#output dataframe
	df = pd.DataFrame(columns=('Subject', 'Thalamic Nuclei', 'PC', 'WMD', 'WW', 'BW', 'q')) 
	 
	#loop through subjects
	for i, subj in enumerate(Subjects):
		
		#create subject dataframe
		sdf = pd.DataFrame(columns=('Subject', 'Thalamic Nuclei', 'PC', 'WMD', 'WW', 'BW', 'q')) 
		sdf['Thalamic Nuclei'] = Nuclei
		sdf['Subject'] = subj

		for measure in measures:
			y = load_graph_metric(subj, seq, window, measure)
			if y.ndim > 1:
				#if dimension more than 1, average across nodes (global prooperty)
				y = np.mean(load_graph_metric(subj, seq, window, measure), axis=0)
			x = tha_morel_ts(subj, seq, window)	

			#fit one var at a time because of potential co-linearity between nuclei
			for j in np.arange(x.shape[1]):
				est = fit_linear_model(y,x[:,j])
				sdf[measure].loc[sdf['Thalamic Nuclei'] == Nuclei[j]] = est.tvalues[1]

		df=pd.concat([df, sdf])

	#plot results
	for measure in measures:
		sns.factorplot(x='Thalamic Nuclei', y=measure, data=df, kind='bar')	






