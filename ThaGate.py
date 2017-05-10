from FuncParcel import *
import numpy as np
import nibabel as nib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


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


if __name__ == "__main__":
	#MaxYeo17_Morel, MaxYeo17_Morel_pM, MaxYeo400_Morel = map_target()

