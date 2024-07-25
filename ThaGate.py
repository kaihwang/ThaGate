import numpy as np
import nibabel as nib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import pickle as pickle
from nilearn.maskers import NiftiLabelsMasker, NiftiMasker
import scipy.linalg as linalg

def MTD(data,window):
    """
        creates a dynamic functional FC metric from 'data'
        data: should be organized in 'time x nodes' matrix
        window: smoothing parameter for dynamic coupling score

        return:
        mtd: time (minus 1) by roi by roi, describing the coupling across time between ROIs
        sma: the same as mtd, but smoothed with a moving window


        References:
        Shine, J. M., Koyejo, O., Bell, P. T., Gorgolewski, K. J., Gilat, M., & Poldrack, R. A. (2015). Estimation of dynamic functional connectivity using Multiplication of Temporal Derivatives. NeuroImage, 122, 399-407.
        Hwang, K., Shine, J. M., Cellier, D., & D’Esposito, M. (2020). The human intraparietal sulcus modulates task-evoked functional connectivity. Cerebral Cortex, 30(3), 875-887.
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


    #temporal smoothing, using numpy convolve 
    # kernel = np.ones(window) / window
    # for i in range(0,nodes):
    #     for j in range(0,nodes):
    #         np.convolve(mtd[:,i,j], kernel, mode='valid')
    
    #temporal smoothing using pandas and center it
    temp = np.reshape(mtd,[der,nodes*nodes])
    sma = pd.DataFrame(temp).rolling(5, center=True).mean().to_numpy()
    sma = np.reshape(sma,[der,nodes,nodes])
    
    return (mtd, sma)


# load the LSS data. Is this the right path? there seems to be multiple deconvovle folders
nii_data = nib.load("/Shared/lss_kahwang_hpc/data/QUANTUM_4S/Deconvolve_afni/sub-10263/ses-20240308/Quantum4S_sub-10263_ses-20240308_cue_LSS.nii.gz")

print(nii_data.shape) #this should be in the shape of x by y by z by number of trials. 
                      #note that we should have one beta for each trial, but if you use SPMG2 or other basis functions, we might get more than one beta for each trial,
                      #in that case we need to only select the amplitude beta

# Extract the trial by trial betas from the PFC ROIs you want to calculate dFC.
# for now, I am just extracting signals from the whole Yeo/Shaeffer ROI template
ROI_mask = nib.load("/Shared/lss_kahwang_hpc/ROIs/Schaefer400+Morel+BG_2.5.nii.gz")
ROI_masker = NiftiLabelsMasker(ROI_mask)
data = ROI_masker.fit_transform(nii_data)
print(data.shape) # time by ROI

# Select ROIs to calculate the dFC matrix
# This is a critical step, we need to select these PFC ROIs in a principled way. 
# Probably based on RSA or decoding results (those that encode task and state).
# for now I am using two random ROIs. We would have to come up with a strategy for this

cortical_ts = data[:, [348, 173]]

# feed into the MTD function, smoothing window of 5
mtd, sma = MTD(cortical_ts,5) #sma is the smoothed version that has NaNs because of the window
                              #The reason we might want to use the smoothed version is  to reduce outliers from high-frequency noise from the derivative calculations

# now we can build a regression model to solve
# mtd = b * evoked_responses
# where the "Y" is cortico-cortical DFC (mtd), and test how much of it can be explained by thalamic evoked responses
# we can even add in model parameters as regressors, for example
# mtd = b1*evoked_responses + b2*switch_probability + b3*evoked_responses*switch_probability.
# below is the example of building the simple regression model, test it against whole-brian (mass univariate)

brain_masker = NiftiMasker()
brain_time_series = brain_masker.fit_transform(nii_data)
print(brain_time_series) #time by voxels

#build the X, if you want to include interaction terms from model parameters do it here
X = np.array([np.ones(len(sma)),sma[:,0,1]]).T
X = np.vstack((np.full(X.shape[1:], np.nan), X)) #add nan to the first element given its derivative, so the length matches the length of the timeseries

#remove nans. We also need to remove "run breaks" later 
nans = np.unique(np.where(np.isnan(X))[0])
if any(nans):
    X = np.delete(X, nans, axis=0)
    brain_time_series = np.delete(brain_time_series, nans, axis=0)

# fit the model, solve the betas
results = linalg.lstsq(X, brain_time_series) #this is solving b in bX = brain_timeseries)

# extract the betas, put it in brain space
beta_nii = brain_masker.inverse_transform(results[0][1,:]) #the first X is intercept, so select the second one
#you can save or plot this brain image, solve this sub by sub to do group stats using standard approaches. See if there are sig clusters in thalamus or anywhere else.