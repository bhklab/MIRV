import pandas as pd, numpy as np
from sklearn.preprocessing import StandardScaler
from itertools import combinations
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests
import seaborn as sns, matplotlib.pyplot as plt

class DataProcessing:
    def __init__(self,
                 patient_id = 'Patient',
                 treatment_id = 'Study',
                 pre_tag = 'baseline',
                 post_tag = 'cycle2',
                 vol_corr=0.1,
                 rad_corr=0.7):
        self.patient_id = patient_id
        self.treatment_id = treatment_id
        self.pre_tag = pre_tag
        self.post_tag = post_tag
        self.vol_corr = vol_corr
        self.rad_corr = rad_corr

    def loadRadiomics(self, path_to_radiomics):
        """
        Load the radiomics data and isolate the pre-treatment timepoint.
        Parameters:
        -----------
        path_to_radiomics : str
            The path to the radiomics data CSV file.
        pre_tag : str
            The tag identifying the pre-treatment timepoint in the 'STUDY' column.
        post_tag : str
            The tag identifying the post-treatment timepoint in the 'STUDY' column.
        Returns:
        --------
        radiomics_pre : pd.DataFrame
            The radiomics data for the pre-treatment timepoint.
        volume_df : pd.DataFrame
            A DataFrame containing volume features and changes with the following columns:
        """
       
        # load the radiomics data
        radiomics_all = pd.read_csv(path_to_radiomics)

        # isolate by timepoint
        radiomics_pre = radiomics_all[radiomics_all[self.treatment_id] == self.pre_tag].reset_index(drop=True)
        radiomics_post = radiomics_all[radiomics_all[self.treatment_id] == self.post_tag].reset_index(drop=True)

        # include only Patient and the radiomics features
        radiomics_pre = radiomics_pre.iloc[:,np.where(radiomics_pre.columns.str.contains('original_shape'))[0][0]:]
        radiomics_pre.insert(0,self.patient_id,radiomics_post[self.patient_id])

        # volume information
        volume_pre = radiomics_pre['original_shape_VoxelVolume'].values
        volume_post = radiomics_post['original_shape_VoxelVolume'].values       
        volume_post[volume_post==np.nan] = 0   # in some cases, the post-treatment volume is NaN because the lesion disappeared

        volume_df = pd.DataFrame({self.patient_id    : radiomics_pre[self.patient_id],
                                'VOLUME_PRE'         : volume_pre,
                                'VOLUME_POST'        : volume_post,
                                'VOLUME_CHANGE_ABS'  : volume_post - volume_pre,
                                'VOLUME_CHANGE_PCT'  : ((volume_post - volume_pre) / volume_pre)*100})
        
        return radiomics_pre, volume_df
    
    def calcResponseOutcomes(self, vol_df):
        """
        Calculate the response outcomes based on the volume changes.
        Parameters:
        -----------
        volume_df : pd.DataFrame
            A DataFrame containing volume features and changes with the following columns:
        Returns:
        --------
        patient_outcomes : pd.DataFrame
            Patient-specific volumetric response outcomes.
        """

        # patient-specific outcomes
        baseline_range = vol_df.groupby(self.patient_id)['VOLUME_PRE'].apply(lambda x: x.max() - x.min()).values
        baseline_stddev = vol_df.groupby(self.patient_id)['VOLUME_PRE'].apply(lambda x: x.std()).values
        volume_range = vol_df.groupby(self.patient_id).apply(lambda x: x.VOLUME_CHANGE_PCT.max()-x.VOLUME_CHANGE_PCT.min())
    
        patient_outcomes = pd.DataFrame({   'Brange': baseline_range,
                                            'Bstddev': baseline_stddev,
                                            'Vrange': volume_range.values, })
                                        
        patient_outcomes.index = volume_range.index

        return patient_outcomes
    
    def radiomicsFeatureReduction(self,rad_df):
        """
        Perform feature reduction on the radiomics data.

        Parameters:
        -----------
        rad_df : pd.DataFrame
            DataFrame containing radiomics features and lesion volume.
        vol_corr : float
            Correlation threshold with lesion volume for feature selection. Default is 0.1.
        rad_corr : float
            Correlation threshold between radiomics features for feature selection. Default is 0.7.

        Returns:
        -----------
        scaled_radiomics: pd.DataFrame
            DataFrame with reduced and scaled radiomics features.

        Steps:
        -----------
        1. Assess the correlation between radiomics features and lesion volume as well as the variance of each feature.
        2. Select features with a variance greater than 10 and a correlation less than 0.2 with lesion volume.
        3. Assess the correlation between the selected features.
        4. Remove any features that are highly correlated with each other.
        5. Remove any connection to lesion volume in the radiomics data.
        6. Scale the data.
        """

        # assess the correlation between radiomics features and lesion volume as well as the variance of each feature
        var = rad_df.var(numeric_only=True)
        cor = rad_df.corr(method='spearman',numeric_only=True)['original_shape_VoxelVolume']
        cols_to_keep = rad_df.columns[np.where(np.logical_and(var>=10,cor<=self.vol_corr))]
        radiomics_varred_corred = rad_df[cols_to_keep]

        print('---------- Radiomic feature reduction ----------')
        print('Original number of features: {}'.format(rad_df.shape[1]))
        print('Features with variance > 10 and correlation with lesion volume < {}: {}'.format(self.vol_corr,radiomics_varred_corred.shape[1]))

        # with the reduced radiomics features, assess the correlation between features
        cor = radiomics_varred_corred.corr()
        m = ~(cor.mask(np.eye(len(cor), dtype=bool)).abs() > self.rad_corr).any()
        features_to_keep = m.index[m.values]
        reduced_radiomics = radiomics_varred_corred[features_to_keep]

        print('Number of features with remaining with correlation to each other < {}: {}'.format(self.rad_corr,len(features_to_keep)))
        print('----------')

        # remove any connection to lesion volume in the radiomics data
        if 'original_shape_VoxelVolume' in reduced_radiomics.columns:
            reduced_radiomics.drop('original_shape_VoxelVolume',axis=1,inplace=True)

        # scale the data for machine learning   
        scaled_radiomics = pd.DataFrame(StandardScaler().fit_transform(reduced_radiomics.values))
        
        # add the USUBJID/Patient column and column names
        scaled_radiomics.index = rad_df.Patient
        scaled_radiomics.columns = reduced_radiomics.columns

        return scaled_radiomics
    
    def calcMIRVMetrics(self,rad_df,resp_df):
        """
        Calculates the cosine similarity and Euclidean distance between pairs of lesions for each patient.
        Steps:
        -----------
        1. Define the number of lesions to consider and the embedding method.
        2. Index rows corresponding to patients with lesion count greater than or equal to the number of lesions defined.
        3. Preallocate lists for storing the results.
        4. For each patient, calculate the pairwise cosine similarity and Euclidean distance between lesions.
        5. Calculate the average and maximum cosine similarity and Euclidean distance for each patient.
        6. Calculate the correlation between the average and maximum cosine similarity and Euclidean distance and the outcome variables.
        7. Plot histograms and scatter plots of the outcome variables.
        """

        # define the number of lesions to consider 
        numLesions = 2
        
        # index rows corresponding to patients with lesion count >= numLesions
        pids, counts = np.unique(rad_df.index, return_counts=True)
        df = rad_df.copy().loc[rad_df.index.isin(pids[counts >= numLesions])]
        outcome_df = resp_df.copy()[resp_df.index.isin(pids[counts >= numLesions])]
        
        # preallocate lists for storing results
        avgTumorSim = []
        maxTumorSim = []
        avgEuclDist = []
        maxEuclDist = []

        for p in np.unique(df.index):
            
            df_patient = df.copy().iloc[np.where(df.index==p)[0],:]

            pc = df_patient.values

            combos = list(combinations(range(len(df_patient)),2))
            
            cos_sim = np.zeros((len(combos),))
            eucl_dist = np.zeros((len(combos),))
            
            for i in range(len(combos)):
                
                cos_sim[i] = 1 - cosine_similarity([pc[combos[i][0],:],pc[combos[i][1],:]])[0][1]
                eucl_dist[i] = np.linalg.norm(pc[combos[i][0],:]-pc[combos[i][1],:])

            # append patient-level results to lists
            avgTumorSim.append(np.mean(cos_sim))
            maxTumorSim.append(np.max(cos_sim))
            avgEuclDist.append(np.mean(eucl_dist))
            maxEuclDist.append(np.max(eucl_dist))

        # add patient-level results to outcome DataFrame
        outcome_df['AvgTumorSim'] = avgTumorSim
        outcome_df['MaxTumorSim'] = maxTumorSim
        outcome_df['AvgEuclDist'] = avgEuclDist
        outcome_df['MaxEuclDist'] = maxEuclDist

        return outcome_df
    
    def correlationMatrix(self,df,drop_cols = None,use_fdr=True,savefigFlag=False):
        """
        Output a correlation matrix with significance values.
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame containing the outcome variables.
        drop_cols : list
            List of columns to drop from the outcome df. Default is None.
        use_fdr : bool
            Use the false discovery rate to adjust p-values. Default is True.
        """

        if drop_cols is not None:
            df = df.drop(drop_cols,axis=1)

        rename_dict = { 'Brange': 'Baseline Volume',
                        'Bstddev': 'Baseline Volume (σ)',
                        'Vrange': 'Δ Volume',
                        'AvgTumorSim': 'MIRV(μ) Dissimilarity',
                        'MaxTumorSim': 'MIRV(max) Dissimilarity',
                        'AvgEuclDist': 'MIRV(μ) Distance', 
                        'MaxEuclDist': 'MIRV(max) Distance'}  
        df.columns = [rename_dict[col] if col in rename_dict.keys() else col for col in df.columns]

        # calculate the correlations and associated p-values
        cor = df.corr(method='spearman')
        pval = df.corr(method=lambda x, y: spearmanr(x, y)[1]) - np.eye(*cor.shape)

        # params
        mask = np.triu(np.ones_like(cor, dtype=bool))
        cmap = sns.color_palette("icefire", as_cmap=True)
        plt.rcParams.update({'font.size': 18})

        # plotting -- correlation matrix
        f, ax = plt.subplots(figsize=(11, 11))
        sns.heatmap(cor, mask=mask, cmap=cmap, vmax=1, center=0,
                    square=True, linewidths=.5, cbar_kws={"shrink": .5},
                    annot=True, fmt=".2f", annot_kws={"color": "white"})
        plt.xticks(np.arange(cor.shape[0]-1) + 0.5, cor.columns[:-1], rotation=90)
        plt.yticks(np.arange(cor.shape[0]-1) + 1.5, cor.columns[1:], rotation=0)
        plt.title('Spearman correlation matrix')
        if savefigFlag:
            plt.savefig('../../results/correlation_matrix.png',bbox_inches='tight',dpi=300)
        plt.show()

        # correcting the p-values
        if use_fdr:
            pval_lower = np.tril(pval, -1)
            pval_corrected = multipletests(pval_lower[pval_lower != 0], method='fdr_bh')[1]
            fdr = np.zeros(pval.shape)
            inds = np.tril_indices_from(fdr,-1)
            for i in range(len(pval_corrected)):
                fdr[inds[0][i],inds[1][i]]=pval_corrected[i]
            fdr = pd.DataFrame(fdr,columns=pval.columns,index=pval.index)
            pval = fdr

        # plotting -- significance matrix
        f, ax = plt.subplots(figsize=(11, 11))
        sns.heatmap(fdr, mask=mask, cmap=cmap, vmax=1, center=0,
                    square=True, linewidths=.5, cbar_kws={"shrink": .5},
                    annot=True, fmt=".2f", annot_kws={"color": "white"})
        plt.xticks(np.arange(pval.shape[0]-1) + 0.5, pval.columns[:-1], rotation=90)
        plt.yticks(np.arange(pval.shape[0]-1) + 1.5, pval.columns[1:], rotation=0)
        plt.title('Significance matrix')
        if savefigFlag:
            plt.savefig('../../results/significance_matrix.png',bbox_inches='tight',dpi=300)
        plt.show()
        
        return cor, pval