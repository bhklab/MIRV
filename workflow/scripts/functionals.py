import pandas as pd, numpy as np
from sklearn.preprocessing import StandardScaler
from itertools import combinations
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import mahalanobis
from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests
import seaborn as sns, matplotlib.pyplot as plt
import matplotlib
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test, multivariate_logrank_test
from statannotations.Annotator import Annotator
from scipy import stats


class DataProcessing:
    def __init__(self,
                 patient_id = 'USUBJID',
                 labels = {'col':'Study','pre':'baseline', 'post':'cycle2'},
                 resp_thresh = 25,
                 corr_thresh = {'vol':0.1,'rad':0.7}):

        self.patient_id = patient_id
        self.labels = labels
        self.resp_thresh = resp_thresh,
        self.corr_thresh = corr_thresh

    def addStudyDateEncoder(self,rf_df):
        """
        Adds a numerical study date encoder column to the given DataFrame. The unique integers are assigned in ascending order of the study dates.
        Parameters:
        --------
        rf_df (pd.DataFrame): A pandas DataFrame containing at least the columns 'patient_id' and 'study_date'.
        Returns:
        --------
        pd.DataFrame: The modified DataFrame with an additional 'study_date_enc' column.
        """

        # preeallocate a study_date_enc column
        rf_df['study_date_enc'] = np.zeros(rf_df.shape[0])
        rf_df = rf_df[[rf_df.columns[0]] + [rf_df.columns[-1]] + list(rf_df.columns[1:-1])]

        for subj in rf_df[self.patient_id].unique():
            # determine the unique study_date for each patient (should be 2)
            study_dates = rf_df[rf_df[self.patient_id] == subj]['study_date'].unique()
            # order the study_dates
            study_dates.sort()
            # assign the study_date encoder for each id/date combo
            for i, date in enumerate(study_dates):
                rf_df.loc[(rf_df[self.patient_id] == subj) & (rf_df['study_date'] == date), 'study_date_enc'] = i 
        # in the study_date_enc column, set all 0s to self.labels['pre'] and all 1s to self.labels['post']
        rf_df['study_date_enc'] = rf_df['study_date_enc'].replace({0:self.labels['pre'],1:self.labels['post']})
        self.labels['col'] = 'study_date_enc'

        return rf_df   

    def matchLesions(self,rf_df):
        """
        Match lesions between the pre- and post-treatment timepoints.
        Parameters:
        -----------
        rf_df : pd.DataFrame
            DataFrame containing radiomics features and lesion volume.
        Returns:
        --------
        pre_rad : pd.DataFrame
            The radiomics data for the pre-treatment timepoint.
        post_rad : pd.DataFrame
            The radiomics data for the post-treatment timepoint.
        np.array : The difference in lesion presence between the pre- and post-treatment timepoints (-1 = lesion present only in pre, 1 = lesion present only in post). 
        """
        pre_rad = rf_df[rf_df['study_date_enc'] == 'baseline'].reset_index(drop=True)
        post_rad = rf_df[rf_df['study_date_enc'] == 'cycle2'].reset_index(drop=True)
        
        # Create a dictionary to map patient_id to their lesion indices
        pre_rad_dict = pre_rad.groupby(self.patient_id).apply(lambda x: x.set_index('Roi').index, include_groups=False).to_dict()
        post_rad_dict = post_rad.groupby(self.patient_id).apply(lambda x: x.set_index('Roi').index, include_groups=False).to_dict()

        lesions_to_keep_pre = []
        lesions_to_keep_post = []
        pre_only_flag = []
        post_only_flag = []

        # find matches in lesions by patient ID lesion location
        for subj in rf_df[self.patient_id].unique():
            if subj in pre_rad_dict and subj in post_rad_dict:
                common_lesions = pre_rad_dict[subj].intersection(post_rad_dict[subj])
                only_pre = pre_rad_dict[subj].difference(post_rad_dict[subj])
                only_post = post_rad_dict[subj].difference(pre_rad_dict[subj])

                pre_only_flag.append(np.size(only_pre)>0)
                post_only_flag.append(np.size(only_post)>0)
                
                pre_keep = pre_rad[pre_rad[self.patient_id] == subj].index[pre_rad_dict[subj].isin(common_lesions)]
                post_keep = post_rad[post_rad[self.patient_id] == subj].index[post_rad_dict[subj].isin(common_lesions)]
                
                lesions_to_keep_pre.extend(pre_keep)
                lesions_to_keep_post.extend(post_keep)

        pre_rad = pre_rad.loc[lesions_to_keep_pre].reset_index(drop=True)
        post_rad = post_rad.loc[lesions_to_keep_post].reset_index(drop=True)

        return pre_rad, post_rad, np.array(post_only_flag, dtype=np.int8) - np.array(pre_only_flag,dtype=np.int8)         

    def loadRadiomics(self, path_to_radiomics):
        """
        Load the radiomics data and isolate the pre-treatment timepoint.
        Parameters:
        -----------
        path_to_radiomics : str
            The path to the radiomics data CSV file.
        labels : dict
            The tag identifying the pre- and post-treatment timepoint in the 'STUDY' column.

        Returns:
        --------
        radiomics_pre : pd.DataFrame
            The radiomics data for the pre-treatment timepoint.
        volume_df : pd.DataFrame
            A DataFrame containing volume features and changes with the following columns:
        """
       
        # load the radiomics data
        radiomics_all = pd.read_csv(path_to_radiomics)

        # if there is no label column, create one
        if self.labels['col'] not in radiomics_all.columns:
            radiomics_all = self.addStudyDateEncoder(radiomics_all)

        if np.unique(radiomics_all[self.labels['col']]).size < 2:
            # this is when there is only one timepoint (i.e., SARC021 full dataset and RADCURE)
            # include only Patient and the radiomics features
            radiomics_out = radiomics_all.iloc[:,np.where(radiomics_all.columns.str.contains('original_shape'))[0][0]:]
            radiomics_out.insert(0, self.patient_id, radiomics_all[self.patient_id])
            volume_pre = radiomics_out['original_shape_VoxelVolume'].values
            volume_df = pd.DataFrame({  self.patient_id: radiomics_out[self.patient_id],
                                        'VOLUME_PRE': volume_pre
                                        })  
            return [radiomics_out,volume_df]
        
        # isolate by timepoint
        radiomics_pre = radiomics_all[radiomics_all[self.labels['col']] == self.labels['pre']].reset_index(drop=True)
        radiomics_post = radiomics_all[radiomics_all[self.labels['col']] == self.labels['post']].reset_index(drop=True)

        # missing data handling -- OCTANE
        matchFlag = False
        if len(radiomics_pre) != len(radiomics_post):
            matchFlag = True
            radiomics_pre, radiomics_post, response_flag = self.matchLesions(radiomics_all)

        # include only Patient and the radiomics features
        radiomics_pre = radiomics_pre.iloc[:,np.where(radiomics_pre.columns.str.contains('original_shape'))[0][0]:]
        radiomics_pre.insert(0,self.patient_id,radiomics_post[self.patient_id])

        # volume information
        volume_pre = radiomics_pre['original_shape_VoxelVolume'].values
        volume_post = radiomics_post['original_shape_VoxelVolume'].values 
        # exception handling for missing data (e.g., SARC021)      
        volume_post[volume_post==np.nan] = 0   

        volume_df = pd.DataFrame({self.patient_id    : radiomics_pre[self.patient_id],
                                'VOLUME_PRE'         : volume_pre,
                                'VOLUME_POST'        : volume_post,
                                'VOLUME_CHANGE_ABS'  : volume_post - volume_pre,
                                'VOLUME_CHANGE_PCT'  : ((volume_post - volume_pre) / volume_pre)*100})
        # exception handling for missing data (e.g., OCTANE)
        if matchFlag:
            return [radiomics_pre, volume_df, response_flag]
        else:
            return [radiomics_pre, volume_df]
    
    def calcResponseOutcomes(self, vol_df, resp_arr=None):
        """
        Calculate the response outcomes based on the volume changes.
        Parameters:
        -----------
        volume_df : pd.DataFrame
            A DataFrame containing volume features and changes.
        resp_arr : np.array (optional)
            An array indicating the presence of new lesions or the disappearance of old lesions.
        Returns:
        --------
        patient_outcomes : pd.DataFrame
            Patient-specific volumetric response outcomes.
        """

        # patient-specific outcomes
        baseline_range = vol_df.groupby(self.patient_id)['VOLUME_PRE'].apply(lambda x: x.max() - x.min())
        baseline_stddev = vol_df.groupby(self.patient_id)['VOLUME_PRE'].apply(lambda x: x.std()).values
        baseline_total = vol_df.groupby(self.patient_id)['VOLUME_PRE'].apply(lambda x: x.sum()).values

        if 'VOLUME_CHANGE_PCT' in vol_df.columns:
            volume_range = vol_df.groupby(self.patient_id).apply(lambda x: x.VOLUME_CHANGE_PCT.max() - x.VOLUME_CHANGE_PCT.min())

            # Define the custom function
            def check_volume_change(group):
                if (group < self.resp_thresh).all():
                    return 1
                # elif (group <= self.resp_thresh).all():
                #     return -1
                else:
                    return 0

            # Apply the custom function to each group
            volume_change_check = vol_df.groupby(self.patient_id)['VOLUME_CHANGE_PCT'].apply(check_volume_change)

        if resp_arr is not None:
            pinds_update = np.where(resp_arr != 0)[0]
            patient_list = vol_df[self.patient_id].unique()[pinds_update]

            for i in range(len(patient_list)):
                pvols = vol_df['VOLUME_CHANGE_PCT'][vol_df[self.patient_id] == patient_list[i]]
                if resp_arr[np.where(patient_list == patient_list[i])[0][0]] == -1:
                    pvols = pvols.append(pd.Series(-100))
                elif resp_arr[np.where(patient_list == patient_list[i])[0][0]] == 1:
                    pvols = pvols.append(pd.Series(100))

                # Overwrite the volume_range value
                volume_range.iloc[pinds_update[i]] = pvols.max() - pvols.min()
                
                # Overwrite the volume_change_check value
                volume_change_check.iloc[pinds_update[i]] = check_volume_change(pvols)

        patient_outcomes = pd.DataFrame({
            self.patient_id: baseline_range.index,
            'Brange': baseline_range.values,
            'Bstddev': baseline_stddev,
            'Btotal': baseline_total,
        })

        if 'VOLUME_CHANGE_PCT' in vol_df.columns:
            patient_outcomes['Vrange'] = volume_range.values
            patient_outcomes['Mixed Response'] = volume_change_check.values
            

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
        var_thresh = np.percentile(var, 50)
        cols_to_keep = rad_df.columns[np.where(np.logical_and(var>=var_thresh,cor<=self.corr_thresh['vol']))]
        # cols_to_keep = rad_df.columns[np.where(cor<=self.corr_thresh['vol'])]
        radiomics_varred_corred = rad_df[cols_to_keep]

        print('---------- Radiomic feature reduction ----------')
        print('Original number of features: {}'.format(rad_df.shape[1]))
        print('Features with variance > 10 and correlation with lesion volume < {}: {}'.format(self.corr_thresh['vol'],radiomics_varred_corred.shape[1]))

        # with the reduced radiomics features, assess the correlation between features
        cor = radiomics_varred_corred.corr()
        m = ~(cor.mask(np.eye(len(cor), dtype=bool)).abs() > self.corr_thresh['rad']).any()
        features_to_keep = m.index[m.values]
        reduced_radiomics = radiomics_varred_corred[features_to_keep]

        print('Number of features with remaining with correlation to each other < {}: {}'.format(self.corr_thresh['rad'],len(features_to_keep)))
        print('----------')

        # remove any connection to lesion volume in the radiomics data
        if 'original_shape_VoxelVolume' in reduced_radiomics.columns:
            reduced_radiomics.drop('original_shape_VoxelVolume',axis=1,inplace=True)

        # scale the data  
        scaled_radiomics = pd.DataFrame(StandardScaler().fit_transform(reduced_radiomics.values))
        
        # add the USUBJID/Patient column and column names
        scaled_radiomics.index = rad_df[self.patient_id]
        scaled_radiomics.columns = reduced_radiomics.columns

        print('Selected Features: ')
        print(scaled_radiomics.columns)
        print('----------')

        return scaled_radiomics
    
    def loadData(self, path_to_data, selected_cols):
        """
        Loads a spreadsheet and isolates the specified columns.
        Parameters:
        -----------
        path_to_data : str
            The path to the data (CSV file).
        selected_cols : list
            The columns to keep.
        Returns:
        --------
        pd.DataFrame
            The clinical data with the relevant columns.
        """

        return pd.read_csv(path_to_data)[selected_cols]
    
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
        """

        # define the number of lesions to consider 
        numLesions = 2
        
        # index rows corresponding to patients with lesion count >= numLesions
        pids, counts = np.unique(rad_df.index, return_counts=True)

        # display some stats about the number of lesions
        print('----------')
        print('Number of patients (MIRV analysis): {}'.format(len(pids[counts >= numLesions])))
        print('Number of lesions (MIRV analysis): {}'.format(np.sum(counts[counts >= numLesions])))
        print('----------')
        print('IQR of lesion count: {}'.format(np.percentile(counts[counts >= numLesions],[25,50,75])))
        print('----------')

        # plot the distribution of lesion counts as a percentage of patients
        plt.figure(figsize=(6,3))
        matplotlib.rcParams.update({'font.size': 20})
        lesion_counts = counts[counts >= numLesions]
        bins = np.arange(1, np.max(lesion_counts) + 2) - 0.5
        plt.hist(lesion_counts, bins=bins, weights=np.ones(len(lesion_counts)) / len(lesion_counts) * 100, color='blue')
        plt.xticks(np.arange(1, np.max(lesion_counts) + 1))
        plt.xlim([2, np.max(lesion_counts)])
        # plt.grid(True)
        plt.xlabel('Number of Tumors')
        plt.ylabel('Patients (%)')
        sns.despine(offset=10, trim=True)
        # plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0f}%'.format(y)))


        df = rad_df.copy().loc[rad_df.index.isin(pids[counts >= numLesions])]
        df.index = df.index.astype(str)
        
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
                
                # cov_mat = np.cov(np.stack((pc[combos[i][0],:],pc[combos[i][1],:]),axis=0),rowvar=False)
                # cov_mat += np.eye(cov_mat.shape[0]) * 1e-10
                # inv_cov_mat = np.linalg.inv(cov_mat)
                # cos_sim[i] = mahalanobis(pc[combos[i][0],:],pc[combos[i][1],:],inv_cov_mat)
                cos_sim[i] = 1 - cosine_similarity([pc[combos[i][0],:],pc[combos[i][1],:]])[0][1]
                eucl_dist[i] = np.linalg.norm(pc[combos[i][0],:]-pc[combos[i][1],:])

            # append patient-level results to lists
            avgTumorSim.append(np.mean(cos_sim))
            maxTumorSim.append(np.max(cos_sim))
            avgEuclDist.append(np.mean(eucl_dist))
            maxEuclDist.append(np.max(eucl_dist))
        if resp_df is not None:
            resp_df.index = resp_df.index.astype(str)
            outcome_df = resp_df.copy()[resp_df[self.patient_id].isin(pids[counts >= numLesions])]
            # add patient-level results to outcome DataFrame
            outcome_df['AvgTumorSim'] = avgTumorSim
            outcome_df['MaxTumorSim'] = maxTumorSim
            outcome_df['AvgEuclDist'] = avgEuclDist
            outcome_df['MaxEuclDist'] = maxEuclDist
        else:
            outcome_df = pd.DataFrame({self.patient_id: pids[counts >= numLesions],
                                        'AvgTumorSim': avgTumorSim,
                                        'MaxTumorSim': maxTumorSim,
                                        'AvgEuclDist': avgEuclDist,
                                        'MaxEuclDist': maxEuclDist})
        

        return outcome_df
    
    def correlationMatrix(self,df,drop_cols = None,use_fdr=True,savefigFlag=False,invertFlag=False):
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
        Returns:
        --------
        cor : pd.DataFrame
            DataFrame containing the correlation coefficients.
        pval : pd.DataFrame
            DataFrame containing the p-values (FDR is use_fdr=True).
        """

        # optional: invert color scheme for plots
        if invertFlag:
            # invert the color scheme
            plt.rcParams.update({
                'lines.color': 'white',
                'patch.edgecolor': 'black',
                'text.color': 'white',
                'axes.facecolor': 'black',
                'axes.edgecolor': 'black',
                'axes.labelcolor': 'white',
                'xtick.color': 'white',
                'ytick.color': 'white',
                'grid.color': 'black',
                'figure.facecolor': 'black',
                'figure.edgecolor': 'black',
                'savefig.facecolor': 'black',
                'savefig.edgecolor': 'black',
                'font.size': 24
            })
        else:
            plt.rcParams.update({'font.size': 24})

        if drop_cols is not None:
            df = df.drop(drop_cols,axis=1)

        rename_dict = { 'ARM': 'Trial Arm',
                        'Brange': 'Baseline Volume (range)',
                        'Bstddev': 'Baseline Volume (σ)',
                        'Btotal': 'Baseline Volume (total)',
                        'Vrange': 'Δ Volume',
                        'Mixed Response': 'Complete Tumor Response',
                        'Pretreatment_bin': 'ctDNA (pre)',
                        'Pre-cycle3_bin': 'ctDNA (post)',
                        'RECIST': 'RECIST (non-PD)',
                        'AvgTumorSim': 'MIRV(μ) Dissimilarity',
                        'MaxTumorSim': 'MIRV(max) Dissimilarity',
                        'AvgEuclDist': 'MIRV(μ) Distance', 
                        'MaxEuclDist': 'MIRV(max) Distance'}  
        df = df.rename(columns=rename_dict)
        df = df[[col for col in rename_dict.values() if col in df.columns]]
        df = df.dropna()

        print('----------')
        print('Number of patients (correlation analysis): {}'.format(df.shape[0]))

        if df.shape[1] > 10:
            plot_dim = 15
        else:
            plot_dim = 12

        # calculate the correlations and associated p-values
        cor = df.corr(method='spearman')
        pval = df.corr(method=lambda x, y: spearmanr(x, y)[1]) - np.eye(*cor.shape)

        # params
        mask = np.triu(np.ones_like(cor, dtype=bool))
        cmap = sns.color_palette("hsv", as_cmap=True)
        plt.rcParams.update({'font.size': 18})

        # plotting -- correlation matrix
        f, ax = plt.subplots(figsize=(plot_dim, plot_dim))
        res = sns.heatmap(cor, mask=mask, cmap=cmap, vmax=1, center=0,
                    square=True, linewidths=0, cbar_kws={"shrink": .5},
                    annot=True, fmt=".2f", annot_kws={"color": "black"})
        plt.xticks(np.arange(cor.shape[0]-1) + 0.5, cor.columns[:-1], rotation=90)
        plt.yticks(np.arange(cor.shape[0]-1) + 1.5, cor.columns[1:], rotation=0)
        plt.title('Correlation matrix')
        plt.tight_layout()
        # Removing frame 
        for _, spine in res.spines.items(): 
            spine.set_visible(False) 
            
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
        f, ax = plt.subplots(figsize=(plot_dim, plot_dim))
        res = sns.heatmap(fdr, mask=mask, cmap=cmap, vmax=1, center=0,
                    square=True, linewidths=0, cbar_kws={"shrink": .5},
                    annot=True, fmt=".2f", annot_kws={"color": "black"})
        plt.xticks(np.arange(pval.shape[0]-1) + 0.5, pval.columns[:-1], rotation=90)
        plt.yticks(np.arange(pval.shape[0]-1) + 1.5, pval.columns[1:], rotation=0)
        plt.title('Significance matrix')
        plt.tight_layout()
        # Removing frame 
        for _, spine in res.spines.items(): 
            spine.set_visible(False) 
        if savefigFlag:
            plt.savefig('../../results/significance_matrix.png',bbox_inches='tight',dpi=300)
        plt.show()
        
        return cor, pval
    
    def compareSurvival(self, df, mirv='MaxEuclDist', survCols=[('T_OS','E_OS')], yearConvert=1, savefigFlag=False, invertFlag=False):
        """
        Compare Overall Survival and Progression-Free Survival by a column in a DataFrame.
        The function performs the following:
        - If the values in the mirv column are continuous, binarize by the median value.
        - If the values are binary, compare between the two values.
        - If the values are categorical, perform a comparison between the different string values.
        - Plots the Kaplan-Meier survival curves for each group.
        - If there are two groups, performs a log-rank test and prints the p-value.
        - If there are more than two groups, performs a multivariate log-rank test and prints the p-value.

        Parameters:
        ----------
        df : pd.DataFrame
            The DataFrame containing survival data.
        mirv : str
            The column name in df to compare survival by. Default is 'MaxEuclDist'.
        survCols : list of tuples
            List of tuples where each tuple contains the survival time column and the event column. Default is [('T_OS', 'E_OS')].
        yearConvert : int
            Factor to convert time to years. Default is 1.
        savefigFlag : bool
            If True, save the survival plot as a PNG file. Default is False.
        invertFlag : bool
            If True, invert the color scheme for the plots. Default is False.

        Returns:
        ----------
        None
        """

        # optional: invert color scheme for plots
        if invertFlag:
            # invert the color scheme
            plt.rcParams.update({   'lines.color': 'white',
                                    'patch.edgecolor': 'black',
                                    'text.color': 'white',
                                    'axes.facecolor': 'black',
                                    'axes.edgecolor': 'black',
                                    'axes.labelcolor': 'white',
                                    'xtick.color': 'white',
                                    'ytick.color': 'white',
                                    'grid.color': 'black',
                                    'figure.facecolor': 'black',
                                    'figure.edgecolor': 'black',
                                    'savefig.facecolor': 'black',
                                    'savefig.edgecolor': 'black',
                                    'font.size': 24
            })
        plt.rcParams.update({'font.size': 18})
        kmf = KaplanMeierFitter()

        if df[mirv].dtype in [np.float64, np.int64]:
            median_value = df[mirv].median()
            df['group'] = df[mirv] >= median_value
            df['group'] = df['group'].replace({True: 'High', False: 'Low'})
        elif df[mirv].dtype == np.bool_:
            df['group'] = df[mirv]
            df['group'] = df['group'].replace({True: 'True', False: 'False'})
        else:
            df['group'] = df[mirv]

        for survival_col, event_col in survCols:
            plt.figure(figsize=(10, 6))
            for name, grouped_df in df.groupby('group'):
                if yearConvert > 1:
                    grouped_df[survival_col] = grouped_df[survival_col] / yearConvert
                kmf.fit(grouped_df[survival_col], event_observed=grouped_df[event_col], label=str(name))
                kmf.plot_survival_function()
            sns.despine(trim=True, offset=5)
            plt.title(f'Survival function by {mirv}')
            plt.xlabel('Time (years)')
            plt.ylabel('Survival probability')
            if savefigFlag:
                plt.savefig(f'../../results/{survival_col}_{mirv}.png',bbox_inches='tight',dpi=300)
            plt.show()

            if df['group'].nunique() == 2:
                group1 = df[df['group'] == df['group'].unique()[0]]
                group2 = df[df['group'] == df['group'].unique()[1]]
                results = logrank_test(group1[survival_col], group2[survival_col], 
                                    event_observed_A=group1[event_col], event_observed_B=group2[event_col])
                print(f'Log-rank test p-value for {mirv}: {results.p_value}')
            else:
                results = multivariate_logrank_test(df[survival_col], df['group'], df[event_col])
                print(f'Log-rank test p-value for {mirv}: {results.p_value}')

    def compareMIRVByCategory(self, boxplot_df, boxplot_vars, mirv_vars=['MaxEuclDist'],savefigFlag=False,invertFlag=False):

        """
        Generate and display boxplots comparing MIRV metrics by specified categories. This function 
        generates boxplots for the specified MIRV metric grouped by the variables
        in boxplot_vars. It also performs significance testing using the Kruskal-Wallis test and
        annotates the plots with the results. Optionally, the plots can be saved as PNG files and
        the color scheme can be inverted.
        Parameters:
        ----------
        boxplot_df : pd.DataFrame 
            DataFrame containing the data to be plotted.
        boxplot_vars : list 
            List of variables to be used for grouping in the boxplots.
        mirv : list
            List of MIRV metrics to be plotted. Default is ['MaxEuclDist'].
        savefigFlag : bool
            If True, save the generated plots as PNG files. Default is False.
        invertFlag : bool
            If True, invert the color scheme of the plots. Default is False.
        Returns:
        ----------
        None

        """

        # optional: invert color scheme for plots
        if invertFlag:
            plt.rcParams.update({
                    'lines.color': 'white',
                    'patch.edgecolor': 'black',
                    'text.color': 'white',
                    'axes.facecolor': 'black',
                    'axes.edgecolor': 'white',
                    'axes.labelcolor': 'white',
                    'xtick.color': 'white',
                    'ytick.color': 'white',
                    'grid.color': 'white',
                    'figure.facecolor': 'black',
                    'figure.edgecolor': 'black',
                    'savefig.facecolor': 'black',
                    'savefig.edgecolor': 'black',
            })

        plot_dict = {   'AvgTumorSim': 'MIRV(μ) Dissimilarity',
                        'MaxTumorSim': 'MIRV(max) Dissimilarity',
                        'AvgEuclDist': 'MIRV(μ) Distance', 
                        'MaxEuclDist': 'MIRV(max) Distance'} 

        # plot a boxplot for each variable in boxplot_vars
        for var in boxplot_vars:
            for mirv in mirv_vars:

                x_var = var
                df_temp = boxplot_df.copy() 

                # Conditional formatting for plots
                # -------------------------------
                # Specific to SARC021
                if x_var == 'CPCELL':
                    df_temp[x_var] = df_temp[x_var].replace({'Undifferentiated Pleomorphic Sarcoma': 'UPS'})
                    # change the order of the categories to 'Leiomyosarcoma','UPS','Liposarcoma','Other'
                    df_temp[x_var] = pd.Categorical(df_temp[x_var], categories=['Leiomyosarcoma','UPS','Liposarcoma','Other'], ordered=True)
                if x_var == 'RECIST':
                    # remove patients with NE response
                    df_temp = df_temp[df_temp['RECIST'] != 'NE']
                if x_var == 'Response_bin':
                    always_0 = np.logical_and(~df_temp['Pretreatment_bin'].astype(bool), ~df_temp['Pre-cycle3_bin'].astype(bool))
                    always_1 = np.logical_and(df_temp['Pretreatment_bin'].astype(bool), df_temp['Pre-cycle3_bin'].astype(bool))
                    first0_second1 = np.logical_and(~df_temp['Pretreatment_bin'].astype(bool), df_temp['Pre-cycle3_bin'].astype(bool))
                    first1_second0 = np.logical_and(df_temp['Pretreatment_bin'].astype(bool), ~df_temp['Pre-cycle3_bin'].astype(bool))
                    df_temp[x_var] = np.where(always_0, 'Always (-)', np.where(always_1, 'Always (+)', np.where(first0_second1, '(-), then (+)', '(+), then (-)')))
                if x_var == 'Pretreatment_bin' or x_var == 'Pre-cycle3_bin':
                    df_temp[x_var] = df_temp[x_var].replace({0: '(-)', 1: '(+)'})
                if x_var == 'Duration of response (days)':
                    df_temp[x_var] = ~df_temp[x_var].isna()
                if x_var == 'ECOG Performance Status':
                    df_temp[x_var] = ~(df_temp[x_var] == 'Baseline ECOG 0')
                    df_temp[x_var] = df_temp[x_var].replace({True: 'BECOG > 0', False: 'BECOG 0'})
                # Specific to RADCURE
                if x_var == 'PTUMSITE':
                    # replace categories with <100 counts with 'Other' 
                    counts = df_temp['PTUMSITE'].value_counts()
                    categories_to_replace = counts[counts < 130].index
                    df_temp['PTUMSITE'] = df_temp['PTUMSITE'].replace(categories_to_replace, 'Unknown/Other')
                # Specific to OCTANE
                if x_var == 'tumour-fraction-zviran-adj':
                    df_temp[x_var] = df_temp[x_var] > 0
                    df_temp[x_var] = df_temp[x_var].replace({True: '(+)', False: '(-)'})
                # if x_var == 'STAGE':
                #     # include only stage numbers
                #     stage_mapping = {
                #                     'I': 'I',
                #                     'II': 'II',
                #                     'III': 'III',
                #                     'IV': 'IV'
                #                 }
                #     df_temp['STAGE'] = df_temp['STAGE'].apply(lambda x: stage_mapping.get(x, None))
                #     df_temp = df_temp[df_temp['STAGE'].isin(stage_mapping.values())]

                df_temp = df_temp[[x_var, mirv]].dropna()
                print('Number of patients: {}'.format(df_temp.shape[0]))

                # Increase font size
                plt.rcParams.update({'font.size': 18})

                # Boxplot of MIRV distance by histology
                num_categories = len(df_temp[x_var].unique())
                palette = sns.color_palette("colorblind", num_categories)
                fig, ax = plt.subplots(figsize=(num_categories * 2, 6))
                ax = sns.boxplot(x=x_var, y=mirv, data=df_temp, palette=palette, hue=x_var, showfliers=False, legend=False,linecolor='black')
                plt.xticks(rotation=45)
                ax.set_ylabel(plot_dict[mirv])
                ax.set_xlabel(None)

                # significance testing and annotation
                histology_groups = df_temp.groupby(x_var)[mirv]
                group_data = [group for name, group in histology_groups]
                stat, p_value = stats.kruskal(*group_data)
                pairs = [(group1, group2) for i, group1 in enumerate(histology_groups.groups.keys()) for group2 in list(histology_groups.groups.keys())[i + 1:]]
                annotator = Annotator(ax, pairs, data=df_temp, x=x_var, y=mirv)
                annotator.configure(test='Kruskal', text_format='star', loc='outside', verbose=2, pvalue_thresholds=[(0.001, '***'), (0.01, '**'), (0.05, '*'), (0.1, '.'), (1, 'ns')])
                annotator.apply_and_annotate()

                # Adjust layout to prevent squishing
                plt.subplots_adjust(bottom=0.3)

                # Just because I like the look of it -- like R plots
                sns.despine(trim=True, offset=10)
                if savefigFlag:
                    plt.savefig(f'../plots/{x_var}_boxplot.png', bbox_inches='tight', transparent=True, dpi=300)
                plt.show()