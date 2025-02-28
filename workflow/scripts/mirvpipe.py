# Initialize
import os
os.chdir(os.path.dirname(__file__))
from functionals import DataProcessing

class MIRVPipe:
    def __init__(   self, 
                    patient_id = 'USUBJID',
                    radiomics = '../../procdata/SARC021/radiomics-lung.csv', 
                    clinical = ['../../rawdata/clinical-all.csv',{'corrvars':['ARM'],'boxplotvars':['ARM']}],
                    recist = ['../../rawdata/SARC021/recist-all.csv',{'corrvars':[],'boxplotvars':[],}],
                    survival = ['../../rawdata/SARC021/survival-all.csv', {'survcols':[],'yearConversion':1}],
                    ctdna = ['../../rawdata/SARC021/ctdna-lms.csv', {'corrvars': ['Pretreatment_bin', 'Pre-cycle3_bin'], 'boxplotvars': ['Pretreatment_bin', 'Pre-cycle3_bin']}]
                 ):
        self.patient_id = patient_id
        self.radiomicsData = radiomics
        self.clinicalData = clinical
        self.recistData = recist
        self.survivalData = survival
        self.ctdnaData = ctdna

    def run(self):
        
        # instantiate the data processing class
        dp = DataProcessing(patient_id = self.patient_id,    # column name for patient ID in PyRadiomics output
                            labels = {'col':'STUDY','pre':'baseline','post':'cycle2'},
                            resp_thresh=33)       # options for 'col': (1) 'diagnostics_Versions_PyRadiomics', (2) 'STUDY'
        print('Data Processing Class Initialized')
        # radiomics data loading and feature reduction
        rad_volume_results = dp.loadRadiomics(self.radiomicsData)

        # feature reduction -- FIX
        radiomics_red = dp.radiomicsFeatureReduction(rad_volume_results[0])
        print('Selected Features: ')
        print(radiomics_red.columns)
        print('----------')
        
        # Calculate MIRV and determine the response outcomes (if available)
        if len(rad_volume_results) == 3:
            response_df = dp.calcResponseOutcomes(rad_volume_results[1],rad_volume_results[2])
            outcome_df = dp.calcMIRVMetrics(radiomics_red, response_df)
        else:
            response_df = dp.calcResponseOutcomes(rad_volume_results[1])
            outcome_df = dp.calcMIRVMetrics(radiomics_red, response_df)

        # ----- SURVIVAL ANALYSIS -----
        if self.survivalData[0] is not None and self.survivalData[1]['survcols'] is not None:
            survcols_flat = [col for sublist in self.survivalData[1]['survcols'] for col in sublist]
            survival_df = dp.loadData(self.survivalData[0], survcols_flat + [self.patient_id])
            survival_df[self.patient_id] = survival_df[self.patient_id].astype(str)
            outcome_df[self.patient_id] = outcome_df[self.patient_id].astype(str)
            survival_df = survival_df[survival_df[self.patient_id].isin(outcome_df[self.patient_id])]
            survival_df = survival_df.merge(outcome_df, on=self.patient_id, how='left')
            survival_df = survival_df.dropna()
            print('Number of patients (survival analysis): {}'.format(survival_df.shape[0]))
            print('----------')
            print('Survival IQR: {}'.format(survival_df['T_OS'].quantile([0.25,0.50,0.75])))
            print('Number of events: {}'.format(survival_df['E_OS'].sum()))
            mirv_cols = ['AvgTumorSim', 'MaxTumorSim', 'AvgEuclDist', 'MaxEuclDist']
            for col in mirv_cols:
                dp.compareSurvival(survival_df, col, self.survivalData[1]['survcols'],savefigFlag=True)

        # ----- CORRELATION ANALYSIS -----
        data_types = [self.clinicalData,self.recistData,self.ctdnaData]
        corr_df = outcome_df.copy()
        for i, data in enumerate(data_types):
            if data[0] is not None and data[1]['corrvars'] is not None:
                data_df = dp.loadData(data[0], data[1]['corrvars'] + [self.patient_id])
                data_df = data_df[data_df[self.patient_id].isin(outcome_df[self.patient_id])]
                if data == self.recistData and 'RECIST' in data_df.columns:
                    data_df['RECIST'] = data_df['RECIST'] != 'PD'
                # if data == self.ctdnaData and 'Pre-cycle3_bin' in data_df.columns:
                #     data_df['Response_bin'] = data_df['Pre-cycle3_bin'] - data_df['Pretreatment_bin']  # fix
                # if data == self.ctdnaData and 'Pre-cycle3_frac' in data_df.columns:
                #     data_df['Response_frac'] = data_df['Pre-cycle3_frac'] - data_df['Pretreatment_frac']  # fix
                corr_df[self.patient_id] = corr_df[self.patient_id].astype(str)
                data_df[self.patient_id] = data_df[self.patient_id].astype(str)
                corr_df = corr_df.merge(data_df, on=self.patient_id, how='left', suffixes=('', '_drop')).reset_index(drop=True)
                # outcome_df = outcome_df.drop(columns=[f'{self.patient_id}_drop'])
        corr_df = corr_df.drop(columns=[self.patient_id])

        if 'HPV' in corr_df.columns:
            corr_df['HPV'] = corr_df['HPV'].replace({'Yes, positive':1,'Yes, Negative':0})
            corr_df.dropna(inplace=True)

        # output correlation matrix with significance values
        # cols_to_drop = ['Response_bin', 'Pretreatment_bin', 'Pre-cycle3_bin']
        cormat, pmat = dp.correlationMatrix(corr_df,drop_cols=['AvgTumorSim','AvgEuclDist'],savefigFlag=False,invertFlag=False) 

        # ----- BOXPLOTS -----
        data_types = [self.clinicalData,self.recistData,self.ctdnaData]
        box_df = outcome_df.copy()
        for i, data in enumerate(data_types):
            if data[0] is not None and data[1]['boxplotvars'] is not None:
                data_df = dp.loadData(data[0], data[1]['boxplotvars'] + [self.patient_id])
                data_df = data_df[data_df[self.patient_id].isin(outcome_df[self.patient_id])]
                if data == self.ctdnaData and 'Pre-cycle3_bin' in data_df.columns:
                    data_df['Response_bin'] = data_df['Pre-cycle3_bin'] - data_df['Pretreatment_bin']  # fix
                box_df[self.patient_id] = box_df[self.patient_id].astype(str)
                data_df[self.patient_id] = data_df[self.patient_id].astype(str)
                box_df = box_df.merge(data_df, on=self.patient_id, how='left', suffixes=('', '_drop')).reset_index(drop=True)
                # outcome_df = outcome_df.drop(columns=[f'{self.patient_id}_drop'])
        # box_df = box_df.drop(columns=[self.patient_id])
        box_df = box_df[box_df['CPCELL'] != 'NE']
        boxplot_vars = box_df.columns[box_df.columns.get_loc('MaxEuclDist')+1:]
        dp.compareMIRVByCategory(box_df, boxplot_vars, mirv_vars=['MaxEuclDist'],savefigFlag=False,invertFlag=False)
        
        results = [rad_volume_results,survival_df,corr_df,box_df,cormat,pmat]
        
        return results

if __name__ == '__main__':
    
    # instantiate the MIRV pipeline for SARC021
    mp_sarc = MIRVPipe(     radiomics = '../../procdata/SARC021/radiomics-lung.csv',
                            clinical  = [   '../../rawdata/SARC021/baseline-all.csv', 
                                        {'corrvars':[],
                                        'boxplotvars':['CPCELL']}], 
                            recist    = [   '../../rawdata/SARC021/recist-all.csv',
                                        {'corrvars':[],
                                        'boxplotvars':['RECIST']}], 
                            survival  = [   '../../rawdata/SARC021/survival-all.csv',
                                        {'survcols':[#('T_PFS','E_PFS'), 
                                                     ('T_OS', 'E_OS')],
                                                     'yearConversion':1}], 
                            ctdna     = [ None, #'../../rawdata/SARC021/ctdna-lms.csv',
                                        {'corrvars':['Pretreatment_bin','Pre-cycle3_bin'],
                                        'boxplotvars':['Pretreatment_bin','Pre-cycle3_bin']}]
                        )

    # instantiate the MIRV pipeline for CRLM
    mp_crlm = MIRVPipe(     radiomics = '../../procdata/MSK/MSK_radiomics.csv',
                            clinical  = [ None,#  '../../rawdata/SARC021/baseline-all.csv', 
                                        {'corrvars':[],
                                        'boxplotvars':[]}], 
                            recist    = [ None,#  '../../rawdata/SARC021/recist-all.csv',
                                        {'corrvars':[],
                                        'boxplotvars':[]}], 
                            survival  = [ None, #  '../../rawdata/SARC021/survival-all.csv',
                                        {'survcols':[],
                                                     'yearConversion':1}], 
                            ctdna     = [ None, #  '../../rawdata/SARC021/ctdna-lms.csv',
                                        {'corrvars':[],
                                        'boxplotvars':[]}]
                        )
    
    # instantiate the MIRV pipeline for RADCURE
    mp_radc = MIRVPipe(     radiomics = '../../procdata/RADCURE/RADCURE_radiomics.csv',
                            clinical  = [   '../../rawdata/RADCURE/RADCURE_clinical.csv', 
                                        {'corrvars':[],
                                        'boxplotvars':['SEX','PTUMSITE','HPV']}], 
                            recist    = [   None,
                                        {'corrvars':[],
                                        'boxplotvars':[]}], 
                            survival  = [   '../../rawdata/RADCURE/RADCURE_clinical.csv',
                                        {'survcols':[('T_OS','E_OS')],
                                        'yearConversion':1.0}], 
                            ctdna     = ['../../rawdata/RADCURE/RADCURE_ctdna.csv',
                                        {'corrvars':['Log Baseline','Log Mid RT'],
                                        'boxplotvars':[]}]
                        )
    
    # instantiate the MIRV pipeline for RADCURE
    mp_octn = MIRVPipe(     radiomics = '../../procdata/OCTANE/OCTANE_radiomics.csv',
                            clinical  = [   '../../rawdata/OCTANE/OCTANE_clinical.csv', 
                                        {'corrvars':[],
                                        'boxplotvars':['Cancer Type']}], 
                            recist    = [   None,
                                        {'corrvars':[],
                                        'boxplotvars':[]}], 
                            survival  = [   '../../rawdata/OCTANE/OCTANE_clinical.csv',
                                        {'survcols':[('T_OS','E_OS')],
                                        'yearConversion':1.0}], 
                            ctdna     = [   '../../rawdata/OCTANE/OCTANE_ctDNA.csv',
                                        {'corrvars':['tumour-fraction-zviran-adj'],
                                        'boxplotvars':['tumour-fraction-zviran-adj']}]
                        )
    
    # run the pipeline
    results = mp_sarc.run()

# %%

import pandas as pd
rad_df = results[0][0]
vol_df = results[0][1]
corr_df = results[2]
box_df = results[3]

# %%
vol_df = vol_df[vol_df['USUBJID'].isin(box_df['USUBJID'])]

vol_changes = vol_df['VOLUME_CHANGE_ABS'].groupby(vol_df['USUBJID']).sum()

# merge box_df with vol_changes
box_df = box_df.merge(vol_changes, on='USUBJID', how='left')


# %%

# convert to cc
box_df['VOLUME_CHANGE_ABS'] = box_df['VOLUME_CHANGE_ABS'] * 2 / 1000

# remove data points where RECIST is 'NE'
box_df = box_df[box_df['RECIST'] != 'NE']


# %%
# make a boxplot of VOLUME_CHANGE_ABS by RECIST
import seaborn as sns
import matplotlib.pyplot as plt

# order RECIST 'PR', 'SD', 'PD'
box_df['RECIST'] = box_df['RECIST'].astype('category')
box_df['RECIST'] = box_df['RECIST'].cat.reorder_categories(['PR', 'SD', 'PD'])
box_df['RECIST'] = box_df['RECIST'].cat.as_ordered()

plt.figure(figsize=(8, 6))
sns.boxplot(x='RECIST', y='VOLUME_CHANGE_ABS', data=box_df, showfliers=False)
sns.despine(right=True,bottom=True)
plt.ylabel('Volume Change (cc)')

# %%

# Order by absolute volume change
box_df = box_df.sort_values(by='VOLUME_CHANGE_ABS', ascending=True)

# Create a bar plot
plt.figure(figsize=(12, 6))
sns.barplot(x=box_df['USUBJID'], y=box_df['VOLUME_CHANGE_ABS'], hue=box_df['RECIST'], dodge=False)
plt.xticks(rotation=90)
plt.xlabel('Patient ID')
plt.ylabel('Volume Change (cc)')
plt.title('Volume Change by Patient and RECIST Status')
plt.legend(title='RECIST Status')
plt.show()

# Create a bar plot with individual volume changes
plt.figure(figsize=(12, 6))
sns.barplot(x=box_df['USUBJID'], y=box_df['VOLUME_CHANGE_ABS'], hue=box_df['RECIST'], dodge=False)
plt.ylabel('Volume Change (cc)')
# Plot individual volume changes for each patient
for patient_id in box_df['USUBJID'].unique():
    patient_vol_changes = vol_df[vol_df['USUBJID'] == patient_id]['VOLUME_CHANGE_ABS']
    patient_vol_changes = patient_vol_changes * 2 / 1000  # convert to cc
    x = [patient_id] * len(patient_vol_changes)
    plt.scatter(x, patient_vol_changes, color='black', alpha=0.5, zorder=2)

# Add custom legend entry for individual volume changes
handles, labels = plt.gca().get_legend_handles_labels()
handles.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='black', markersize=10, alpha=0.5, linestyle='None'))
labels.append('Individual Lesion Response')
plt.legend(handles=handles, labels=labels, title='RECIST Status')
sns.despine(right=True,bottom=True)
# Remove x-axis ticks and labels
plt.xticks([])
plt.xlabel('')
# plt.xlabel('Patient ID')
plt.ylabel('Volume Change (cc)')
# plt.title('Volume Change by Patient and RECIST Status')
plt.savefig('drew-waterfall.png', dpi=300, bbox_inches='tight')
plt.show()

# from rpy2.robjects import pandas2ri, r
# from rpy2.robjects.packages import importr
# from rpy2.robjects.conversion import localconverter

# # Assuming results is already defined and contains the necessary data
# results = mp_sarc.run()

# # Extract correlation matrix and p-values from results
# cor = results[4]
# pval = results[5]

# # Ensure that the correlation matrix and p-value matrix are numeric
# cor = cor.apply(pd.to_numeric, errors='coerce')
# pval = pval.apply(pd.to_numeric, errors='coerce')

# # Activate the automatic conversion of pandas objects to R objects
# pandas2ri.activate()

# # Import the corrplot package from R
# corrplot = importr('corrplot')

# # Convert pandas DataFrames to R data frames
# with localconverter(pandas2ri.converter):
#     cor_r = pandas2ri.py2rpy(cor)
#     pval_r = pandas2ri.py2rpy(pval)

# # Call the R corrplot function
# corrplot.corrplot(cor_r, type="upper", order="hclust", p_mat=pval_r, sig_level=0.05, insig="blank")
# %%
