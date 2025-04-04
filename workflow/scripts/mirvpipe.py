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
        col_label = 'diagnostics_Versions_PyRadiomics' if 'all' in self.radiomicsData else 'STUDY'
        dp = DataProcessing(patient_id=self.patient_id,    # column name for patient ID in PyRadiomics output
                    labels={'col': col_label, 'pre': 'baseline', 'post': 'cycle2'},
                    resp_thresh=33)  
        print('Data Processing Class Initialized')

        # radiomics data loading and feature reduction
        rad_volume_results = dp.loadRadiomics(self.radiomicsData)

        if self.ctdnaData[0] is not None:
            ctdna_df = dp.loadData(self.ctdnaData[0], self.ctdnaData[1]['corrvars'] + [self.patient_id])
            ctdna_df[self.patient_id] = ctdna_df[self.patient_id].astype(str)
            rad_volume_results[0] = rad_volume_results[0][rad_volume_results[0][self.patient_id].isin(ctdna_df[self.patient_id])]

        # feature reduction
        radiomics_red = dp.radiomicsFeatureReduction(rad_volume_results[0])

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
            mirv_cols = ['MaxTumorSim', 'MaxEuclDist']
            for col in mirv_cols:
                dp.compareSurvival(survival_df, col, self.survivalData[1]['survcols'],savefigFlag=True)
        else:
            survival_df = None

        # ----- CORRELATION ANALYSIS -----
        data_types = [self.clinicalData,self.recistData,self.ctdnaData]
        corr_df = outcome_df.copy()
        for i, data in enumerate(data_types):
            if data[0] is not None and data[1]['corrvars'] is not None:
                data_df = dp.loadData(data[0], data[1]['corrvars'] + [self.patient_id])
                data_df = data_df[data_df[self.patient_id].isin(outcome_df[self.patient_id])]
                corr_df[self.patient_id] = corr_df[self.patient_id].astype(str)
                data_df[self.patient_id] = data_df[self.patient_id].astype(str)
                corr_df = corr_df.merge(data_df, on=self.patient_id, how='left', suffixes=('', '_drop')).reset_index(drop=True)
        corr_df = corr_df.drop(columns=[self.patient_id])

        # correlation matrix
        if self.radiomicsData == '../../procdata/SARC021/radiomics-all.csv':
            cormat, pmat = None, None
        else:
            cormat, pmat = dp.correlationMatrix(corr_df,drop_cols=['AvgTumorSim','AvgEuclDist'],savefigFlag=False,invertFlag=False) 
        
        return [rad_volume_results,survival_df,corr_df,cormat,pmat]

if __name__ == '__main__':
    
    # instantiate the MIRV pipeline for all patients with survival data
    mp_sarc_surv = MIRVPipe(radiomics = '../../procdata/SARC021/radiomics-all.csv',
                            clinical  = [   '../../rawdata/SARC021/baseline-all.csv', 
                                        {'corrvars':['CPCELL','AGE','BECOG'],
                                        'boxplotvars':['CPCELL']}], 
                            recist    = [   '../../rawdata/SARC021/recist-all.csv',
                                        {'corrvars':['RECIST'],
                                         'boxplotvars':['RECIST']}], 
                            survival  = [   '../../rawdata/SARC021/survival-all.csv',
                                        {'survcols':[('T_OS', 'E_OS')],
                                                     'yearConversion':1}], 
                            ctdna     = [None,None]
                        )
    
    # instantiate the MIRV pipeline for patients with ctDNA
    mp_sarc_liqb = MIRVPipe(radiomics = '../../procdata/SARC021/radiomics-all.csv',
                            clinical  = [   '../../rawdata/SARC021/baseline-all.csv', 
                                        {'corrvars':[],
                                        'boxplotvars':['CPCELL']}], 
                            recist    = [   '../../rawdata/SARC021/recist-all.csv',
                                        {'corrvars':[],
                                        'boxplotvars':['RECIST']}], 
                            survival  = [None,None], 
                            ctdna     = [ '../../rawdata/SARC021/ctdna-lms.csv',
                                        {'corrvars':['Pretreatment_bin','Pre-cycle3_bin'],
                                        'boxplotvars':['Pretreatment_bin','Pre-cycle3_bin']}]
                        )
    
    # instantiate the MIRV pipeline lung subset, no ctDNA
    mp_sarc_lung = MIRVPipe(radiomics = '../../procdata/SARC021/radiomics-lung.csv',
                            clinical  = [   '../../rawdata/SARC021/baseline-all.csv', 
                                        {'corrvars':[],
                                        'boxplotvars':['CPCELL']}], 
                            recist    = [   '../../rawdata/SARC021/recist-all.csv',
                                        {'corrvars':[],
                                        'boxplotvars':['RECIST']}], 
                            survival  = [None,None], 
                            ctdna     = [None,None]
                        )
    
    # run the pipeline by subset
    # results_lung = mp_sarc_lung.run()
    # results_liqb = mp_sarc_liqb.run()
    results_surv = mp_sarc_surv.run()
    print('MIRV pipeline complete')

    # %% Additional survival analysis

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from lifelines import CoxPHFitter, KaplanMeierFitter
    from lifelines.statistics import logrank_test
    from lifelines.statistics import multivariate_logrank_test
    from lifelines.statistics import proportional_hazard_test

    # update matplotlib parameters for black background
    plt.rcParams.update(plt.rcParamsDefault)
    plt.rcParams.update({'font.size': 24})

    # Clean-up the data
    df_surv = results_surv[1]
    df_corr = results_surv[2]
    df_merged = df_surv.merge(df_corr, on=['AvgEuclDist', 'AvgTumorSim', 
                                           'Brange', 'Bstddev', 'Btotal',
                                           'MaxEuclDist', 'MaxTumorSim'])

    df_merged.drop(columns=['AvgEuclDist', 'AvgTumorSim',
                            'Brange', 'Bstddev', 'group',
                            'USUBJID'], inplace=True)

    # Cox Proportional Hazards Regression (multivariable)
    cph = CoxPHFitter()
    df_merged = df_merged.dropna()

    # Feature formatting - patient age, tumor burden, CPCELL, RECIST
    df_merged['AGE'] = df_merged['AGE'] >= 65 
    df_merged['Btotal'] = df_merged['Btotal'] >= df_merged['Btotal'].median()
    df_merged['CPCELL'] = df_merged['CPCELL'].astype('category').cat.codes
    df_merged['RECIST'] = df_merged['RECIST'].astype('category').cat.codes

    df_merged.rename(columns={'MaxTumorSim':'MIRV (max) Dissimilarity',
                              'MaxEuclDist':'MIRV (max) Distance',
                              'T_OS':'Overall survival (years)',
                              'E_OS':'Overall survival event',
                              'AGE':'Patient Age >= 65',
                              'Btotal':'Baseline Volume (total)',
                              'CPCELL':'Histologic classification',
                              'BECOG':'ECOG Performance Status'},
                     inplace=True)

    # add interactions terms between clinical variables and MIRV
    df_merged['ECOG x MIRV Dissimilarity'] = df_merged['ECOG Performance Status'] * df_merged['MIRV (max) Dissimilarity']
    df_merged['Histology x MIRV Dissimilarity'] = df_merged['Histologic classification'] * df_merged['MIRV (max) Dissimilarity']
    df_merged['Age x MIRV Dissimilarity'] = df_merged['Patient Age >= 65'] * df_merged['MIRV (max) Dissimilarity']
    df_merged['RECIST x MIRV Dissimilarity'] = df_merged['RECIST'] * df_merged['MIRV (max) Dissimilarity']

    # add interactions terms between clinical variables and MIRV (max) Distance
    # df_merged['ECOG x MIRV Distance'] = df_merged['ECOG Performance Status'] * df_merged['MIRV (max) Distance']
    # df_merged['Histology x MIRV Distance'] = df_merged['Histologic classification'] * df_merged['MIRV (max) Distance']
    # df_merged['Age x MIRV Distance'] = df_merged['Patient Age >= 65'] * df_merged['MIRV (max) Distance']
    # df_merged['RECIST x MIRV Distance'] = df_merged['RECIST'] * df_merged['MIRV (max) Distance']

    # Fit the Cox Proportional Hazards model
    cph = CoxPHFitter()
    cph.fit(df_merged, duration_col='Overall survival (years)', event_col='Overall survival event')

    # Print the summary of the fitted model
    cph.print_summary()

    # Plot the coefficients
    cph.plot()
    sns.despine(offset=10,trim=True)
    plt.show()

# %%
