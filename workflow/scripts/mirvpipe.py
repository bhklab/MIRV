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
                dp.compareSurvival(survival_df, col, self.survivalData[1]['survcols'],plotFlag=False)
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
            cormat, pmat = dp.correlationMatrix(corr_df,drop_cols=[],savefigFlag=False,invertFlag=False) 
        
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
    mp_sarc_liqb = MIRVPipe(radiomics = '../../procdata/SARC021/radiomics-liqb.csv',
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
    results_lung = mp_sarc_lung.run()

    results_liqb = mp_sarc_liqb.run()
    results_surv = mp_sarc_surv.run()
    print('MIRV pipeline complete')

# %% POST-HOC ANALYSIS

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test
from lifelines.statistics import multivariate_logrank_test
from lifelines.statistics import proportional_hazard_test
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import roc_auc_score, roc_curve
import warnings

# update matplotlib parameters for black background
plt.rcParams.update(plt.rcParamsDefault)
plt.rcParams.update({'font.size': 24})

# Clean-up the data
df_surv = results_surv[1]
df_corr = results_surv[2]
df_merged = df_surv.merge(df_corr, on=[ 'Brange', 'Bstddev', 'Btotal',
                                        'MaxEuclDist', 'MaxTumorSim'])

df_merged.drop(columns=['Brange', 'Bstddev', 'group',
                        'USUBJID'], inplace=True)

# Cox Proportional Hazards Regression (multivariable)
cph = CoxPHFitter()
df_merged = df_merged.dropna()
print(df_merged.CPCELL.value_counts())

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

# add interactions terms between clinical variables and MIRV (max) Distance (does not change overall result - removing to make the plot more readable)
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


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test
from lifelines.plotting import add_at_risk_counts

def plot_km_with_risk_table(kmf_high, kmf_low, fig):
    ax = fig.add_subplot(111)
    kmf_high.plot_survival_function(ax=ax, ci_show=True, color='#764179')
    kmf_low.plot_survival_function(ax=ax, ci_show=True, color='#3BAEE6')
    ax.set_ylabel('Survival Probability')
    ax.legend(['High', 'Low'], loc='upper right')
    sns.despine(trim=True, offset=10, ax=ax)
    add_at_risk_counts(kmf_high, kmf_low, ax=ax, fig=fig)
    ax.set_xlabel('Time (years)')
    plt.show()

def analyze_by_histology(df, mirv_col):
    for hist in df['Histologic classification'].unique():
        df_subset = df[df['Histologic classification'] == hist].copy()
        median_mirv = df_subset[mirv_col].median()
        df_subset['MIRV_group'] = np.where(df_subset[mirv_col] > median_mirv, 'High', 'Low')
        kmf_high = KaplanMeierFitter()
        kmf_low = KaplanMeierFitter()
        for group, kmf in zip(['High', 'Low'], [kmf_high, kmf_low]):
            mask = df_subset['MIRV_group'] == group
            kmf.fit(
                durations=df_subset.loc[mask, 'Overall survival (years)'],
                event_observed=df_subset.loc[mask, 'Overall survival event'],
                label=group
            )
        fig = plt.figure(figsize=(8, 6))
        plot_km_with_risk_table(kmf_high, kmf_low, fig)
        results = logrank_test(
            df_subset.loc[df_subset['MIRV_group'] == 'High', 'Overall survival (years)'],
            df_subset.loc[df_subset['MIRV_group'] == 'Low', 'Overall survival (years)'],
            event_observed_A=df_subset.loc[df_subset['MIRV_group'] == 'High', 'Overall survival event'],
            event_observed_B=df_subset.loc[df_subset['MIRV_group'] == 'Low', 'Overall survival event']
        )
        print(f"Log-rank test p-value for Histologic classification = {hist}: {results.p_value}")
        cph = CoxPHFitter()
        cph.fit(df_subset[[mirv_col, 'Overall survival (years)', 'Overall survival event']],
                duration_col='Overall survival (years)', event_col='Overall survival event')
        print(f"Univariable Cox model for {mirv_col} in Histology = {hist}:")
        cph.print_summary()

# Example usage:
mirv = ['MIRV (max) Dissimilarity', 'MIRV (max) Distance']
idx = 0  # or loop over both metrics if desired
analyze_by_histology(df_merged, mirv[idx])

# %% COMPARATIVE ROC CURVES

def prepare_df(results, clinical_path, recist_path):
    df_vol = results[0][1]
    btotal = df_vol.groupby('USUBJID')['VOLUME_PRE'].sum()
    df_burden = pd.DataFrame({'USUBJID': btotal.index, 'Btotal': btotal.values})
    df = df_burden.merge(results[2], on='Btotal', how='left').dropna()
    df_clin = pd.read_csv(clinical_path)
    df_recist = pd.read_csv(recist_path)
    df = df.merge(df_clin[['USUBJID', 'CPCELL', 'AGE', 'BECOG']], on='USUBJID', how='left')
    df = df.merge(df_recist[['USUBJID', 'RECIST']], on='USUBJID', how='left')
    # Encode
    df['CPCELL'] = df['CPCELL'].astype('category').cat.codes
    df['AGE'] = (df['AGE'] >= 65).astype(int)
    df['BECOG'] = df['BECOG'].astype('category').cat.codes
    recist_order = ['NE', 'CR', 'PR', 'SD', 'PD']
    df['RECIST'] = pd.Categorical(df['RECIST'], categories=recist_order, ordered=True).codes
    return df

def run_logistic_cv(df, y_col, feature_sets, pretty_names, param_grid, cv):
    roc_results = {}
    feature_importances = {}
    Xy = df.copy()
    y = Xy[y_col].values
    for model_name, feats in feature_sets.items():
        X = Xy[feats].values
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            grid = GridSearchCV(LogisticRegression(), param_grid, cv=cv, scoring='roc_auc', n_jobs=1)
            grid.fit(X, y)
        print(f"{model_name}: Best params: {grid.best_params_}, Mean AUC: {grid.best_score_:.2f}")
        # ROC curve
        tprs, aucs = [], []
        mean_fpr = np.linspace(0, 1, 100)
        for train, test in cv.split(X, y):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = LogisticRegression(**grid.best_params_)
                model.fit(X[train], y[train])
            probas_ = model.predict_proba(X[test])[:, 1]
            fpr, tpr, _ = roc_curve(y[test], probas_)
            auc = roc_auc_score(y[test], probas_)
            aucs.append(auc)
            tprs.append(np.interp(mean_fpr, fpr, tpr))
            tprs[-1][0] = 0.0
        mean_tpr = np.mean(tprs, axis=0)
        mean_auc = np.mean(aucs)
        std_auc = np.std(aucs)
        roc_results[model_name] = (mean_fpr, mean_tpr, mean_auc, std_auc)
        # Feature importance
        coefs = grid.best_estimator_.coef_[0]
        feature_importances[model_name] = dict(zip([pretty_names[f] for f in feats], coefs))
    return roc_results, feature_importances

# --- Setup ---
feature_sets = {
    "Baseline": ['Btotal', 'CPCELL', 'AGE', 'BECOG'],
    "Baseline+RECIST": ['Btotal', 'CPCELL', 'AGE', 'BECOG', 'RECIST'],
    "Baseline+MIRV": ['Btotal', 'CPCELL', 'AGE', 'BECOG', 'MaxTumorSim', 'MaxEuclDist'],
    "Baseline+RECIST+MIRV": ['Btotal', 'CPCELL', 'AGE', 'BECOG', 'RECIST', 'MaxTumorSim', 'MaxEuclDist'],
}
pretty_names = {
    "Btotal": "Baseline Volume (total)",
    "CPCELL": "Histologic Classification",
    "AGE": "Patient Age >= 65",
    "BECOG": "ECOG Performance Status",
    "RECIST": "RECIST",
    "MaxTumorSim": "MIRV Dissimilarity",
    "MaxEuclDist": "MIRV Distance"
}
param_grid = {'C': np.logspace(-3, 3, 7), 'penalty': ['l2'], 'solver': ['lbfgs'], 'max_iter': [1000]}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# --- Run analyses ---
analyses = {
    'lung': {
        'results': results_lung,
        'y_col': 'Mixed Response'
    },
    'liqb': {
        'results': results_liqb,
        'y_col': 'Pre-cycle3_bin'
    }
}
roc_results_all = {}
feature_importances_all = {}

for analysis_type, params in analyses.items():
    df = prepare_df(params['results'], '../../rawdata/SARC021/baseline-all.csv', '../../rawdata/SARC021/recist-all.csv')
    roc_results, feature_importances = run_logistic_cv(df, params['y_col'], feature_sets, pretty_names, param_grid, cv)
    roc_results_all[analysis_type] = roc_results
    feature_importances_all[analysis_type] = feature_importances

# --- Plotting ---
fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
for idx, (analysis_type, roc_results) in enumerate(roc_results_all.items()):
    ax = axes[idx]
    for model_name, (mean_fpr, mean_tpr, mean_auc, std_auc) in roc_results.items():
        ax.plot(mean_fpr, mean_tpr, lw=2, label=f"{model_name} (AUC={mean_auc:.2f}±{std_auc:.2f})")
    ax.plot([0, 1], [0, 1], 'k--', lw=1)
    ax.set_xlabel('False Positive Rate')
    if idx == 0:
        ax.set_ylabel('True Positive Rate')
    ax.set_title('Lung Subset' if analysis_type == 'lung' else 'ctDNA Subset')
    ax.grid(False)
# Shared legend
handles, labels = axes[1].get_legend_handles_labels()
fig.legend(handles, labels, loc='center left', bbox_to_anchor=(0.85, 0.5), frameon=False)
plt.tight_layout(rect=[0, 0, 0.85, 1])
sns.despine(trim=True, offset=10)
plt.savefig('../../results/roc_curves_comparative.png', dpi=300, bbox_inches='tight')
plt.show()

# --- Text output for reporting ---
for analysis_type, roc_results in roc_results_all.items():
    print(f"\n{analysis_type.capitalize()} Subset AUCs:")
    for model_name, (_, _, mean_auc, std_auc) in roc_results.items():
        print(f"  {model_name}: {mean_auc:.2f} ± {std_auc:.2f}")

print('-----------------------------------')
print('---------PIPELINE FINISHED---------')
print('-----------------------------------')