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
import scipy
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test
from lifelines.statistics import multivariate_logrank_test
from lifelines.statistics import proportional_hazard_test
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import roc_auc_score, roc_curve
import warnings
from statsmodels.stats.multitest import multipletests

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
plt.savefig('../../results/cox_model_coefficients.png', dpi=600, transparent=True, bbox_inches='tight')
plt.show()


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test
from lifelines.plotting import add_at_risk_counts

from matplotlib.lines import Line2D
from matplotlib.patches import Patch
# plt.style.use('dark_background')

def plot_km_with_risk_table(kmf_high, kmf_low, fig, mirv_col='MIRV (max) Dissimilarity', hist=0):
    ax = fig.add_subplot(111)
    # Plot survival functions
    kmf_high.plot_survival_function(ax=ax, ci_show=True, color='#764179')
    kmf_low.plot_survival_function(ax=ax, ci_show=True, color='#3BAEE6')

    # Define custom legend to match the plot elements
    legend_elements = [
        Line2D([0], [0], color='#764179', lw=2, label='High'),
        Line2D([0], [0], color='#3BAEE6', lw=2, label='Low')
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    # Add labels and risk table
    ax.set_ylabel('Survival Probability')
    ax.set_xlabel('Time (years)')
    sns.despine(trim=True, offset=10, ax=ax)
    add_at_risk_counts(kmf_high, kmf_low, ax=ax, fig=fig)
    plt.savefig(f'../../results/km_plot_{mirv_col}_histology_{hist}.png', dpi=600, transparent=True,bbox_inches='tight')

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
        plot_km_with_risk_table(kmf_high, kmf_low, fig,mirv_col=mirv_col, hist=hist)
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

def analyze_by_histology_with_fdr(df, mirv_cols):
    """
    Analyze survival by histology and MIRV metrics, output p-values, and correct for multiple testing using FDR.

    Parameters:
    df (pd.DataFrame): Merged DataFrame containing survival and MIRV metrics.
    mirv_cols (list): List of MIRV metrics to analyze.

    Returns:
    pd.DataFrame: DataFrame containing histology, MIRV metric, raw p-values, and FDR-corrected p-values.
    """
    results = []
    
    for mirv_col in mirv_cols:
        for hist in df['Histologic classification'].unique():
            df_subset = df[df['Histologic classification'] == hist].copy()
            median_mirv = df_subset[mirv_col].median()
            df_subset['MIRV_group'] = np.where(df_subset[mirv_col] > median_mirv, 'High', 'Low')
            
            # Perform log-rank test
            logrank_results = logrank_test(
                df_subset.loc[df_subset['MIRV_group'] == 'High', 'Overall survival (years)'],
                df_subset.loc[df_subset['MIRV_group'] == 'Low', 'Overall survival (years)'],
                event_observed_A=df_subset.loc[df_subset['MIRV_group'] == 'High', 'Overall survival event'],
                event_observed_B=df_subset.loc[df_subset['MIRV_group'] == 'Low', 'Overall survival event']
            )
            
            # Collect results
            results.append({
                'Histology': hist,
                'MIRV Metric': mirv_col,
                'Raw p-value': logrank_results.p_value
            })
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Apply FDR correction
    results_df['FDR-corrected p-value'] = multipletests(results_df['Raw p-value'], method='fdr_bh')[1]
    
    return results_df

# Example usage
mirv_cols = ['MIRV (max) Dissimilarity', 'MIRV (max) Distance']
results_df = analyze_by_histology_with_fdr(df_merged, mirv_cols)

# Print results
print(results_df)

# Save results to CSV
results_df.to_csv('../../results/histology_mirv_pvalues.csv', index=False)
# %%
analyze_by_histology(df_merged, 'MIRV (max) Dissimilarity')

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
        tprs, aucs, probs, reals = [], [], [], []
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
            probs.append(probas_)
            reals.append(y[test])
        probs = np.concatenate(probs)
        reals = np.concatenate(reals)
        mean_tpr = np.mean(tprs, axis=0)
        mean_auc = np.mean(aucs)
        std_auc = np.std(aucs)
        roc_results[model_name] = (mean_fpr, mean_tpr, mean_auc, std_auc, probs, reals)
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

# %%
# --- Plotting ---

colors = ['#ffc20a','#0c7bdc','#d41159','#40b0a6']
i = 0
fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
for idx, (analysis_type, roc_results) in enumerate(roc_results_all.items()):
    ax = axes[idx]
    for model_name, (mean_fpr, mean_tpr, mean_auc, std_auc, probs, reals) in roc_results.items():
        ax.plot(mean_fpr, mean_tpr, lw=2, label=f"{model_name}",color=colors[i])
        i += 1
    ax.plot([0, 1], [0, 1], 'w--', lw=1)
    ax.set_xlabel('False Positive Rate',color='white')
    if idx == 0:
        ax.set_ylabel('True Positive Rate', color='white')
    ax.set_title('Lung Subset' if analysis_type == 'lung' else 'ctDNA Subset')
    ax.grid(False)
    i = 0
# Shared legend
handles, labels = axes[1].get_legend_handles_labels()
fig.legend(handles, labels, loc='center left', bbox_to_anchor=(0.85, 0.5), frameon=False)
# invert color scheme
plt.style.use('dark_background')
plt.tight_layout(rect=[0, 0, 0.85, 1])
sns.despine(trim=True, offset=10)
plt.savefig('../../results/roc_curves_comparative.png', dpi=600, transparent=True, bbox_inches='tight')
plt.show()

# --- Text output for reporting ---
for analysis_type, roc_results in roc_results_all.items():
    print(f"\n{analysis_type.capitalize()} Subset AUCs:")
    for model_name, (_, _, mean_auc, std_auc, _, _) in roc_results.items():
        print(f"  {model_name}: {mean_auc:.2f} ± {std_auc:.2f}")

print('-----------------------------------')
print('---------PIPELINE FINISHED---------')
print('-----------------------------------')

# %% DELONG TEST

def Delong_test(true, prob_A, prob_B):
    """
    Perform DeLong's test for comparing the AUCs of two models.

    Parameters
    ----------
    true : array-like of shape (n_samples,)
        True binary labels in range {0, 1}.
    prob_A : array-like of shape (n_samples,)
        Predicted probabilities by the first model.
    prob_B : array-like of shape (n_samples,)
        Predicted probabilities by the second model.

    Returns
    -------
    z_score : float
        The z score from comparing the AUCs of two models.
    p_value : float
        The p value from comparing the AUCs of two models.

    Example
    -------
    >>> true = [0, 1, 0, 1]
    >>> prob_A = [0.1, 0.4, 0.35, 0.8]
    >>> prob_B = [0.2, 0.3, 0.4, 0.7]
    >>> z_score, p_value = Delong_test(true, prob_A, prob_B)
    >>> print(f"Z-Score: {z_score}, P-Value: {p_value}")
    """

    def compute_midrank(x):
        J = np.argsort(x)
        Z = x[J]
        N = len(x)
        T = np.zeros(N, dtype=np.float64)
        i = 0
        while i < N:
            j = i
            while j < N and Z[j] == Z[i]:
                j += 1
            T[i:j] = 0.5 * (i + j - 1)
            i = j
        T2 = np.empty(N, dtype=np.float64)
        T2[J] = T + 1
        return T2

    def compute_ground_truth_statistics(true):
        assert np.array_equal(np.unique(true), [0, 1]), "Ground truth must be binary."
        order = (-true).argsort()
        label_1_count = int(true.sum())
        return order, label_1_count

    # Prepare data
    order, label_1_count = compute_ground_truth_statistics(np.array(true))
    sorted_probs = np.vstack((np.array(prob_A), np.array(prob_B)))[:, order]

    # Fast DeLong computation starts here
    m = label_1_count  # Number of positive samples
    n = sorted_probs.shape[1] - m  # Number of negative samples
    k = sorted_probs.shape[0]  # Number of models (2)

    # Initialize arrays for midrank computations
    tx, ty, tz = [np.empty([k, size], dtype=np.float64) for size in [m, n, m + n]]
    for r in range(k):
        positive_examples = sorted_probs[r, :m]
        negative_examples = sorted_probs[r, m:]
        tx[r, :], ty[r, :], tz[r, :] = [
            compute_midrank(examples) for examples in [positive_examples, negative_examples, sorted_probs[r, :]]
        ]

    # Calculate AUCs
    aucs = tz[:, :m].sum(axis=1) / (m * n) - (m + 1.0) / (2.0 * n)

    # Compute variance components
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m

    # Compute covariance matrices
    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n

    # Calculating z-score and p-value
    l = np.array([[1, -1]])
    z = np.abs(np.diff(aucs)) / np.sqrt(np.dot(np.dot(l, delongcov), l.T)).flatten()
    p_value = scipy.stats.norm.sf(abs(z)) * 2

    z_score = -z[0].item()
    p_value = p_value[0].item()

    return z_score, p_value

# %% TESTING STATS


# Perform pairwise comparisons of models for each dataset using DeLong's test

for dataset_name, roc_results in roc_results_all.items():
    print(f"\nComparing models for dataset: {dataset_name.capitalize()}")
    model_names = list(roc_results.keys())
    
    # Iterate through all pairs of models
    for i in range(len(model_names)):
        for j in range(i + 1, len(model_names)):
            model_A = model_names[i]
            model_B = model_names[j]
            
            # Extract true labels and predicted probabilities
            true = roc_results[model_A][5]
            prob_A = roc_results[model_A][4]
            prob_B = roc_results[model_B][4]
            
            # Perform DeLong's test
            z_score, p_value = Delong_test(true, prob_A, prob_B)
            
            # Print results
            print(f"  {model_A} vs {model_B}: Z-Score = {z_score:.4f}, P-Value = {p_value:.4f}")

# %%
