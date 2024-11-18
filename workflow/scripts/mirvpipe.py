# Initialize
import os
os.chdir(os.path.dirname(__file__))
from functionals import DataProcessing

class MIRVPipe:
    def __init__(self, 
                 radiomicsPath = '../../procdata/radiomics_file.csv', 
                 ):
        self.radiomicsData = radiomicsPath

    def run(self):
        
        # instantiate the data processing class
        dp = DataProcessing(patient_id = 'Patient',    # column name for patient ID in PyRadiomics output
                            treatment_id = 'Study',    # column name for treatment ID in PyRadiomics output (assuming both pre-and post-treatment images were processed together)
                            pre_tag = 'baseline',      # tag for pre-treatment images
                            post_tag = 'cycle2')       # tag for post-treatment images

        # load the radiomics data
        radiomics_pre, volume_df = dp.loadRadiomics(self.radiomicsData)
        # radiomics feature reduction
        radiomics_red = dp.radiomicsFeatureReduction(radiomics_pre)
        
        # determine the response outcomes
        response_df = dp.calcResponseOutcomes(volume_df)
        # calculate the MIRV metrics and add to the response dataframe
        outcome_df = dp.calcMIRVMetrics(radiomics_red, response_df)

        # output correlation matrix with significance values
        cormat, pmat = dp.correlationMatrix(outcome_df,drop_cols=[],savefigFlag=False) 
        
        return radiomics_red, response_df, outcome_df #, cormat, pmat  

if __name__ == '__main__':
    
    mp = MIRVPipe(radiomicsPath='../../procdata/results-all.csv')
    
    # run the pipeline
    radiomics_red, response_df, outcome_df = mp.run()