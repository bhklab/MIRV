import pickle

def load_pickle(file_path):
    with open(file_path, 'rb') as file:
        return pickle.load(file)

survival_data = load_pickle('../../procdata/SARC021/results_surv.pkl')

surv = survival_data[2]