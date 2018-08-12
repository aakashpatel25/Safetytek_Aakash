from restplus.api.model import data_processing as cleaning
import pandas as pd, numpy as np

def get_ml_model_data(args):
	"""
		Given list of arguments converts the arguments into pandas dataframe and processing the data and reutrns numpy array of 
		features that are going to be input to the machine learning model.

		@param args: list of argument obtained in the request

		@return numpy_array: array of features.
	"""
	pandas_df = pd.DataFrame(args, index=[0])
	pandas_df.columns = ['Age', 'Class Name', 'Department Name', 'Division Name', 'Positive Feedback Count', 'Review Text', 'Title']
	pd_df, other_data = cleaning.get_one_hot_encode_for_model(pandas_df, train=False)

	pd_df = cleaning.preprocess_data(pd_df, train=False, tokenizer_name='tokenizer.pickle')
	dataset = np.hstack((pd_df, other_data))

	return dataset