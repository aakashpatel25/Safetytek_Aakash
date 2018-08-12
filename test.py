import numpy as np, pandas as pd, data_processing as cleaning, os
from keras.models import load_model

def main(file_path, token_name, rating_model, recom_model):
	data = pd.read_csv(file_path, encoding='utf-8')
	pd_df = data.replace(np.nan, '', regex=True)
	
	pd_df, other_data = cleaning.get_one_hot_encode_for_model(pd_df, train=False)

	pd_df = cleaning.preprocess_data(pd_df, train=False, tokenizer_name=token_name)
	pd_df = np.hstack((pd_df, other_data))

	rating_model = load_model(rating_model)
	recommendation_model = load_model(recom_model)

	ratings = rating_model.predict(pd_df, verbose=1)
	recommendations = recommendation_model.predict(pd_df, verbose=1)

	rating = [x.index(max(x)) for x in ratings.tolist()]
	recommendations = [x.index(max(x)) for x in recommendations.tolist()]

	path = 'prediction'
	with open (os.path.join(path,"rating.txt"),"w")as fp:
		fp.write("\n".join(rating))

	with open (os.path.join(path,"recommendations.txt"),"w")as fp:
		fp.write("\n".join(recommendations))
	

if __name__=="__main__":
	import argparse
	parser = argparse.ArgumentParser(description = 'Review Rating and Recommendation Neural Network Training Argument Parser')
	parser.add_argument('--file_path', type=str, default='data.csv' ,help='Enter the file path of the dataset')
	parser.add_argument('--tokenizer', type=str, default='tokenizer.pickle' ,help='Enter tokenizer to be loaded')
	parser.add_argument('--rating_model', type=str, default='rating_model.h5', 
						help='Enter the name for saving the rating model')
	parser.add_argument('--recom_model', type=str, default='recommendation_model.h5', 
						help='Enter the name for saving the recommendation model')
	args = parser.parse_args()
	main(args.file_path, args.tokenizer, args.rating_model, args.recom_model)