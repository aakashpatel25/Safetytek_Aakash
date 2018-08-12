from keras.layers import Conv1D, Input, Dense, concatenate, Activation, GlobalMaxPooling1D, Flatten, Dropout, Lambda, RepeatVector
from sklearn.model_selection import train_test_split
from keras.layers.embeddings import Embedding
import numpy as np, pandas as pd, data_processing as cleaning
from keras.utils import plot_model
from keras.models import Model

def get_nlp_model(vocabSize, input_size, other_feature, output_size, word_dimension):
	"""
		Builds model architecutre and returns compiled model.

		Model architecture is inspired by 'A Sensitivity Analysis of (and Practitioners Guide to) 
		Convolutional Neural Networks for Sentence Classification' research paper. This architecture is manily used to 
		handle review title and texts. Another component of the model architecture is to use other defined features.
		Other features are combined with high level CNN word features. Eventually these features are used to classify
		the output.

		@param vocabSize: size of vocabulary for the purpose of initializing word embeddings
		@param input_size: total size of the sentence features and other features.
		@param other_feature: denotes size of other features present in input apart from text sqeuences.
		@param output_size: size of the output. Number of nodes in last layer of neural network
		@param word_dimension: Size of each of the word embeddings in the network.

		@returns model: Keras model with specified architecture.
	"""
	review_input = Input(shape=(input_size,), dtype='float32')
	dimension = input_size - other_feature
	sentence_data = Lambda(lambda x: x[:,:-other_feature])(review_input)
	other_data = Lambda(lambda x: x[:,dimension:])(review_input)

	review_encoder = Embedding(vocabSize, word_dimension, input_length=dimension, trainable=True)(sentence_data)
	bigram_conv = Conv1D(filters=10, kernel_size=2, padding='valid', activation='relu', strides=1)(review_encoder)
	bigram_pooling = GlobalMaxPooling1D()(bigram_conv)
	trigram_conv = Conv1D(filters=10, kernel_size=3, padding='valid', activation='relu', strides=1)(review_encoder)
	trigram_pooling = GlobalMaxPooling1D()(trigram_conv)
	fourgram_conv = Conv1D(filters=10, kernel_size=4, padding='valid', activation='relu', strides=1)(review_encoder)
	fourgram_pool = GlobalMaxPooling1D()(fourgram_conv)
	# fivegram_conv = Conv1D(filters=10, kernel_size=5, padding='valid', activation='relu', strides=1)(review_encoder)
	# fivegram_pool = GlobalMaxPooling1D()(fivegram_conv)
	# sevengram_conv = Conv1D(filters=10, kernel_size=7, padding='valid', activation='relu', strides=1)(review_encoder)
	# sevengram_pool = GlobalMaxPooling1D()(fivegram_conv)
	merged = concatenate([bigram_pooling, trigram_pooling, fourgram_pool, other_data], axis=1)
	joined = Dense(30, activation='relu')(merged)
	dropout = Dropout(0.2)(joined)
	out = Dense(output_size, activation='softmax')(dropout)
	model = Model(inputs=review_input, outputs=out)
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model


def main(file_path, token_name,rating_epc, recom_epc, batch_s, rating_m, recom_m):
	data = pd.read_csv(file_path, encoding='utf-8')
	data = data.replace(np.nan, '', regex=True)
	
	pd_rating, pd_recom = cleaning.get_one_hot_encoded(data['Rating']), cleaning.get_one_hot_encoded(data['Recommended IND'])
	data, other_data = cleaning.get_one_hot_encode_for_model(data, train=True)
	data = data.drop(['Rating', 'Recommended IND'], axis=1)

	pd_df_train, pd_df_test, rating_train, rating_test, recommendation_train, recommendation_test, other_data_train, \
	other_data_test = train_test_split(data, pd_rating, pd_recom, other_data,test_size=0.2, random_state=25)

	pd_df_train, vocabSize = cleaning.preprocess_data(pd_df_train, train=True, tokenizer_name=token_name)
	pd_df_test = cleaning.preprocess_data(pd_df_test, train=False, tokenizer_name=token_name)
	pd_df_train = np.hstack((pd_df_train, other_data_train))
	pd_df_test = np.hstack((pd_df_test, other_data_test))

	total_input_size = pd_df_train.shape[1]

	rating_model = get_nlp_model(vocabSize, total_input_size, total_input_size-60, rating_train.shape[1], 300)
	recommendation_model = get_nlp_model(vocabSize, total_input_size, total_input_size-60, recommendation_train.shape[1], 300)

	plot_model(rating_model, to_file='rating_model.png', show_shapes=True, show_layer_names=True)
	plot_model(recommendation_model, to_file='recommendation_mode.png', show_shapes=True, show_layer_names=True)

	# Train the rating and recommendation model using training set.
	rating_model.fit(pd_df_train, rating_train, validation_data=(pd_df_test, rating_test), epochs=rating_epc, 
					 batch_size=batch_s, verbose=1)
	recommendation_model.fit(pd_df_train, recommendation_train, validation_data=(pd_df_test, recommendation_test), 
							 epochs=recom_epc, batch_size=batch_s, verbose=1)

	# Evaluate model on the test set and print results
	rating_scores = rating_model.evaluate(pd_df_test, rating_test, verbose=1)
	print("Rating Model CNN Error: %.2f%%" % (100-rating_scores[1]*100))

	recommendation_score = recommendation_model.evaluate(pd_df_test, recommendation_test, verbose=1)
	print("Recommendation Model CNN Error: %.2f%%" % (100-recommendation_score[1]*100))

	# Save the trained model.
	rating_model.save(rating_m)
	recommendation_model.save(recom_m)


if __name__=="__main__":
	import argparse
	parser = argparse.ArgumentParser(description = 'Review Rating and Recommendation Neural Network Training Argument Parser')
	parser.add_argument('--file_path', type=str, default='data.csv' ,help='Enter the file path of the dataset')
	parser.add_argument('--tokenizer_name', type=str, default='tokenizer.pickle' ,help='Enter the tokenizer_name')
	parser.add_argument('--rating_epochs', type=int, default=50 ,help='Enter number of epochs for rating prediction model')
	parser.add_argument('--recom_epochs', type=int, default=50 ,
						help='Enter number of epochs for recommendation prediction model')
	parser.add_argument('--batch_size', type=int, default=200 ,help='Enter size of each batch while traning the netwrok')
	parser.add_argument('--rating_model', type=str, default='rating_model.h5', 
						help='Enter the name for saving the rating model')
	parser.add_argument('--recom_model', type=str, default='recommendation_model.h5', 
						help='Enter the name for saving the recommendation model')
	args = parser.parse_args()
	main(args.file_path, args.tokenizer_name, args.rating_epochs, args.recom_epochs, args.batch_size, args.rating_model, 
		 args.recom_model)