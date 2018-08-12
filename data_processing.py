from keras.preprocessing.sequence import pad_sequences
import numpy as np, pandas as pd, nltk, pickle, os
from sklearn.preprocessing import LabelBinarizer
from nltk.stem.wordnet import WordNetLemmatizer
from keras.preprocessing.text import Tokenizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))
lmtzr = WordNetLemmatizer()

def tokenize(text):
	"""
		Given a text tokenize the text.

		@param text: text to be tokenized.
		@return token: list of tokenized words from the given text.
	"""
	word_tokens = word_tokenize(text)
	word_set = nltk.Text(word_tokens)
	return [w.lower() for w in word_set if w.isalpha()]


def stopword_removal(tokens):
	"""
		Given list of tokens remove stopwords.

		@param tokens: list of tokens
		@return toekns: list of tokens with removed stopwords
	"""
	filtered_sentence = [w for w in tokens if not w in stop_words]
	return filtered_sentence


def lemmatize(tokens):
	"""
		Given a list of tokens lemmatize each token.

		@param tokens: list of tokens
		@return tokens: list of lemmatized tokens.
	"""
	return [lmtzr.lemmatize(w) for w in tokens]



def preprocess_string(text):
	"""
		Given a text, tokenize, lemmatize and remove stopwords. 

		@param text: text to be processed
		@return processed_text: returns tokenized list of words that are cleaned.
	"""
	return lemmatize(stopword_removal(tokenize(text)))


def join_string(list_of_word):
	"""
		Given a list of words join the list using space.

		@param list_of_words: list of words
		@returns string: joined text.
	"""
	return " ".join(list_of_word)


def getTokenizer(tokenizer=True, tokenizer_name='tokenizer.pickle', vocabSize=0, data=None):
	"""
		Get Keras tokenizer to process textual data given vocabSize and given data.

		@param tokenizer: If new tokenizer is to be created or existing tokenizer is to be used.
		@param vocabSize: Size of the text data corpus.
		@param data: Text data on which the new tokenizer is to be fit.
	"""
	path = os.path.join('Encoder',tokenizer_name)
	if tokenizer == False:
		tokenizer = Tokenizer(num_words= vocabSize)
		tokenizer.fit_on_texts(data)
		with open(path, 'wb') as handle:
			pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
		return tokenizer
	else:
		with open(path, 'rb') as handle:
			tokenizer = pickle.load(handle)
		return tokenizer


def keras_tokenize(tokenizer, data, pad_len=6):
	"""
		Given a tokenizer and pandas data series convert all data in sequences and pad each sequences to a max lenghth.

		@param tokenizer: Keras trained tokenizer
		@param data: pandas series
		@param pad_len: max length to which data is to be padded.

		@return numpy_arr: padded sequence on the given data.
	"""
	sequences = tokenizer.texts_to_sequences(data)
	return pad_sequences(sequences, maxlen=pad_len)


def get_one_hot_encoded(pd_df):
	"""
		Given a pandas dataframe converts categorical data into one hot encoded format and returns a numpy array of labels.

		@param pd_df: pandas dataframe

		@return numpy_array: onehot encoded array.
	"""
	return pd.get_dummies(data=pd_df). values


def get_sklearn_onehot_econder(data, encoder_name, train=True):
	"""
		Given pandas dataframe and onehot encoder converts categorical feature into onehotecoder.

		@param pd_df: pandas data frame
		@param train: If the dataset is train dataset

		@return pd_df, pd_rating, encoded, pd_recom: pandas dataframe, rating labels, recommendation lables
	"""
	path = os.path.join('Encoder',encoder_name)
	if train:
		lablEnc = LabelBinarizer()
		lablEnc.fit(data)
		class_data = lablEnc.transform(data)
		with open(path, 'wb') as handle:
			pickle.dump(lablEnc, handle, protocol=pickle.HIGHEST_PROTOCOL)
	else:
		with open(path, 'rb') as handle:
			encoder = pickle.load(handle)
		class_data = encoder.transform(data)

	return class_data


def get_one_hot_encode_for_model(pd_df, train=True):
	"""
		Creates a one hot ecoder on entire dataset and saves it in pickle format.

		@param pd_df: pandas dataframe
		@param train: if the dataset is to be used for training

		@returns: cleaned pandas df and numpy feature vector.
	"""
	class_names = get_sklearn_onehot_econder(pd_df['Class Name'], 'class.pickle', train)
	div_names = get_sklearn_onehot_econder(pd_df['Division Name'], 'division.pickle', train)
	dept_names = get_sklearn_onehot_econder(pd_df['Department Name'], 'department.pickle', train)
	data = pd_df.drop(['Clothing ID', 'Division Name', 'Department Name', 'Class Name'], axis=1)
	other_features = np.hstack((class_names, div_names, dept_names))
	return data, other_features



def preprocess_df(pd_df, train=False):
	"""
		Processes the pandas data frame by removing stopwords, punctuations and lemmitizing each words. Converts
		categorical data into one hot encoded dataset.

		@param pd_df: Pandas dataframe to be cleaned.
		@param tain: If the data set is being used for trainig purpose.

		@return vocabSize, pd_df: Size of vocabulary in given text corpus and cleaned numerical features.  
	"""
	pd_df['Age'] = pd_df['Age'].astype(float)
	pd_df['Positive Feedback Count'] = pd_df['Positive Feedback Count'].astype(float)


	pd_df['Title'] = pd_df.apply(lambda row: preprocess_string(row['Title']), axis=1)
	pd_df['Review Text']=pd_df.apply(lambda row: preprocess_string(row['Review Text']), axis=1)

	### Used to determine the padding for the words. Manually determined which length for each review title and text 
	### would be good not to loose conext or pad for each words. 
	# print pd_df['Title'].map(len).value_counts().to_dict()
	# print "----------------------------"
	# print pd_df['Review Text'].map(len).value_counts().to_dict()
	
	if train:
		vocab = set()
		pd_df['Review Text'].apply(vocab.update)
		pd_df['Title'].apply(vocab.update)
		vocabSize = len(vocab)

	pd_df['Title'] = pd_df.apply(lambda row: join_string(row['Title']), axis=1)
	pd_df['Review Text'] = pd_df.apply(lambda row: join_string(row['Review Text']), axis=1)

	if train:
		return pd_df, vocabSize
	else:
		return pd_df


def preprocess_data(pd_df, train=False, tokenizer_name='tokenizer.pickle'):
	"""
		Processes the data and tokenizes the words, encodes categorical features and returns processed data.

		@param pd_df: pandas data frame
		@param train: If the dataset is training data.
		@param tokenizer_name: Either the name of the saved tokenizer or name by which tokenizer should be saved.

		@returns data, vocabSize: Cleaned numerical features and vocabulary size of the data.
	"""
	if train:
		pd_df, vocabSize = preprocess_df(pd_df, True)
		tokenizer = getTokenizer(False,  tokenizer_name, vocabSize, pd_df['Title'].append(pd_df['Review Text']))
	else:
		pd_df = preprocess_df(pd_df)
		tokenizer = getTokenizer(True, tokenizer_name)

	a_title_seq = keras_tokenize(tokenizer, pd_df['Title'], 6)
	a_text_seq = keras_tokenize(tokenizer, pd_df['Review Text'], 54)

	pd_df = pd_df.drop('Title', axis=1)
	pd_df = pd_df.drop('Review Text', axis=1)

	pd_df = pd_df.reindex_axis(sorted(pd_df.columns), axis=1)

	data = pd_df.values
	data = np.hstack((a_title_seq, a_text_seq, data))

	if train:
		return data, vocabSize
	else:
		return data
