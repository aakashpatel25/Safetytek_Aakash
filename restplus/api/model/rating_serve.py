from restplus.api.model import modelconfig, utility
from keras.utils import np_utils
import tensorflow as tf, json, logging

def serve_rating_model(text_sequences):
	"""
		Serve rating model.

		@param text_sequences: text feature text

		@return json_text: probabilities of response.
	"""
	return utility.predict(persistent_sess, x, y, text_sequences)


logger = logging.getLogger(__name__)
logger.info('Loading frozen model graph from {}'.format(modelconfig.FROZEN_MODEL_FILE_NAME_RATING))
graph = utility.load_graph(modelconfig.FROZEN_MODEL_FILE_NAME_RATING)
x = graph.get_tensor_by_name('prefix/input_1:0')
y = graph.get_tensor_by_name('prefix/dense_2/Softmax:0')

persistent_sess = tf.Session(graph=graph)
logger.info('Finished loading frozen graph and starting tensorflow session')
