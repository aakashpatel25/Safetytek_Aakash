import tensorflow as tf, json, logging

def load_graph(frozen_graph_filename):
	"""
		Loads a tensorflow graph given the stored graph file name.

		@param frozen_graph_filename: Stored tensorflow .pb file

		@return graph: Loaded tesnsorflow graph.
	"""
	with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
		graph_def = tf.GraphDef()
		graph_def.ParseFromString(f.read())

	with tf.Graph().as_default() as graph:
		tf.import_graph_def(graph_def, name="prefix")
	return graph


def predict(tf_session, x, y, text_sequences):
	"""
		Given image data runs tensorflow graph of frozen digit recognizer model
		and returns probabilty.

		@parameter image_data
		@returns json_response
	"""
	y_out = tf_session.run(y, feed_dict={
        x: text_sequences
	})
	json_data = json.dumps({'status': 200,
							'probability': y_out.tolist()})
	return json_data