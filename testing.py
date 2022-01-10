import tensorflow as tf
import numpy as np
from model import Model

def get_test_inputs():
	test_title = np.load('data/indexed_input/test/test_title.npy')
	test_body = np.load('data/indexed_input/test/test_body.npy')
	test_subtitle = np.load("data/indexed_input/test/test_subtitle.npy")
	test_caption = np.load("data/indexed_input/test/test_caption.npy")
	test_labels = np.load('data/indexed_input/test/test_label.npy')
	return ({"titleInputs" : test_title, 'subtitleInputs': test_subtitle, 'bodyInputs': test_body, 'captionInputs': test_caption}, test_labels)

def load_model(model_name='DINHATI', max_tit=29, max_sub=114, max_body=35, max_sent=12, max_cap=24, emb_mat=None, drop=0.1, hidden1=512, hidden2=256):
	m = Model(model_name)
	model,_ = m.get_model(MAX_WORD_TIT=max_tit, MAX_WORD_SUB=max_sub, MAX_WORD_SENT=max_body, MAX_LEN_SENT=max_sent, MAX_WORD_CAP=max_cap, EMB_MAT=emb_mat, DROP=drop, HIDDEN1=hidden1, HIDDEN2=hidden2)
	model.load_weights("data/src/checkpoint/bestmodel_checkpoint")
	return model

def get_embedding():
	return np.load('data/embedding_matrix/embedding_matrix_kor.npy')

def test():
	test_inputs = get_test_inputs()
	embedding_matrix = get_embedding()
	model = load_model(model_name='DINHATI', max_tit=29, max_sub=114, max_body=35, max_sent=12, max_cap=24, emb_mat=embedding_matrix, drop=0.1, hidden1=512, hidden2=256)

	test_dataset = tf.data.Dataset.from_tensor_slices(test_inputs).batch(1024, drop_remainder=False) 
	test_accuracy = tf.keras.metrics.BinaryAccuracy(name='test_accuracy')

	# Testing
	for x in test_dataset :
		features, labels = x
		predictions = model(features, training=False)
		test_accuracy.update_state(labels, predictions)

	print("****** Testing Result ******")
	print("Test Accuarcy : {}".format(test_accuracy.result()*100))
	print("****** Testing Done!! ****** ")
	test_accuracy.reset_states()