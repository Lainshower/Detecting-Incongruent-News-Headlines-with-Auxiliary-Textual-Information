import tensorflow as tf
import numpy as np
import pandas as pd
from model import Model
from utils.callbacks import EarlyStopping,ModelSave

class Trainer(object):

	def __init__(self, **kwargs):
		self.model_name = kwargs.get("model_name", "DINHATI")
		self.max_tit = kwargs.get("max_tit", 29)
		self.max_sub = kwargs.get("max_sub", 114)
		self.max_body = kwargs.get("max_body", 35)
		self.max_sent = kwargs.get("max_sent", 12)
		self.max_cap = kwargs.get("max_cap", 24)
		self.drop = kwargs.get("drop", 0.1)
		self.hidden1 = kwargs.get('hidden1', 512)
		self.hidden2 = kwargs.get("hidden2", 256)
		self.batch = kwargs.get('batch', 1024)
		self.epochs = kwargs.get('epochs', 15)
		self.min_delta = kwargs.get('min_delta', 0.01)
		self.patience = kwargs.get('patience', 3)
		self.lr = kwargs.get('lr', 0.0001)
		self.beta_1 = kwargs.get('beta1', 0.9)
		self.beta_2 = kwargs.get('beta2', 0.999)
		self.epsilon = kwargs.get('epsilon', 1e-7)
		self.seed = kwargs.get('seed', 486)
		self.emb_mat = self.get_embedding()
		self.strategy = tf.distribute.MultiWorkerMirroredStrategy()
		self.loss_function = tf.keras.losses.BinaryCrossentropy(from_logits=False, reduction=tf.keras.losses.Reduction.NONE)
		self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr, beta_1=self.beta_1, beta_2=self.beta_2, epsilon=self.epsilon, amsgrad=False, name='Adam')

	def get_training_inputs(self):
		train_title = np.load("data/indexed_input/train/train_title.npy") 
		train_body = np.load("data/indexed_input/train/train_body.npy")
		train_subtitle = np.load("data/indexed_input/train/train_subtitle.npy")
		train_caption = np.load("data/indexed_input/train/train_caption.npy")
		train_labels = np.load("data/indexed_input/train/train_label.npy")
		return ({"titleInputs" : train_title, 'subtitleInputs': train_subtitle, 'bodyInputs': train_body, 'captionInputs': train_caption}, train_labels)

	def get_validation_inputs(self):
		valid_title = np.load("data/indexed_input/valid/valid_title.npy")
		valid_body = np.load("data/indexed_input/valid/valid_body.npy")
		valid_subtitle = np.load("data/indexed_input/valid/valid_subtitle.npy")
		valid_caption = np.load("data/indexed_input/valid/valid_caption.npy")
		valid_labels = np.load("data/indexed_input/valid/valid_label.npy")
		return ({"titleInputs" : valid_title, 'subtitleInputs': valid_subtitle, 'bodyInputs': valid_body, 'captionInputs': valid_caption}, valid_labels)

	def get_embedding(self):
		return np.load('data/embedding_matrix/embedding_matrix_kor.npy')
	
	def load_model(self):
		m = Model(self.model_name)
		model,_ = m.get_model(MAX_WORD_TIT=self.max_tit, MAX_WORD_SUB=self.max_sub, MAX_WORD_SENT=self.max_body, MAX_LEN_SENT=self.max_sent, MAX_WORD_CAP=self.max_cap,
		EMB_MAT=self.emb_mat, DROP=self.drop, HIDDEN1=self.hidden1, HIDDEN2=self.hidden2)
		return model

	def compute_loss(self,labels, predictions):
		replica_loss = self.loss_function(labels, predictions)
		return  replica_loss * (1. / self.strategy.num_replicas_in_sync)

	# Train Step
	def train_step(self,inputs):
		features, labels = inputs

		with tf.GradientTape() as tape:
			predictions = self.model(features, training=True)
			loss = self.compute_loss(labels, predictions)

		gradients = tape.gradient(loss, self.model.trainable_variables)
		self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
		self.train_accuracy.update_state(labels, predictions)
		return loss 

	@tf.function
	def distributed_train_step(self,dataset_inputs):
		per_replica_losses = self.strategy.run(self.train_step, args=(dataset_inputs,))
		return self.strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)

	# Validation step
	def valid_step(self,inputs):
		features, labels = inputs
		
		predictions = self.model(features, training=False)
		v_loss = self.loss_function(labels, predictions)

		self.valid_loss.update_state(v_loss)
		self.valid_accuracy.update_state(labels, predictions)

	@tf.function
	def distributed_valid_step(self,dataset_inputs):
		return self.strategy.run(self.valid_step, args=(dataset_inputs,))


	def training(self):

		# get inputs
		train_inputs = self.get_training_inputs()
		valid_inputs = self.get_validation_inputs()
		
		BUFFER_SIZE = train_inputs[0]['titleInputs'].shape[0]
		GLOBAL_BATCH_SIZE = self.batch * self.strategy.num_replicas_in_sync
		
		with self.strategy.scope():
			
			# load Dataset
			train_dataset = tf.data.Dataset.from_tensor_slices(train_inputs).shuffle(buffer_size=BUFFER_SIZE, seed=self.seed).batch(GLOBAL_BATCH_SIZE, drop_remainder=False) 
			train_dist_dataset = self.strategy.experimental_distribute_dataset(train_dataset)
			valid_dataset = tf.data.Dataset.from_tensor_slices(valid_inputs).batch(GLOBAL_BATCH_SIZE, drop_remainder=False) 
			valid_dist_dataset = self.strategy.experimental_distribute_dataset(valid_dataset)

			# load model
			self.model = self.load_model()
		
			self.valid_loss = tf.keras.metrics.Mean(name='valid_loss')
			self.train_accuracy = tf.keras.metrics.BinaryAccuracy(name='train_accuracy')
			self.valid_accuracy = tf.keras.metrics.BinaryAccuracy(name='valid_accuracy')

			tloss_seq = []
			tacc_seq = []
			vloss_seq = []
			vacc_seq = []

			print(" ****** Start Training!! ******")
			for epoch in range(self.epochs):
				
				import time
				start_epoch_time = time.time()
				print("\n ********** Epoch {} Start!!! ********** ".format(epoch+1))

				# Training
				total_loss = 0.0
				num_batches = 0
				
				for batch in train_dist_dataset:
					start_batch_time = time.time()
					batch_loss = self.distributed_train_step(batch)
					total_loss = total_loss + batch_loss
					num_batches += 1
					print("Time taken for {}th batch : {}".format(num_batches,time.time() - start_batch_time))
					print("Batch Loss : {}".format(batch_loss))
				
				train_loss = total_loss / num_batches

				tloss_seq.append(train_loss)
				tacc_seq.append(self.train_accuracy.result())

				print("********** Epoch {} Result ********** ".format(epoch+1))
				print("Training Loss : {}, Training Accuarcy : {}".format(train_loss,self.train_accuracy.result()*100))
				
				# Validation
				for batch in valid_dist_dataset:
					self.distributed_valid_step(batch)
				
				vloss_seq.append(self.valid_loss.result())
				vacc_seq.append(self.valid_accuracy.result())

				print("Validation Loss : {}, Validation Accuarcy : {}".format(self.valid_loss.result(),self.valid_accuracy.result()*100))
				print("Time taken for epoch: %.2fs" % (time.time() - start_epoch_time)) 

				save = ModelSave(vacc_seq,self.valid_accuracy.result())
				if save:
					print("Save Model with best validation accuracy")
					self.model.save_weights("src/checkpoint/bestmodel_checkpoint")
					print("Saving Finish!!!")

				stopEarly = EarlyStopping(tacc_seq, vacc_seq, min_delta=self.min_delta, patience=self.patience)
				if stopEarly:
					print("Callback_EarlyStopping signal received at epoch= %d/%d"%(epoch+1,self.epochs))
					print("Best Training Accruacy : {}".format(max(tacc_seq)*100))
					print("Best Validation Accuracy : {}".format(max(vacc_seq)*100))
					print("Terminating Training ")
					break

				self.valid_loss.reset_states()
				self.train_accuracy.reset_states()
				self.valid_accuracy.reset_states()

			print(" ****** End Training!! ******")