import tensorflow as tf
import numpy as np

class Model(object):

    def __init__(self,name):
        self.model_name = name

    def DINHATI(self, MAX_WORD_TIT, MAX_WORD_SUB, MAX_WORD_SENT, MAX_LEN_SENT, MAX_WORD_CAP, EMB_MAT, DROP, HIDDEN1, HIDDEN2):

        #-----------------------------------
        #          Embedding Layer          
        #-----------------------------------

        embedding_layer = tf.keras.layers.Embedding(EMB_MAT.shape[0], EMB_MAT.shape[1], weights=[EMB_MAT], mask_zero=False, trainable=True, name='WordEmbedding')

        #-----------------------------------
        #          Headline Encoder
        #-----------------------------------

        title_Input = tf.keras.layers.Input(shape=(MAX_WORD_TIT,), name="titleInputs", dtype='float32')

        title_Embedding = embedding_layer(title_Input) 

        title = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(int(EMB_MAT.shape[1]/2), return_sequences=True),merge_mode='concat',name='title')(title_Embedding)

        subtitle_Input = tf.keras.layers.Input(shape=(MAX_WORD_SUB,), name="subtitleInputs", dtype='float32')

        subtitle_Embedding = embedding_layer(subtitle_Input) 

        subtitle = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(int(EMB_MAT.shape[1]/2),return_sequences=True),merge_mode='concat',name='subtitle')(subtitle_Embedding)
        
        subtitle = tf.keras.layers.GlobalAveragePooling1D(name='pooling_subtitle')(subtitle)
        
        subtitle  = tf.expand_dims(subtitle , axis=1)
        
        attended_title = tf.keras.layers.Attention(name='title_subtitle_attention')([title,subtitle])

        headline = tf.concat([title, attended_title, subtitle],axis=1,name="Headline")

        #-----------------------------------
        #          Body Encoder
        #-----------------------------------

        sent_Input = tf.keras.layers.Input(shape=(MAX_WORD_SENT,), name="sentInputs", dtype='float32')

        sent_Embedding = embedding_layer(sent_Input)

        sentence = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(int(EMB_MAT.shape[1]/2),dropout=DROP,return_sequences=True),merge_mode='concat',name='sent_gru')(sent_Embedding)

        sentence = tf.keras.layers.GlobalAveragePooling1D(name='body_sentence')(sentence)

        body_encoder = tf.keras.Model(inputs=[sent_Input],outputs=[sentence],name='body_encoder')

        body_Input = tf.keras.layers.Input(shape=(MAX_LEN_SENT,MAX_WORD_SENT), name='bodyInputs' ,dtype='float32')

        body_Embedding = tf.keras.layers.TimeDistributed(body_encoder, name='sentenceEncoding')(body_Input)

        caption_Input = tf.keras.layers.Input(shape=(MAX_WORD_CAP,), name="captionInputs", dtype='float32')

        caption_Embedding = embedding_layer(caption_Input) 

        caption = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(int(EMB_MAT.shape[1]/2), dropout=DROP,return_sequences=True),merge_mode='concat',name='caption')(caption_Embedding)

        caption = tf.keras.layers.GlobalAveragePooling1D(name='pooling_caption')(caption)

        caption  = tf.expand_dims(caption , axis=1)

        body = tf.concat([body_Embedding,caption],axis=1)

        body = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(int(EMB_MAT.shape[1]/2), dropout=0.3, return_sequences=True),merge_mode='concat',name='body')(body)

        #-----------------------------------
        #          NEWS Encoder
        #-----------------------------------

        body_attended_headline = tf.keras.layers.Attention(name='headline_body_attention')([headline,body])

        news = tf.concat([headline, body_attended_headline],axis=1,name='concat_headlines')

        #-----------------------------------
        #          Classifier
        #-----------------------------------

        news = tf.keras.layers.Flatten()(news)

        news = tf.keras.layers.Dropout(0.1)(news)

        dense01 = tf.keras.layers.Dense(HIDDEN1,kernel_initializer='glorot_uniform',activation="relu",name='FFN01')(news)

        dense01 = tf.keras.layers.Dropout(DROP)(dense01)

        dense02 = tf.keras.layers.Dense(HIDDEN2,kernel_initializer='glorot_uniform',activation="relu",name='FFN02')(dense01)

        dense02 = tf.keras.layers.Dropout(DROP)(dense02)

        output = tf.keras.layers.Dense(1,activation="sigmoid")(dense02)

        cap_bod_model = tf.keras.Model(inputs=[title_Input, subtitle_Input, body_Input, caption_Input], outputs=[output],name="Model")

        return cap_bod_model, body_encoder  
    
    def get_model(self, MAX_WORD_TIT, MAX_WORD_SUB, MAX_WORD_SENT, MAX_LEN_SENT, MAX_WORD_CAP, EMB_MAT, DROP, HIDDEN1, HIDDEN2):
        if self.model_name == "DINHATI":
            return self.DINHATI(MAX_WORD_TIT, MAX_WORD_SUB, MAX_WORD_SENT, MAX_LEN_SENT, MAX_WORD_CAP, EMB_MAT, DROP, HIDDEN1, HIDDEN2)
        else:
            raise ValueError ("Please check Model Name")