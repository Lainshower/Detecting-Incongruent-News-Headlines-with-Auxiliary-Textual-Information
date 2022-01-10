import pandas as pd
import numpy as np
import os
import re
import copy
import pickle
import itertools
import argparse
from konlpy.tag import Mecab
from gensim import models
from nltk import sent_tokenize
from tensorflow.keras.preprocessing import sequence 
from sklearn.model_selection import train_test_split

# Replace NAN to ""
def delete(text):
    if type(text) == float : 
        text = ""
    return text

# creating word2index
def create_word2index(data,tokenizer,number,eng,chi):
	word_for_tokenize = pd.concat([data["title"],data["subtitle"],data["body"],data["caption"]])
	token_list = word_for_tokenize.apply(lambda row: tokenizer.pos(row))
	token_list = token_list.reset_index(drop=True)
	token_pos_list = list(itertools.chain(*list(token_list.values)))
	token_set = set(token_pos_list)

	print("**********Creating word2index**********")
	word2index = {"<OOV>":1,"<숫자>":2,"<영어>":3,'<한자>':4} # 1 for OOV, 2 for number, 3 for english, 4 for chinese
	index = 5
	not_sh_chi =[]
	for token,pos in token_set :
		if number.match(token) is not None: # we treat number as <숫자> token
			continue
		elif eng.match(token) is not None: # 영어는 pos tag가 'SL'(외국어)이 아닌 경우 word2index에 추가, 그러나 하나도 없음
			if token not in word2index and pos != 'SL':
				word2index[token] = index
				index = index + 1
				continue
		elif chi.match(token) is not None : # 한자는 pos tag가 'SH'(외국어)이 아닌 경우 word2index에 추가
			if token not in word2index and pos != 'SH':
				not_sh_chi.append((token,pos))
				word2index[token] = index
				index = index + 1
				continue
		elif pos in ['SF', 'SE', 'SSO', 'SSC', 'SC', 'SY']: # 특수기호는 같은 token이라도 pos tag가 다를 수 잇으므로 이를 반영해 word2_index를 만듦
			temp  = f'<{token}_{pos}>'
			if temp not in word2index:
				word2index[temp] = index
				index = index + 1
			continue
		else :
			if token not in word2index:
				word2index[token] = index
				index = index + 1
				continue
			else :
				pass
	print("**********DONE!!!**********")

	return word2index

# creating embedding matrix
def create_embedding_matrix(word2index,fasttext):
	print ("**********Creating word embedding matrix**********")
	MAX_VOC = len(word2index)+1 # +1 for padding
	print("MAX_VOC : ",MAX_VOC)
	EMBEDDING_DIM = 300
	embedding_matrix = np.zeros(shape=(MAX_VOC, EMBEDDING_DIM), dtype='float32') # zero vector (index=0)
	for key in tqdm(word2index):
		if key[0]=="<":
			embedding_matrix[word2index[key]] = np.random.uniform(-0.25, 0.25, EMBEDDING_DIM) # Random initialization for special tokens
		else:
			embedding_matrix[word2index[key]] = fasttext.wv.word_vec(key)
	print("**********DONE!!!**********")
	print()
	print("**********Embedding Matrix**********")
	print("Embedding matrix shape : {}".format(embedding_matrix.shape))

	return embedding_matrix

# tokenize
def tokenize(data,word2index,tokenizer,number,eng,chi):
	# Sent tokenize body and define body tokenizer
	data["body"] = data["body"].apply(sent_tokenize)
	sent_tokenizer = lambda x : list(map(tokenizer.pos,x))

	# POS tokenize data
	data["title"]=data["title"].apply(tokenizer.pos)
	data["subtitle"]=data["subtitle"].apply(tokenizer.pos)
	data["body"]=data["body"].apply(sent_tokenizer)
	data["caption"]=data["caption"].apply(tokenizer.pos)


	for i in range(data.shape[0]):
		for index,(token,pos) in enumerate(data["title"][i]):
			if number.match(token) is not None:
				data["title"][i][index] = ("<숫자>","special_token")
			if eng.match(token) is not None:
				data["title"][i][index] = ("<영어>","special_token")
			if chi.match(token) is not None: 
				if pos == 'SH':
					data["title"][i][index] = ("<한자>","special_token")
				if pos in ['SF', 'SE', 'SSO', 'SSC', 'SC', 'SY']: 
					temp  = f'<{token}_{pos}>'
					data["title"][i][index] = (temp,"special_token")
		for index,(token,pos) in enumerate(data["subtitle"][i]):
			if number.match(token) is not None:
				data["subtitle"][i][index] = ("<숫자>","special_token")
			if eng.match(token) is not None:
				data["subtitle"][i][index] = ("<영어>","special_token")
			if chi.match(token) is not None: 
				if pos == 'SH':
					data["subtitle"][i][index] = ("<한자>","special_token")
				if pos in ['SF', 'SE', 'SSO', 'SSC', 'SC', 'SY']: 
					temp  = f'<{token}_{pos}>'
					data["subtitle"][i][index] = (temp,"special_token")
		for index,sent in enumerate(data["body"][i]):
			for idx,(token,pos) in enumerate(data["body"][i][index]):
				if number.match(token) is not None:
					data["body"][i][index][idx] = ("<숫자>","special_token")
				if eng.match(token) is not None:
					data["body"][i][index][idx] = ("<영어>","special_token")
				if chi.match(token) is not None: 
					if pos == 'SH':
						data["body"][i][index][idx] = ("<한자>","special_token")
				if pos in ['SF', 'SE', 'SSO', 'SSC', 'SC', 'SY']: 
					temp  = f'<{token}_{pos}>'
					data["body"][i][index][idx] = (temp,"special_token")
		for index,(token,pos) in enumerate(data["caption"][i]):
			if number.match(token) is not None:
				data["caption"][i][index] = ("<숫자>","special_token")
			if eng.match(token) is not None:
				data["caption"][i][index] = ("<영어>","special_token")
			if chi.match(token) is not None: 
				if pos == 'SH':
					data["caption"][i][index] = ("<한자>","special_token")
			if pos in ['SF', 'SE', 'SSO', 'SSC', 'SC', 'SY']: 
				temp  = f'<{token}_{pos}>'
				data["caption"][i][index] = (temp,"special_token")		
	return data

def find_index(tup):
	global word2index
	if tup[0] in word2index:
		return word2index[tup[0]]
	else :
		return 1

if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	## Required parameters
	parser.add_argument("--path", required=True, type=str)
	parser.add_argument("--data_set", required=True, type=str)	
	parser.add_argument("--w2i", action="store_true")
	parser.add_argument("--emb", action="store_true")
	parser.add_argument('--max_tit', type=int, default=29)
	parser.add_argument('--max_sub', type=int, default=114)
	parser.add_argument('--max_body', type=int, default=35)
	parser.add_argument('--max_sent', type=int, default=12)
	parser.add_argument('--max_cap', type=int, default=24)
	parser.add_argument('--test_size', type=float, default=0.2)
	parser.add_argument('--valid_size', type=float, default=0.1)
	args = parser.parse_args()

	data_path = os.path.join(args.path,args.data_set)
	data = pd.read_csv(data_path)
	data = data[['title','subtitle','body','caption','category','classcode','label']]

	data["subtitle"]= data["subtitle"].apply(delete)
	data["caption"] = data["caption"].apply(delete)

	# load mecab tokenizer
	tokenizer = Mecab()

	# declare regular expression variables
	number = re.compile('[0-9]+')
	eng = re.compile('[a-zA-Z]+')
	chi  = re.compile('[一-龥]+')

	# create word2index
	if args.wi2:
		word2index = create_word2index(data,tokenizer,number,eng,chi)
		word2index_path = ' '
		with open(word2index_path,'wb') as fw:
			pickle.dump(word2index, fw)
	else:
		with open(os.path.join(args.path,'word2index','word2index.pickle'),'rb') as fr:
    			word2index = pickle.load(fr)

	# create embedding matrix
	if args.emb:
		fasttext_path = ''
		fasttext_kor = models.fasttext.load_facebook_model(fasttext_path)
		embedding_matrix = create_embedding_matrix(word2index,fasttext_kor)
		np.save(os.path.join(args.path,'embedding_matrix','embedding_matrix_kor'), embedding_matrix)
	else:
		pass
	
	# tokenize data
	data = tokenize(data,word2index,tokenizer,number,eng,chi)

	data['body'] = data['body'].apply(itertools.chain)

	# convert token to index
	indexing = lambda sent : np.array(list(map(find_index,sent)))
	body_indexing = lambda sent_list : np.array(list(map(indexing,sent_list)))
	data["title"] = data["title"].apply(indexing)
	data["subtitle"] = data["subtitle"].apply(indexing)
	data["body"] = data["body"].apply(body_indexing)
	data["caption"] = data["caption"].apply(indexing)

	# pad max_len sentences in body
	data["body"] = data["body"].apply(lambda row : sequence.pad_sequences(row,maxlen=args.max_body,padding='post', truncating='post'))	
	
	# train, dev, test split
	x_train, x_test, y_train, y_test = train_test_split(data[["title","subtitle","body","caption",'category']], data[['category','label']], test_size=args.test_size, shuffle=True, stratify=data[['category',"label"]], random_state=486)
	x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=args.valid_size, stratify=y_train[['category',"label"]], random_state=2018)
	
	# truncating & padding
	train_title = sequence.pad_sequences(x_train["title"],maxlen=args.max_tit,padding='post', truncating='post')
	train_subtitle = sequence.pad_sequences(x_train["subtitle"],maxlen=args.max_sub,padding='post', truncating='post')
	train_body= sequence.pad_sequences(x_train["body"],maxlen=args.max_sent,padding='post', truncating='post')
	train_caption = sequence.pad_sequences(x_train["caption"],maxlen=args.max_cap,padding='post', truncating='post')
	valid_title = sequence.pad_sequences(x_valid["title"],maxlen=args.max_tit,padding='post', truncating='post')
	valid_subtitle = sequence.pad_sequences(x_valid["subtitle"],maxlen=args.max_sub,padding='post', truncating='post')
	valid_body= sequence.pad_sequences(x_valid["body"],maxlen=args.max_sent,padding='post', truncating='post')
	valid_caption = sequence.pad_sequences(x_valid["caption"],maxlen=args.max_cap,padding='post', truncating='post')
	test_title = sequence.pad_sequences(x_test["title"],maxlen=args.max_tit,padding='post', truncating='post')
	test_subtitle = sequence.pad_sequences(x_test["subtitle"],maxlen=args.max_sub,padding='post', truncating='post')
	test_body= sequence.pad_sequences(x_test["body"],maxlen=args.max_sent,padding='post', truncating='post')
	test_caption = sequence.pad_sequences(x_test["caption"],maxlen=args.max_cap,padding='post', truncating='post')
	y_train = y_train['label'].values
	y_valid = y_valid['label'].values
	y_test = y_test['label'].values
	
	# save indexed inputs
	index_inputs_path = os.path.join(args.path,'indexed_input')
	np.save(index_inputs_path+'/train/train_title', train_title)
	np.save(index_inputs_path+'/train/train_subtitle', train_subtitle)
	np.save(index_inputs_path+'/train/train_body', train_body)
	np.save(index_inputs_path+'/train/train_caption', train_caption)
	np.save(index_inputs_path+'/train/valid_title', valid_title)
	np.save(index_inputs_path+'/train/valid_subtitle', valid_subtitle)
	np.save(index_inputs_path+'/train/valid_body', valid_body)
	np.save(index_inputs_path+'/train/valid_caption', valid_caption)
	np.save(index_inputs_path+'/test/test_title', test_title)
	np.save(index_inputs_path+'/test/test_subtitle', test_subtitle)
	np.save(index_inputs_path+'/test/test_body', test_body)
	np.save(index_inputs_path+'/test/test_caption', test_caption)
	np.save(index_inputs_path+'/train/train_label',y_train)
	np.save(index_inputs_path+'/train/valid_label',y_valid)
	np.save(index_inputs_path+'/test/test_label',y_test)