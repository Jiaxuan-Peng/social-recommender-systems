import os
import time
import pickle
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import sys
import csv
import eval
import eval_new
from input_new import DataInput
from model_new_03 import Model

#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
random.seed(1234)
np.random.seed(1234)
tf.random.set_seed(1234)

learning_rate = 0.1
keep_prob = 0.5
lambda1 = 0.001
lambda2 = 0.001
trunc_len = 10
train_batch_size = 64
test_batch_size = 64

workdir = 'social-recommender-systems' # change to your workdir
with open('yelp/rating.pkl', 'rb') as f:
	train_set = pickle.load(f)
	test_set = pickle.load(f)
with open('yelp/social.pkl', 'rb') as f:
    u_friend_list = pickle.load(f)
    u_read_list = pickle.load(f)
    uf_read_list = pickle.load(f) 
    i_friend_list = pickle.load(f)
    i_read_list = pickle.load(f)
    if_read_list = pickle.load(f)
    i_link_list = pickle.load(f)
    user_count, item_count = pickle.load(f)
with open('yelp/item_vector.npy', 'rb') as f:
	i_vector = pd.DataFrame(np.load('yelp/item_vector.npy'))
with open('yelp/user_vector.npy', 'rb') as f:
	u_vector = pd.DataFrame(np.load('yelp/user_vector.npy'))

pos_list = []
for i in range(len(i_vector)-1):
    pos_list.append(list(i_vector.iloc[i]))
random.shuffle(pos_list)  # changes the x list in place
i_vector_train_set = pos_list[:int(0.8 * len(pos_list))]
i_vector_test_set = pos_list[int(0.8 * len(pos_list)):len(pos_list)]

pos_list = []
for i in range(len(u_vector)-1):
    pos_list.append(list(u_vector.iloc[i]))
random.shuffle(pos_list)  # changes the x list in place
u_vector_train_set = pos_list[:int(0.8 * len(pos_list))]
u_vector_test_set = pos_list[int(0.8 * len(pos_list)):len(pos_list)]

#with open('yelp/i_vector.pkl', 'rb') as f:#i_vector
#    i_vector_train_set = pickle.load(f)
#    i_vector_test_set = pickle.load(f)
#with open('yelp/u_vector.pkl', 'rb') as f:
#    u_vector_train_set = pickle.load(f)
#    u_vector_test_set = pickle.load(f)

def calc_metric(score_label_u):
	score_label_u = sorted(score_label_u, key=lambda d:d[0], reverse=True)
	precision = np.array([eval.precision_k(score_label_u, k) for k in range(1, 21)])
	ndcg = np.array([eval.ndcg_k(score_label_u, k) for k in range(1, 21)])
	auc = eval.auc(score_label_u)
	mae = eval.mae(score_label_u)
	rmse = eval.rmse(score_label_u)
	return precision, ndcg, auc, mae, rmse


def get_metric(score_label):
    Precision = np.zeros(20)
    NDCG = np.zeros(20)
    AUC = 0.
    score_df = pd.DataFrame(score_label, columns=['uid', 'score', 'label', 'iid'])
    num = 0
    score_label_all = []
    hist_all = []
    for uid, hist in score_df.groupby('uid'):
        if hist.shape[0] < 10:
            continue
        score = hist['score'].tolist()
        label = hist['label'].tolist()
        score_label_u = []
        for i in range(len(score)):
            score_label_u.append([score[i], label[i]])
            score_label_all.append([score[i], label[i]])
        hist_all.append(hist)
        precision, ndcg, auc, mae, rmse = calc_metric(score_label_u)
        Precision += precision
        NDCG += ndcg
        AUC += auc
        num += 1
    score_label_all = sorted(score_label_all, key=lambda d: d[0], reverse=True)
    GPrecision = np.array([eval.precision_k(score_label_all, k * len(score_label_all) / 100) for k in range(1, 21)])
    GAUC = eval.auc(score_label_all)
    MAE = eval.mae(score_label_all)
    RMSE = eval_new.rmse(score_label_all)
    Segregation = eval_new.Segregation(score_label)
    I_similarity = eval_new.I_similarity(hist_all)
    Consumed_diversity = eval_new.Consumed_diversity(u_read_list)
    U_homogeneity = eval_new.U_homogeneity(i_read_list)
    return Precision / num, NDCG / num, AUC / num, GPrecision, GAUC, MAE, RMSE, Segregation, I_similarity,Consumed_diversity,U_homogeneity

def _eval(sess, model):
	loss_sum = 0.
	batch = 0
	score_label = []
	for _, datainput, u_readinput, u_friendinput, uf_readinput, u_vectorinput, u_read_l, u_friend_l, uf_read_linput, u_vector_l,\
		i_readinput, i_friendinput, if_readinput, i_vectorinput, i_linkinput, i_read_l, i_friend_l, if_read_linput, i_vector_l in \
	DataInput(test_set, u_read_list, u_friend_list, uf_read_list, u_vector_test_set, i_read_list, i_friend_list, if_read_list, i_vector_test_set, \
		i_link_list, test_batch_size, trunc_len):
		score_, loss = model.eval(sess, datainput, u_readinput, u_friendinput, uf_readinput, u_vectorinput, u_read_l, u_friend_l, uf_read_linput, u_vector_l, \
	    i_readinput, i_friendinput, if_readinput, i_vectorinput, i_linkinput, i_read_l, i_friend_l, if_read_linput, i_vector_l, lambda1, lambda2)
		for i in range(len(score_)):
			score_label.append([datainput[1][i], score_[i], datainput[2][i],datainput[0][i]])#(uid,score,label,iid)
		loss_sum += loss
		batch += 1
	Segregation, RMSE,I_similarity,Consumed_diversity,U_homogeneity = get_metric(score_label)
	return loss_sum/batch, RMSE, Segregation, I_similarity, Consumed_diversity, U_homogeneity

gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
with tf.compat.v1.Session() as sess:
	model = Model(user_count, item_count)
	#model.restore(sess, 'model/DUAL_GAT.ckpt')

	Test_loss, RMSE, I_similarity, Consumed_diversity, U_homogeneity = _eval(sess, model)
	print('Test_loss: %.4f RMSE: %.4f I_similarity: %.4f Consumed_diversity: %.4f U_homogeneity: %.4f' %
	(Test_loss, RMSE, I_similarity, Consumed_diversity, U_homogeneity))

	sys.stdout.flush()
	
