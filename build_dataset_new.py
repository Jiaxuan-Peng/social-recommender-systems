#导入i_vector和u_vector并划分训练集和测试集

import random
import pickle
import numpy as np
import pandas as pd

random.seed(1234)

workdir = 'social-recommender-systems' # change to your workdir
i_vector = pd.DataFrame(np.load('yelp/item_vector.npy'))
u_vector = pd.DataFrame(np.load('yelp/user_vector.npy'))
with open('yelp/i_vector.pkl', 'rb') as f:
	i_vector_list = pickle.load(f)
click_f1 = np.loadtxt('yelp/yelp.train.rating', dtype = np.int32)
click_f2 = np.loadtxt('yelp/yelp.val.rating', dtype = np.int32)
click_f3 = np.loadtxt('yelp/yelp.test.rating', dtype = np.int32)
click_f = np.concatenate([click_f1,click_f2,click_f3])
trust_f = np.loadtxt('yelp/yelp.links', dtype = np.int32)

click_list = []#uid, iid, label
trust_list = []

u_read_list = []
u_friend_list = []
uf_read_list = []
i_read_list = []
i_friend_list = []
if_read_list = []
i_link_list = []
user_count = 0
item_count = 0

for s in click_f:
	uid = s[0]
	iid = s[1]
	label = s[2]
	if uid > user_count:
		user_count = uid
	if iid > item_count:
		item_count = iid
	click_list.append([uid, iid, label])

pos_list = []
for i in range(len(click_list)):
	pos_list.append((click_list[i][0], click_list[i][1], click_list[i][2]))
random.shuffle(pos_list)#changes the x list in place
train_set = pos_list[:int(0.8*len(pos_list))]
test_set = pos_list[int(0.8*len(pos_list)):len(pos_list)]


with open('yelp/rating.pkl', 'wb') as f:
	pickle.dump(train_set, f, pickle.HIGHEST_PROTOCOL)#store the object data to the file
	pickle.dump(test_set, f, pickle.HIGHEST_PROTOCOL)

train_df = pd.DataFrame(train_set, columns = ['uid', 'iid', 'label'])
test_df = pd.DataFrame(test_set, columns = ['uid', 'iid', 'label'])
click_df = pd.DataFrame(click_list, columns = ['uid', 'iid', 'label'])

train_df = train_df.sort_values(axis = 0, ascending = True, by = 'uid')
for u in range(user_count+1):
	hist = train_df[train_df['uid']==u]
	#hist = hist[hist['label']>3]
	u_read = hist['iid'].unique().tolist()#the itemset of users
	print(hist['iid'])
	if u_read==[]:
		u_read_list.append([0])
	else:
		u_read_list.append(u_read)

###add the trainset and testset of i_vector or u_vector here?


train_df = train_df.sort_values(axis = 0, ascending = True, by = 'iid')
for i in range(item_count+1):
	hist = train_df[train_df['iid']==i]
	#hist = hist[hist['label']>3]
	i_read = hist['uid'].unique().tolist()#the userset of items
	if i_read==[]:
		i_read_list.append([0])
	else:
		i_read_list.append(i_read)
#print(i_read_list[1])#[1172, 143, 169, 360, 299, 2266, 1753, 813, 285, 1756, 1]

for s in trust_f:
	uid = s[0]
	fid = s[1]
	if uid > user_count or fid > user_count:
		continue
	trust_list.append([uid, fid])

trust_df = pd.DataFrame(trust_list, columns = ['uid', 'fid'])

trust_df = trust_df.sort_values(axis = 0, ascending = True, by = 'uid')
for u in range(user_count+1):
	hist = trust_df[trust_df['uid']==u]
	u_friend = hist['fid'].unique().tolist()#the friend-list of users
	if u_friend==[]:
		u_friend_list.append([0])
		uf_read_list.append([[0]])
	else:
		u_friend_list.append(u_friend)
		uf_read_f = []
		for f in u_friend:
			uf_read_f.append(u_read_list[f])
		uf_read_list.append(uf_read_f)#the itemset of friends

for i in range(item_count+1):
	if len(i_read_list[i])<=30:
		i_friend_list.append([0])
		if_read_list.append([[0]])
		i_link_list.append([0])
		continue
	i_friend = []
	for j in range(item_count+1):
		if len(i_read_list[j])<=30:
			sim_ij = 0
		else:
			sim_ij = 0
			for s in i_read_list[i]:
				sim_ij += np.sum(i_read_list[j]==s)
		i_friend.append([j, sim_ij])#similarity between two items
	i_friend_cd = sorted(i_friend, key=lambda d:d[1], reverse=True)#sorted by sim_ij
	i_friend_i = []
	i_link_i = []
	for k in range(20):
		if i_friend_cd[k][1]>5:
			i_friend_i.append(i_friend_cd[k][0])
			i_link_i.append(i_friend_cd[k][1])
	if i_friend_i==[]:
		i_friend_list.append([0])
		if_read_list.append([[0]])
		i_link_list.append([0])
	else:
		i_friend_list.append(i_friend_i)
		i_link_list.append(i_link_i)
		if_read_f = []
		for f in i_friend_i:
			if_read_f.append(i_read_list[f])
		if_read_list.append(if_read_f)

with open('yelp/social.pkl', 'wb') as f:
	pickle.dump(u_friend_list, f, pickle.HIGHEST_PROTOCOL)#the friend-list of users
	pickle.dump(u_read_list, f, pickle.HIGHEST_PROTOCOL)#the itemset of users
	pickle.dump(uf_read_list, f, pickle.HIGHEST_PROTOCOL)#the friend-list of users
	pickle.dump(i_friend_list, f, pickle.HIGHEST_PROTOCOL)#i_friend_i count
	pickle.dump(i_read_list, f, pickle.HIGHEST_PROTOCOL)#the userset of items
	pickle.dump(if_read_list, f, pickle.HIGHEST_PROTOCOL)#the friend-list of items
	pickle.dump(i_link_list, f, pickle.HIGHEST_PROTOCOL)#similarity
	pickle.dump((user_count, item_count), f, pickle.HIGHEST_PROTOCOL)


