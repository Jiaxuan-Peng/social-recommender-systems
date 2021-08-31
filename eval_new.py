import numpy as np
# import communities as c
# from communities.algorithms import louvain_method
# from communities.visualization import louvain_animation
# import pickle
import numpy as np
from communities.algorithms import louvain_method
import copy
def rmse(score_label):
    n = 0
    error = 0
    for s in score_label:
        error += (s[1] - s[0]) ** 2
        n += 1
    return np.sqrt(error/n)

def cul_siilarity(i,j):
    tmp1=np.dot(i,j)
    tmp2=np.sqrt(np.dot(i,i))*np.sqrt(np.dot(j,j))
    return tmp1/(tmp2+0.000000001)

def I_similarity(hist_all):#the similarity of the recommendation list
    i_vector=np.load('yelp/item_vector.npy')
    similarity=0
    n=0
    for k in range(len(hist_all)):
        for i in hist_all[k]['iid'].values:
            for j in hist_all[k]['iid'].values:
                if i!=j:
                    similarity+=cul_siilarity(i_vector[i],i_vector[j])#(i_vector[i+1],i_vector[j+1])
                    n+=1
    return similarity/n
    # union=0
    # sum = 0
    # for s in i:
    #     if s in j:
    #         union = union+1#交集
    #         intersection = len(i)+len(j)-union #并集
    #         sum += float(union/intersection)
    # return sum*2/(i*(i-1))
def Consumed_diversity(u_read_list):#the mean consumed item diversity of per user
    i_vector = np.load('yelp/item_vector.npy')
    # with open('yelp/social.pkl', 'rb') as f:
    #     u_friend_list = pickle.load(f)  # for a specific user, his friend list
    #     u_read_list = pickle.load(f)
    similarity = 0
    n = 0
    for k in range(len(u_read_list)):
        if u_read_list[k]!=[0]:
            for i in u_read_list[k]:
                for j in u_read_list[k]:
                    if i != j:
                        similarity += cul_siilarity(i_vector[i-1], i_vector[j-1])
                        if similarity==np.nan:
                            print(similarity)
                        n += 1
    return 1-(similarity / n)
def U_homogeneity(i_read_list):#the mean user similarity of per item
    u_vector = np.load('yelp/user_vector.npy')
    # with open('yelp/social.pkl', 'wb') as f:
    #     pickle.dump(u_friend_list, f, pickle.HIGHEST_PROTOCOL)  # the friend-list of users
    #     pickle.dump(u_read_list, f, pickle.HIGHEST_PROTOCOL)  # the itemset of users
    #     pickle.dump(uf_read_list, f, pickle.HIGHEST_PROTOCOL)  # the friend-list of users
    #     pickle.dump(i_friend_list, f, pickle.HIGHEST_PROTOCOL)  # i_friend_i count
    #     pickle.dump(i_read_list, f, pickle.HIGHEST_PROTOCOL)  # the userset of items
    similarity = 0
    n = 0
    for k in range(len(i_read_list)):
        if i_read_list[k] != [0]:
            for i in i_read_list[k]:
                for j in i_read_list[k]:
                    if i != j:
                        similarity += cul_siilarity(u_vector[i-1], u_vector[j-1])
                        n += 1
    return (similarity / n)


def cul_adj_matrix(score_label):  # ['uid', 'iid','score']
    # max_uid = max(score_label[:, 0])
    # score_label[:, 1] = score_label[:, 1] + max_uid
    maxtrix_len = len(set(list(score_label[:, 0]))) + len(set(list(score_label[:, 1])))
    # uid_group = copy.deepcopy(score_label[:, 0])
    # iid_group = copy.deepcopy(score_label[:, 1])
    uid_group={}#建立uid到matrix_id的字典映射
    iid_group={}#建立iid到matrix_ide的字典映射
    j = 0
    for i in score_label[:, 0]:
        if i not in uid_group.keys():
            uid_group[i]=j
            j=j+1
    for i in score_label[:, 1]:
        if i not in iid_group.keys():
            iid_group[i]=j
            j=j+1
    uid_group_num = len(uid_group.keys())
    iid_group_num = len(iid_group.keys())
    adj_matrix = np.zeros((maxtrix_len, maxtrix_len))
    for i in range(len(score_label)):
        adj_matrix[int(uid_group[score_label[i,0]]), int(iid_group[score_label[i,1]])] = 1
    return adj_matrix,uid_group,iid_group
def cul_sjk(class1,class2,j,k):
    if (j in class1 and k in class1) or (j in class2 and k in class2):
        return 1
    else:
        return -1


def Segregation(score_label):
    score_label=np.array(score_label)
    segregation=0.0
    x=int(50)
    for i in range(int(len(score_label)/x)):#x组数据求邻接矩阵，最多100个点
        tmp=(np.array([score_label[i*x:(i+1)*x,0],
                                   score_label[i*x:(i+1)*x,3],
                                   score_label[i*x:(i+1)*x,1]])).T
        adj_matrix,uid_group,iid_group=cul_adj_matrix(tmp)#['uid', 'score', 'label', 'iid']
        classify_result=louvain_method(adj_matrix)#todo wait to alter
        class1 = classify_result[0][0]
        class2 = classify_result[0][1]
        max_uid = max(tmp[:,0])
        edge_num=len(score_label)
        tmp_segregation=0
        for j in range(len(adj_matrix)):
            for k in range(len(adj_matrix)):
                dj=sum(adj_matrix[j])
                dk=sum(adj_matrix[k])
                sjk=cul_sjk(class1,class2,j,k)
                tmp_segregation+=(adj_matrix[j,k]-(dj*dk)/(2*edge_num))*((sjk+1)/2)
        tmp_segregation=tmp_segregation/(2*m)
        segregation+=tmp_segregation
    return segregation





    # union=0
    # sum = 0
    # for s in u:
    #     if s in v:
    #         union = union+1#交集
    #         intersection = len(u)+len(v)-union #并集
    #         sum += float(union/intersection)
    # return sum*2/(u*(v-1))

#modularity
# adj_matrix = [...]
# communities, _ = louvain_method(adj_matrix)#louvain_method(adj_matrix : numpy.ndarray, n : int = None) -> list
# draw_communities(adj_matrix, communities)#draw_communities(adj_matrix : numpy.ndarray, communities : list, dark : bool = False, filename : str = None, seed : int = 1)

#animation: louvain_animation(adj_matrix : numpy.ndarray, frames : list, dark : bool = False, duration : int = 15, filename : str = None, dpi : int = None, seed : int = 2)
#https://cloud.tencent.com/developer/article/1801337