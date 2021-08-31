# import self as self
import tensorflow as tf
import pickle

class Model(object):

    def __init__(self, user_count, item_count):

        self.user = tf.compat.v1.placeholder(tf.compat.v1.int32, [None, ])  # [B]
        self.item = tf.compat.v1.placeholder(tf.compat.v1.int32, [None, ])  # [B]
        self.label = tf.compat.v1.placeholder(tf.compat.v1.float32, [None, ])  # [B]

        self.u_att1 = tf.compat.v1.placeholder(tf.compat.v1.int32, [None, ])  # [B]
        self.u_att2 = tf.compat.v1.placeholder(tf.compat.v1.int32, [None, ])  # [B]
        self.u_vector = tf.compat.v1.placeholder(tf.compat.v1.int32, [None, None])  # [B, F, R]
        self.u_vector_l = tf.compat.v1.placeholder(tf.compat.v1.float32, [None, ])  # [B]
        #tensorflow.python.framework.errors_impl.InvalidArgumentError: You must feed a value for placeholder tensor
        # 'Placeholder_5' with dtype int32 and shape [?,?][[{{node Placeholder_5}}]]

        #tensorflow.python.framework.errors_impl.InvalidArgumentError: You must feed a value for placeholder tensor
        # 'Placeholder_5' with dtype int32 and shape [?,?]




        self.i_att1 = tf.compat.v1.placeholder(tf.compat.v1.int32, [None, ])  # [B]
        self.i_att2 = tf.compat.v1.placeholder(tf.compat.v1.int32, [None, ])  # [B]
        self.i_vector = tf.compat.v1.placeholder(tf.compat.v1.int32, [None, None])  # [B, F, R]
        self.i_vector_l = tf.compat.v1.placeholder(tf.compat.v1.float32, [None, ])  # [B]

        self.u_read = tf.compat.v1.placeholder(tf.compat.v1. , [None, None])  # [B, R]
        self.u_read_l = tf.compat.v1.placeholder(tf.compat.v1.int32, [None, ])  # [B]
        self.u_friend = tf.compat.v1.placeholder(tf.compat.v1.int32, [None, None])  # [B, F]
        self.u_friend_l = tf.compat.v1.placeholder(tf.compat.v1.int32, [None, ])  # [B]
        self.uf_read = tf.compat.v1.placeholder(tf.compat.v1.int32, [None, None, None])  # [B, F, R]
        self.uf_read_linput = tf.compat.v1.placeholder(tf.compat.v1.int32, [None, None])  # [B, F]

        self.i_read = tf.compat.v1.placeholder(tf.compat.v1.int32, [None, None])  # [B, R]
        self.i_read_l = tf.compat.v1.placeholder(tf.compat.v1.int32, [None, ])  # [B]
        self.i_friend = tf.compat.v1.placeholder(tf.compat.v1.int32, [None, None])  # [B, R]
        self.i_friend_l = tf.compat.v1.placeholder(tf.compat.v1.int32, [None, ])  # [B]
        self.if_read = tf.compat.v1.placeholder(tf.compat.v1.int32, [None, None, None])  # [B, F, R]
        self.if_read_linput = tf.compat.v1.placeholder(tf.compat.v1.int32, [None, None])  # [B, F]
        self.i_link = tf.compat.v1.placeholder(tf.compat.v1.float32, [None, None, 1])  # [B, F, 1]

        self.learning_rate = tf.compat.v1.placeholder(tf.compat.v1.float32)
        self.training = tf.compat.v1.placeholder(tf.compat.v1.int32)
        self.rate = tf.compat.v1.placeholder(tf.compat.v1.float32)
        self.lambda1 = tf.compat.v1.placeholder(tf.compat.v1.float32)
        self.lambda2 = tf.compat.v1.placeholder(tf.compat.v1.float32)

        # --------------embedding layer-------------------

        hidden_units_u = 10  # user embedding size
        hidden_units_i = 10  # item embedding size

        user_emb_w = tf.compat.v1.get_variable("norm_user_emb_w", [user_count + 1, hidden_units_u], initializer=None)
        item_emb_w = tf.compat.v1.get_variable("norm_item_emb_w", [item_count + 1, hidden_units_i], initializer=None)
        item_b = tf.compat.v1.get_variable("norm_item_b", [item_count + 1], initializer=tf.compat.v1.constant_initializer(0.0))
        # embedding for user and item
        uid_emb = tf.compat.v1.nn.embedding_lookup(user_emb_w, self.user)
        iid_emb = tf.compat.v1.nn.embedding_lookup(item_emb_w, self.item)
        i_b = tf.compat.v1.gather(item_b, self.item)

        #metadata of user_vector
        user_b = tf.compat.v1.get_variable("norm_user_b", [user_count + 1], initializer=tf.compat.v1.constant_initializer(0.0))
        u_vector_emb = tf.compat.v1.nn.embedding_lookup(user_emb_w, self.u_vector)
        u_att1_emb = tf.compat.v1.nn.embedding_lookup(user_emb_w, self.u_att1)
        u_att2_emb = tf.compat.v1.nn.embedding_lookup(user_emb_w, self.u_att2)
        u_b = tf.compat.v1.gather(user_b, self.user)

        #metadata of item_vector
        i_vector_emb = tf.compat.v1.nn.embedding_lookup(item_emb_w, self.i_vector)
        i_att1_emb = tf.compat.v1.nn.embedding_lookup(item_emb_w, self.i_att1)
        i_att2_emb = tf.compat.v1.nn.embedding_lookup(item_emb_w, self.i_att2)

        #X/G_u User-Item: embedding for user's clicked items/the itemset of users
        ur_emb = tf.compat.v1.nn.embedding_lookup(item_emb_w, self.u_read)  # [B, R, H]
        key_masks = tf.compat.v1.sequence_mask(self.u_read_l, tf.compat.v1.shape(ur_emb)[1])  # [B, R]
        key_masks = tf.compat.v1.expand_dims(key_masks, axis=2)  # [B, R, 1]
        key_masks = tf.compat.v1.tile(key_masks, [1, 1, tf.compat.v1.shape(ur_emb)[2]])  # [B, R, H]
        key_masks = tf.compat.v1.reshape(key_masks, [-1, tf.compat.v1.shape(ur_emb)[1], tf.compat.v1.shape(ur_emb)[2]])  # [B, R, H]
        paddings = tf.compat.v1.zeros_like(ur_emb)  # [B, R, H]
        ur_emb = tf.compat.v1.where(key_masks, ur_emb, paddings)  # [B, R, H]

        #Y/M/G_i Item-User: embedding for item's clicking users/the userset of items
        ir_emb = tf.compat.v1.nn.embedding_lookup(user_emb_w, self.i_read)  # [B, R, H]
        key_masks = tf.compat.v1.sequence_mask(self.i_read_l, tf.compat.v1.shape(ir_emb)[1])  # [B, R]
        key_masks = tf.compat.v1.expand_dims(key_masks, axis=2)  # [B, R, 1]
        key_masks = tf.compat.v1.tile(key_masks, [1, 1, tf.compat.v1.shape(ir_emb)[2]])  # [B, R, H]
        key_masks = tf.compat.v1.reshape(key_masks, [-1, tf.compat.v1.shape(ir_emb)[1], tf.compat.v1.shape(ir_emb)[2]])  # [B, R, H]
        paddings = tf.compat.v1.zeros_like(ir_emb)  # [B, R, H]
        ir_emb = tf.compat.v1.where(key_masks, ir_emb, paddings)  # [B, R, H]

        #P/G_s User-User: embedding for user's friends' clicked items
        ufr_emb = tf.compat.v1.nn.embedding_lookup(item_emb_w, self.uf_read)
        key_masks = tf.compat.v1.sequence_mask(self.uf_read_linput, tf.compat.v1.shape(ufr_emb)[2])  # [B, F, R]
        key_masks = tf.compat.v1.expand_dims(key_masks, axis=3)  # [B, F, R, 1]
        key_masks = tf.compat.v1.tile(key_masks, [1, 1, 1, tf.compat.v1.shape(ufr_emb)[3]])  # [B, F, R, H]
        paddings = tf.compat.v1.zeros_like(ufr_emb)  # [B, F, R, H]
        ufr_emb = tf.compat.v1.where(key_masks, ufr_emb, paddings)  # [B, F, R, H]

        #Q Item-Item: embedding for item's related items' clicking users
        ifr_emb = tf.compat.v1.nn.embedding_lookup(item_emb_w, self.u_read)  # [B, F, R, H]
        key_masks = tf.compat.v1.sequence_mask(self.u_read_l, tf.compat.v1.shape(ifr_emb)[1])  # [B, F, R]
        key_masks = tf.compat.v1.expand_dims(key_masks, axis=2)  # [B, F, R, 1]
        key_masks = tf.compat.v1.tile(key_masks, [1, 1, tf.compat.v1.shape(ifr_emb)[2]])  # [B, F, R, H]
        paddings = tf.compat.v1.zeros_like(ifr_emb)  # [B, F, R, H]
        ifr_emb = tf.compat.v1.where(key_masks, ifr_emb, paddings)  # [B, F, R, H]

        #embedding for item's related items
        fiid_emb = tf.compat.v1.nn.embedding_lookup(item_emb_w, self.i_vector)
        key_masks = tf.compat.v1.sequence_mask(self.i_vector_l, tf.compat.v1.shape(fiid_emb)[1])  # [B, F]
        key_masks = tf.compat.v1.expand_dims(key_masks, axis=2)  # [B, F, 1]
        key_masks = tf.compat.v1.tile(key_masks, [1, 1, tf.compat.v1.shape(fiid_emb)[2]])  # [B, F, H]
        paddings = tf.compat.v1.zeros_like(fiid_emb)  # [B, F, H]
        fiid_emb = tf.compat.v1.where(key_masks, fiid_emb, paddings)  # [B, F, H]

        #O User-Opinion: embedding for user's vectors
        fuid_emb = tf.compat.v1.nn.embedding_lookup(user_emb_w, self.u_vector)
        key_masks = tf.compat.v1.sequence_mask(self.u_vector_l, tf.compat.v1.shape(fuid_emb)[1])  # [B, F]
        key_masks = tf.compat.v1.expand_dims(key_masks, axis=2)  # [B, F, 1]
        key_masks = tf.compat.v1.tile(key_masks, [1, 1, tf.compat.v1.shape(fuid_emb)[2]])  # [B, F, H]
        paddings = tf.compat.v1.zeros_like(fuid_emb)  # [B, F, H]
        fuid_emb = tf.compat.v1.where(key_masks, fuid_emb, paddings)  # [B, F, H]

        # --------------social influence-------------------

        uid_emb_exp1 = tf.compat.v1.tile(uid_emb, [1, tf.compat.v1.shape(fuid_emb)[1] + 1])
        uid_emb_exp1 = tf.compat.v1.reshape(uid_emb_exp1,
                                            [-1, tf.compat.v1.shape(fuid_emb)[1] + 1, hidden_units_u])  # [B, F, H]
        iid_emb_exp1 = tf.compat.v1.tile(iid_emb, [1, tf.compat.v1.shape(ifr_emb)[1] + 1])
        iid_emb_exp1 = tf.compat.v1.reshape(iid_emb_exp1,
                                            [-1, tf.compat.v1.shape(ifr_emb)[1] + 1, hidden_units_i])  # [B, F, H]
        uid_emb_ = tf.compat.v1.expand_dims(uid_emb, axis=1)
        iid_emb_ = tf.compat.v1.expand_dims(iid_emb, axis=1)

        # GAT1 User-Opinion: graph convolution on user's embedding for user static preference
        uid_in = tf.compat.v1.layers.dense(uid_emb_exp1, hidden_units_u, use_bias=False, name='trans_uid')
        uuid_in = tf.compat.v1.layers.dense(tf.compat.v1.concat([uid_emb_, fuid_emb], axis=1), hidden_units_u,
                                            use_bias=False, reuse=True, name='trans_uid')
        din_gat_uid = tf.compat.v1.concat([uid_in, uuid_in], axis=-1)
        d1_gat_uid = tf.compat.v1.layers.dense(din_gat_uid, 1, activation=tf.compat.v1.nn.leaky_relu, name='gat_uid')
        d1_gat_uid = tf.compat.v1.nn.dropout(d1_gat_uid, rate=self.rate)
        d1_gat_uid = tf.compat.v1.reshape(d1_gat_uid, [-1, tf.compat.v1.shape(ir_emb)[1] + 1, 1])  # [B, F, 1]
        weights_uid = tf.compat.v1.nn.softmax(d1_gat_uid, axis=1)  # [B, F, 1]
        weights_uid = tf.compat.v1.tile(weights_uid, [1, 1, hidden_units_u])  # [B, F, H]
        uid_gat = tf.compat.v1.reduce_sum(tf.compat.v1.multiply(weights_uid, uuid_in), axis=1)
        uid_gat = tf.compat.v1.reshape(uid_gat, [-1, hidden_units_u])

        # GAT2 Item-Item: graph convolution on item's embedding for item static attribute
        iid_in = tf.compat.v1.layers.dense(iid_emb_exp1, hidden_units_i, use_bias=False, name='trans_iid')
        fiid_in = tf.compat.v1.layers.dense(tf.compat.v1.concat([iid_emb_, ifr_emb], axis=1), hidden_units_i,
                                            use_bias=False, reuse=True, name='trans_iid')
        din_gat_iid = tf.compat.v1.concat([iid_in, fiid_in], axis=-1)
        d1_gat_iid = tf.compat.v1.layers.dense(din_gat_iid, 1, activation=tf.compat.v1.nn.leaky_relu, name='gat_iid')
        d1_gat_iid = tf.compat.v1.nn.dropout(d1_gat_iid, rate=self.rate)
        d1_gat_iid = tf.compat.v1.reshape(d1_gat_iid, [-1, tf.compat.v1.shape(ur_emb)[1] + 1, 1])  # [B, F, 1]
        weights_iid = tf.compat.v1.nn.softmax(d1_gat_iid, axis=1)  # [B, F, 1]
        weights_iid = tf.compat.v1.tile(weights_iid, [1, 1, hidden_units_i])  # [B, F, H]
        iid_gat = tf.compat.v1.reduce_sum(tf.compat.v1.multiply(weights_iid, fiid_in), axis=1)
        iid_gat = tf.compat.v1.reshape(iid_gat, [-1, hidden_units_i])

        # GAT3 Item-User: graph convolution on user's clicked items for user dynamic preference
        uid_emb_exp2 = tf.compat.v1.tile(uid_emb, [1, tf.compat.v1.shape(ir_emb)[1]])
        uid_emb_exp2 = tf.compat.v1.reshape(uid_emb_exp2,
                                            [-1, tf.compat.v1.shape(ir_emb)[1], hidden_units_u])  # [B, R, H]
        iid_emb_exp2 = tf.compat.v1.tile(iid_emb, [1, tf.compat.v1.shape(ur_emb)[1]])
        iid_emb_exp2 = tf.compat.v1.reshape(iid_emb_exp2,
                                            [-1, tf.compat.v1.shape(ur_emb)[1], hidden_units_i])  # [B, R, H]
        ur_emb_ = tf.compat.v1.expand_dims(ur_emb, axis=1)  # [B, 1, R, H]
        ir_emb_ = tf.compat.v1.expand_dims(ir_emb, axis=1)  # [B, 1, R, H]
        uid_emb_exp3 = tf.compat.v1.expand_dims(uid_emb, axis=1)
        uid_emb_exp3 = tf.compat.v1.expand_dims(uid_emb_exp3, axis=2)  # [B, 1, 1, H]
        uid_emb_exp3 = tf.compat.v1.tile(uid_emb_exp3,
                                         [1, tf.compat.v1.shape(ifr_emb)[1], tf.compat.v1.shape(ifr_emb)[2],
                                          1])  # [B, F, R, H]
        iid_emb_exp3 = tf.compat.v1.expand_dims(iid_emb, axis=1)
        iid_emb_exp3 = tf.compat.v1.expand_dims(iid_emb_exp3, axis=2)  # [B, 1, 1, H]
        iid_emb_exp3 = tf.compat.v1.tile(iid_emb_exp3,
                                         [1, tf.compat.v1.shape(ufr_emb)[1], tf.compat.v1.shape(ufr_emb)[2],
                                          1])  # [B, F, R, H]

        uint_in = tf.compat.v1.multiply(ur_emb, iid_emb_exp2)  # [B, R, H]
        uint_in = tf.compat.v1.reduce_max(uint_in, axis=1)  # [B, H]
        uint_in = tf.compat.v1.layers.dense(uint_in, hidden_units_i, use_bias=False, name='trans_uint')  # [B, H]
        uint_in_ = tf.compat.v1.expand_dims(uint_in, axis=1)  # [B, 1, H]
        uint_in = tf.compat.v1.tile(uint_in, [1, tf.compat.v1.shape(ir_emb)[1] + 1])
        uint_in = tf.compat.v1.reshape(uint_in, [-1, tf.compat.v1.shape(ir_emb)[1] + 1, hidden_units_i])  # [B, F, H]
        fint_in = tf.compat.v1.multiply(ir_emb, iid_emb_exp3)  # [B, F, R, H]
        fint_in = tf.compat.v1.reduce_max(fint_in, axis=2)  # [B, F, H]
        fint_in = tf.compat.v1.layers.dense(fint_in, hidden_units_i, use_bias=False, reuse=True, name='trans_uint')
        fint_in = tf.compat.v1.concat([uint_in_, fint_in], axis=1)  # [B, F, H]
        din_gat_uint = tf.compat.v1.concat([uint_in, fint_in], axis=-1)
        d1_gat_uint = tf.compat.v1.layers.dense(din_gat_uint, 1, activation=tf.compat.v1.nn.leaky_relu, name='gat_uint')
        d1_gat_uint = tf.compat.v1.nn.dropout(d1_gat_uint, rate=self.rate)
        d1_gat_uint = tf.compat.v1.reshape(d1_gat_uint, [-1, tf.compat.v1.shape(ir_emb)[1] + 1, 1])  # [B, F, 1]
        weights_uint = tf.compat.v1.nn.softmax(d1_gat_uint, axis=1)  # [B, F, 1]
        weights_uint = tf.compat.v1.tile(weights_uint, [1, 1, hidden_units_i])  # [B, F, H]
        uint_gat = tf.compat.v1.reduce_sum(tf.compat.v1.multiply(weights_uint, fint_in), axis=1)
        uint_gat = tf.compat.v1.reshape(uint_gat, [-1, hidden_units_i])

        # GAT4 User-User
        uid_emb_exp4 = tf.compat.v1.tile(uid_emb, [1, tf.compat.v1.shape(ir_emb)[1] + 1])
        uid_emb_exp4 = tf.compat.v1.reshape(uid_emb_exp4, [-1, tf.compat.v1.shape(ir_emb)[1] + 1, hidden_units_u])  # [B, F, H]
        ufid_in = tf.compat.v1.layers.dense(uid_emb_exp4, hidden_units_u, use_bias=False, name='trans_ufid')
        fuid_in = tf.compat.v1.layers.dense(tf.compat.v1.concat([uid_emb_, ir_emb], axis=1), hidden_units_u,
                                            use_bias=False, reuse=True, name='trans_ufid')
        din_gat_ufid = tf.compat.v1.concat([ufid_in, fuid_in], axis=-1)
        d1_gat_ufid = tf.compat.v1.layers.dense(din_gat_ufid, 1, activation=tf.compat.v1.nn.leaky_relu,
                                                name='gat_ufid')  # attention coefficient
        d1_gat_ufid = tf.compat.v1.nn.dropout(d1_gat_ufid, rate=self.rate)  # prevent overfit
        d1_gat_ufid = tf.compat.v1.reshape(d1_gat_ufid, [-1, tf.compat.v1.shape(ufr_emb)[1] + 1, 1])  # [B, F, 1]
        weights_ufid = tf.compat.v1.nn.softmax(d1_gat_ufid,
                                               axis=1)  # [B, F, 1] #output attention mechnism, normalization
        weights_ufid = tf.compat.v1.tile(weights_ufid, [1, 1, hidden_units_u])  # [B, F, H]
        ufid_gat = tf.compat.v1.reduce_sum(tf.compat.v1.multiply(weights_ufid, fuid_in), axis=1)
        ufid_gat = tf.compat.v1.reshape(ufid_gat, [-1, hidden_units_u])

        # GAT5 User-Item:  graph convolution on item's clicking users for item dynamic attribute
        iinf_in = tf.compat.v1.multiply(ir_emb, uid_emb_exp2)  # [B, R, H]
        iinf_in = tf.compat.v1.reduce_max(iinf_in, axis=1)  # [B, H]
        iinf_in = tf.compat.v1.layers.dense(iinf_in, hidden_units_u, use_bias=False, name='trans_iinf')  # [B, H]
        iinf_in_ = tf.compat.v1.expand_dims(iinf_in, axis=1)  # [B, 1, H]
        iinf_in = tf.compat.v1.tile(iinf_in, [1, tf.compat.v1.shape(ur_emb)[1] + 1])
        iinf_in = tf.compat.v1.reshape(iinf_in, [-1, tf.compat.v1.shape(ur_emb)[1] + 1, hidden_units_u])  # [B, F, H]
        finf_in = tf.compat.v1.multiply(ur_emb, uid_emb_exp3)  # [B, F, R, H]
        finf_in = tf.compat.v1.reduce_max(finf_in, axis=2)  # [B, F, H]
        finf_in = tf.compat.v1.layers.dense(finf_in, hidden_units_u, use_bias=False, reuse=True, name='trans_iinf')
        finf_in = tf.compat.v1.concat([iinf_in_, finf_in], axis=1)  # [B, F, H]
        din_gat_iinf = tf.compat.v1.concat([iinf_in, finf_in], axis=-1)
        d1_gat_iinf = tf.compat.v1.layers.dense(din_gat_iinf, 1, activation=tf.compat.v1.nn.leaky_relu, name='gat_iinf')
        d1_gat_iinf = tf.compat.v1.nn.dropout(d1_gat_iinf, rate=self.rate)
        d1_gat_iinf = tf.compat.v1.reshape(d1_gat_iinf, [-1, tf.compat.v1.shape(ur_emb)[1] + 1, 1])  # [B, F, 1]
        weights_iinf = tf.compat.v1.nn.softmax(d1_gat_iinf, axis=1)  # [B, F, 1]
        weights_iinf = tf.compat.v1.tile(weights_iinf, [1, 1, hidden_units_u])  # [B, F, H]
        iinf_gat = tf.compat.v1.reduce_sum(tf.compat.v1.multiply(weights_iinf, finf_in), axis=1)
        iinf_gat = tf.compat.v1.reshape(iinf_gat, [-1, hidden_units_u])

        # --------------fusion layer-------------------
        i_b_exp = tf.compat.v1.reshape(i_b, [-1, 1])
        # fusion_item_embedding = iinf_gat + uuid_gat + ufid_gat
        # fusion_user_embedding = uint_gat + iid_gat
        self.fusion_item_embedding = tf.compat.v1.concat([iinf_gat, uid_gat, ufid_gat], axis=1)
        self.fusion_user_embedding = tf.compat.v1.concat([uint_gat, iid_gat], axis=1)
        rate = tf.compat.v1.concat([self.fusion_item_embedding, self.fusion_user_embedding], axis=-1)

        # --------------output layer---------------
        self.score = i_b

        # loss function
        loss_emb_reg = tf.compat.v1.reduce_sum(tf.compat.v1.abs(i_b)) + tf.compat.v1.reduce_sum(
            tf.compat.v1.abs(iid_emb)) + tf.compat.v1.reduce_sum(tf.compat.v1.abs(uid_emb)) + tf.compat.v1.reduce_sum(
            tf.compat.v1.abs(fuid_emb))
        self.loss = tf.compat.v1.reduce_mean(tf.compat.v1.square(self.score - self.label)) + self.lambda1 * loss_emb_reg

        # Step variable
        self.global_step = tf.compat.v1.Variable(0, trainable=False, name='global_step')
        self.global_epoch_step = \
            tf.compat.v1.Variable(0, trainable=False, name='global_epoch_step')
        self.global_epoch_step_op = \
            tf.compat.v1.assign(self.global_epoch_step, self.global_epoch_step + 1)

        # optimization for loss function
        self.opt = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
        trainable_params = tf.compat.v1.trainable_variables(scope='norm')
        gradients = tf.compat.v1.gradients(self.loss, trainable_params)
        clip_gradients, _ = tf.compat.v1.clip_by_global_norm(gradients, 5 * self.learning_rate)
        self.train_op = self.opt.apply_gradients(zip(clip_gradients, trainable_params), global_step=self.global_step)

    # --------------end model---------------
    def train(self, sess, datainput, u_readinput, u_friendinput, uf_readinput, u_vectorinput, u_read_l, u_friend_l,\
          uf_read_linput, i_readinput, i_friendinput, if_readinput, i_vectorinput, i_linkinput,\
          i_read_l, i_friend_l, if_read_linput, lr, rate, lambda1, lambda2):
        loss, _ = sess.run([self.loss, self.train_op], feed_dict={
            self.item: datainput[0], self.user: datainput[1], self.label: datainput[2], \
            self.u_read: u_readinput, self.u_friend: u_friendinput, self.uf_read: uf_readinput,
            self.u_vectorinput: u_vectorinput, self.u_read_l: u_read_l, self.u_friend_l: u_friend_l,
            self.uf_read_linput: uf_read_linput, self.i_read: i_readinput, self.i_friend: i_friendinput, self.if_read: if_readinput,
            self.i_link: i_linkinput, self.i_read_l: i_read_l, self.i_friend_l: i_friend_l, self.if_read_linput: if_read_linput,
            self.i_vectorinput: i_vectorinput, self.training: 1, self.learning_rate: lr, self.rate: rate, self.lambda1: lambda1,
            self.lambda2: lambda2,
        })
        return loss

    def eval(self, sess, datainput, u_readinput, u_friendinput, uf_readinput, u_vectorinput, u_read_l, u_friend_l, uf_read_linput, u_vector_l,\
             i_readinput, i_friendinput, if_readinput, i_vectorinput, i_linkinput, i_read_l, i_friend_l, if_read_linput, i_vector_l, lambda1, lambda2):
        score, loss = sess.run([self.score, self.loss], feed_dict={
            self.item: datainput[0], self.user: datainput[1], self.label: datainput[2], \
            self.u_read: u_readinput, self.u_friend: u_friendinput, self.uf_read: uf_readinput, \
            self.u_att1: u_vectorinput[0], self.u_att2: u_vectorinput[1], self.u_vector_l: u_vector_l, \
            self.u_read_l: u_read_l, self.u_friend_l: u_friend_l, self.uf_read_linput: uf_read_linput,
            self.i_read: i_readinput, \
            self.i_friend: i_friendinput, self.if_read: if_readinput, self.i_link: i_linkinput, self.i_read_l: i_read_l, \
            self.i_friend_l: i_friend_l, self.if_read_linput: if_read_linput, \
            self.i_att1: i_vectorinput[0], self.i_att2: i_vectorinput[1], self.i_vector_l: i_vector_l, \
            self.training: 0, self.rate: 1, self.lambda1: lambda1, self.lambda2: lambda2,
        })
        return score, loss

    def save(self, sess, path):
        saver = tf.compat.v1.train.Saver()
        saver.save(sess, save_path=path)

    def restore(self, sess, path):
        saver = tf.compat.v1.train.Saver()
        saver.restore(sess, save_path=path)