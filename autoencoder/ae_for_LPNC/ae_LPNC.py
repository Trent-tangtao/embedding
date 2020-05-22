import numpy as np
import scipy.sparse as sp
import tensorflow as tf
from keras import optimizers
from keras import backend as K
from keras.models import Model
from keras.layers import Input,Dense,Dropout,add,Lambda
from layers import DenseTied

# 文章中提到利用MVN提高LP的准确率,以行来计算，因为输入是邻接矩阵
def mvn(tensor):
    epsilon = 1e-6
    mean = K.mean(tensor, axis=1, keepdims=True)
    std = K.std(tensor, axis=1, keepdims=True)
    mvn = (tensor-mean) / (std + epsilon)
    return mvn


# 交叉熵loss
def ce(y_true,y_pred):
    return K.mean(K.binary_crossentropy(target=y_true,output=y_pred,from_logits=True,),axis=-1)


# mbce loss
def mbce(y_true,y_pred):
    mask = K.cast(K.not_equal(y_true,-1.0),dtype=np.float32)
    num_examples = K.sum(mask, axis=1)
    pos = K.cast(K.equal(y_true, 1.0), dtype=np.float32)
    num_pos = K.sum(pos, axis=None)
    neg = K.cast(K.equal(y_true, 0.0), dtype=np.float32)
    num_neg = K.sum(neg, axis=None)
    pos_ratio = 1.0 - num_pos/num_neg
    mbce = mask * tf.nn.weighted_cross_entropy_with_logits(targets=y_true,logits=y_pred,pos_weight=pos_ratio)
    mbce = K.sum(mbce,axis=1)/num_examples
    return K.mean(mbce,axis=-1)


# 类别loss:softmax cross-entropy loss with masking
def masked_categorical_crossentropy(y_true,y_pred):
    mask = y_true[:, -1]
    mask = K.cast(mask,dtype=np.float32)
    y_true = y_true[:, :-1]    # 维度有点晕了
    loss = K.categorical_crossentropy(target=y_true,output=y_pred,from_logits=True)
    loss *= mask
    return K.mean(loss,axis=-1)


# 自编码器
def autoencoder(dataset,adj,weights=None):
    h,w = adj.shape
    sparse_net = dataset in ['conflict', 'metabolic', 'protein']

    kwargs = dict(
        use_bias=True,
        kernel_initializer='glorot_normal',
        kernel_regularizer=None,
        bias_initializer='zeros',
        bias_regularizer=None,
        trainable=True,
    )
    # 输入
    data = Input(shape=(w,),dtype=np.float32, name='data')
    # dropout
    if sparse_net:
        # 对于 conflict, metabolic, protein networks
        noisy_data = Dropout(rate=0.2, name='drop0')(data)
    else:
        # 对于 citation, blogcatalog, arxiv-grqc, and powergrid networks
        noisy_data = Dropout(rate=0.5, name='drop0')(data)

    # 第一层encode
    encoded = Dense(256, activation='relu', name='encoded1', **kwargs)(noisy_data)
    if sparse_net:
        encoded = Lambda(mvn, name='mvn1')(encoded)
        encoded = Dropout(rate=0.5, name='drop1')(encoded)

    # 第二层encode
    encoded = Dense(128, activation='relu',name='encoded2', **kwargs)(encoded)
    if sparse_net:
        encoded = Lambda(mvn, name='mvn2')(encoded)
        encoded = Dropout(rate=0.5, name='drop2')(encoded)

    # 提取两个encode层
    encoder = Model([data], encoded)
    encoded1 = encoder.get_layer('encoded1')
    encoded2 = encoder.get_layer('encoded2')

    # 第一层decode
    # 因为需要参数共享，所以这里面的层需要自己实现
    decoded = DenseTied(256, tie_to=encoded2, transpose=True,activation='relu', name='decoded2')(encoded)
    if sparse_net:
        decoded = Lambda(mvn, name='mvn3')(decoded)
        decoded = Dropout(rate=0.5, name='drop3')(decoded)

    # 第二层decode
    decoded = DenseTied(w, tie_to=encoded1, transpose=True, activation='linear', name='decoded1')(decoded)

    # 编译
    adam = optimizers.Adam(lr=0.001, decay=0.0)
    autoencoder = Model(inputs=[data], outputs=[decoded])
    autoencoder.compile(optimizer=adam, loss=mbce)

    if weights is not None:
        autoencoder.load_weights(weights)

    return encoder, autoencoder


# 能够处理节点特征的自编码器
def autoencoder_with_node_features(dataset, adj, feats, weights=None):
    aug_adj = sp.hstack([adj, feats])
    h, w = aug_adj.shape

    kwargs = dict(
        use_bias=True,
        kernel_initializer='glorot_normal',
        kernel_regularizer=None,
        bias_initializer='zeros',
        bias_regularizer=None,
        trainable=True,
    )
    # 输入层
    data = Input(shape=(w,), dtype=np.float32, name='data')
    # dropout
    if dataset in ['protein', 'cora', 'citeseer', 'pubmed']:
        # dropout 0.5 is needed for protein and citation (large nets)
        noisy_data = Dropout(rate=0.5, name='drop1')(data)
    else:
        # dropout 0.2 is needed for conflict and metabolic (small nets)
        noisy_data = Dropout(rate=0.2, name='drop1')(data)

    # 第一层encode
    encoded = Dense(256, activation='relu',name='encoded1', **kwargs)(noisy_data)
    encoded = Lambda(mvn, name='mvn1')(encoded)

    # 第二层encode
    encoded = Dense(128, activation='relu',  name='encoded2', **kwargs)(encoded)
    encoded = Lambda(mvn, name='mvn2')(encoded)
    encoded = Dropout(rate=0.5, name='drop2')(encoded)

    # encoder模型
    encoder = Model([data], encoded)
    encoded1 = encoder.get_layer('encoded1')
    encoded2 = encoder.get_layer('encoded2')

    # 第一层decode
    decoded = DenseTied(256, tie_to=encoded2, transpose=True, activation='relu', name='decoded2')(encoded)
    decoded = Lambda(mvn, name='mvn3')(decoded)

    # 第二层decode
    decoded = DenseTied(w, tie_to=encoded1, transpose=True, activation='linear', name='decoded1')(decoded)

    # c编译
    adam = optimizers.Adam(lr=0.001, decay=0.0)
    if dataset in ['metabolic', 'conflict']:
        # 如果节点特征是真实数值的， 需要分开计算调整，loss分开
        decoded_feats = Lambda(lambda x: x[:, adj.shape[1]:], name='decoded_feats')(decoded)
        decoded = Lambda(lambda x: x[:, :adj.shape[1]],   name='decoded')(decoded)
        # 疑问 autoencoder = Model(inputs=[data,feats], outputs=[decoded, decoded_feats] )
        autoencoder = Model(inputs=[data], outputs=[decoded, decoded_feats])
        autoencoder.compile(optimizer=adam,  loss={'decoded': mbce, 'decoded_feats': ce},
                             loss_weights={'decoded': 1.0, 'decoded_feats': 1.0})
    else:
        # 节点特征是二值化的，直接合并计算
        autoencoder = Model(inputs=[data], outputs=decoded)
        autoencoder.compile(optimizer=adam, loss=mbce)

    if weights is not None:
        autoencoder.load_weights(weights)

    return encoder, autoencoder


# 多任务的编码器
def autoencoder_multitask(dataset, adj, feats, labels, weights=None):
    adj = sp.hstack([adj, feats])
    h, w = adj.shape

    kwargs = dict(
        use_bias=True,
        kernel_initializer='glorot_normal',
        kernel_regularizer=None,
        bias_initializer='zeros',
        bias_regularizer=None,
        trainable=True,
    )
    # 输入层
    data = Input(shape=(w,), dtype=np.float32, name='data')

    # 第一层encode
    encoded = Dense(256, activation='relu', name='encoded1', **kwargs)(data)
    # 第二层encode
    encoded = Dense(128, activation='relu',  name='encoded2', **kwargs)(encoded)
    # dropout
    if dataset == 'pubmed':
        encoded = Dropout(rate=0.5, name='drop')(encoded)
    else:
        encoded = Dropout(rate=0.8, name='drop')(encoded)

    # encoder模型
    encoder = Model([data], encoded)
    encoded1 = encoder.get_layer('encoded1')
    encoded2 = encoder.get_layer('encoded2')

    # 第一层decode
    decoded = DenseTied(256, tie_to=encoded2, transpose=True, activation='relu', name='decoded2')(encoded)

    # 节点分类
    feat_data = Input(shape=(feats.shape[1],))
    pred1 = Dense(labels.shape[1], activation='linear')(feat_data)
    pred2 = Dense(labels.shape[1], activation='linear')(decoded)
    prediction = add([pred1, pred2], name='prediction')

    # 第二层decode
    decoded = DenseTied(w, tie_to=encoded1, transpose=True, activation='linear', name='decoded1')(decoded)

    # 编译
    adam = optimizers.Adam(lr=0.001, decay=0.0)
    autoencoder = Model(inputs=[data, feat_data], outputs=[decoded, prediction])
    autoencoder.compile(optimizer=adam, loss={'decoded1': mbce,'prediction': masked_categorical_crossentropy},
         # 疑问 loss_weights={'decoded': 1.0, 'prediction': 1.0}
        loss_weights = {'decoded1': 1.0, 'prediction': 1.0}
    )

    if weights is not None:
        autoencoder.load_weights(weights)

    return encoder, autoencoder




