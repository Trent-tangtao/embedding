import numpy as np
import scipy.sparse as sp
from keras import backend as K
from sklearn.metrics import roc_auc_score as auc_score
from sklearn.metrics import average_precision_score as ap_score
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler
from utils import generate_data, batch_data
from utils_gcn import load_citation_data, split_citation_data
from ae_LPNC import autoencoder_with_node_features


# 以'citeseer'数据集为例，讨论节点分类过程
dataset = 'citeseer'
print('\nLoading dataset {:s}...\n'.format(dataset))
adj, feats,_,_,_,_,_,_ = load_citation_data(dataset)
feats = MaxAbsScaler().fit_transform(feats).tolil()
print('\nPreparing test split...\n')
test_inds = split_citation_data(adj)
test_inds = np.vstack({tuple(row) for row in test_inds})
train = adj.copy()
train.setdiag(0.0)

test_r = test_inds[:, 0]
test_c = test_inds[:, 1]
# Collect edge labels for evaluation
# NOTE: matrix is undirected and symmetric
labels = []
labels.extend(np.squeeze(adj[test_r, test_c].toarray()))
labels.extend(np.squeeze(adj[test_c, test_r].toarray()))
# Mask test edges as missing with -1.0 values
train[test_r, test_c] = -1.0
train[test_c, test_r] = -1.0
# Impute missing edges of input adj with 0.0 for good results
adj[test_r, test_c] = 0.0
adj[test_c, test_r] = 0.0
adj.setdiag(1.0) # enforce self-connections

print('\nCompiling autoencoder model...\n')
encoder, ae = autoencoder_with_node_features(dataset, adj, feats)
adj = sp.hstack([adj, feats]).tolil()
train = sp.hstack([train, feats]).tolil()


epochs = 50
train_batch_size = 8
val_batch_size = 256

print('\nFitting autoencoder model...\n')
dummy = np.empty(shape=(adj.shape[0], 1))
y_true = dummy.copy()
mask = dummy.copy()

train_data = generate_data(adj, train, feats, y_true, mask, shuffle=True)
batch_data = batch_data(train_data, train_batch_size)
num_iters_per_train_epoch = adj.shape[0] / train_batch_size
for e in range(epochs):
    print('\nEpoch {:d}/{:d}'.format(e+1, epochs))
    print('Learning rate: {:6f}'.format(K.eval(ae.optimizer.lr)))
    curr_iter = 0
    train_loss = []
    for batch_adj, batch_train, batch_f, dummy_y, dummy_m in batch_data:
        # Each iteration/loop is a batch of train_batch_size samples
        if dataset in ['conflict', 'metabolic']:
            batch_adj = StandardScaler().fit_transform(batch_adj)
            res = ae.train_on_batch([batch_adj],
                                    [batch_train, batch_f])
        else:
            res = ae.train_on_batch([batch_adj], [batch_train])
        train_loss.append(res)
        curr_iter += 1
        if curr_iter >= num_iters_per_train_epoch:
            break
    train_loss = np.asarray(train_loss)
    train_loss = np.mean(train_loss, axis=0)
    print('Avg. training loss: {:s}'.format(str(train_loss)))
    print('\nEvaluating val set...')
    outputs, predictions = [], []
    for step in range(int(adj.shape[0] / val_batch_size) + 1):
        low = step * val_batch_size
        high = low + val_batch_size
        batch_adj = adj[low:high].toarray()
        if batch_adj.shape[0] == 0:
            break
        if dataset in ['conflict', 'metabolic']:
            batch_adj = StandardScaler().fit_transform(batch_adj)
            decoded_lp = ae.predict_on_batch([batch_adj])[0]
        else:
            decoded_lp = ae.predict_on_batch([batch_adj])
        outputs.append(decoded_lp)
    decoded_lp = np.vstack(outputs)
    predictions.extend(decoded_lp[test_r, test_c])
    predictions.extend(decoded_lp[test_c, test_r])
    print('Val AUC: {:6f}'.format(auc_score(labels, predictions)))
    print('Val AP: {:6f}'.format(ap_score(labels, predictions)))
print('\nAll done.') 

# Evaluating val set...
# Val AUC: 0.852273
# Val AP: 0.893639