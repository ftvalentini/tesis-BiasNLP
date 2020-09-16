import numpy as np
import pandas as pd
import os, datetime, pickle
import scipy.sparse

from scripts.utils.corpora import load_vocab
from metrics.gradient import \
                        bias_rnd, gradient_bias_w, gradient_wi_x, bias_gradient


#%% Corpus parameters
VOCAB_FILE = "embeddings/vocab-C3-V20.txt" # enwiki = C3
VECTORS_FILE = "embeddings/full_vectors-C3-V20-W8-D1-D100-R0.05-E150-S1.pkl"
COOC_FILE = "embeddings/cooc-C3-V20-W8-D1.npz"

#%% Load data
print("Loading data...\n")
# vocab dicts
str2idx, idx2str, str2count = load_vocab(VOCAB_FILE)
# GloVe word embeddings
with open(VECTORS_FILE, 'rb') as f:
    W, b_w, U, b_u = pickle.load(f)
# Cooc matrix
X = scipy.sparse.load_npz(COOC_FILE)

#%% PARAMS
word_context = "cocaine"
words_target_a = ['ecuador']# ,'him','his','himself']
words_target_b = ['europe']#,'her','hers','herself']
# words indices
idx_c = str2idx[word_context]
idx_a = sorted([str2idx[w] for w in words_target_a])
idx_b = sorted([str2idx[w] for w in words_target_b])

#%% Gradient Bias-W

# ### test: subir por la direccion del gradiente aumenta el bias
# grad_b_w = gradient_bias_w(W, idx_c, idx_a, idx_b)
# W_tmp = W.copy()
# biases_grad = list()
# # A. actualizaciones con gradiente:
# for i in range(10):
#     W_tmp += grad_b_w * 0.1
#     biases_grad.append(bias_rnd(W_tmp, idx_c, idx_a, idx_b))
# biases_grad
# # B. actualizaciones con pesos del gradiente permutados ("random"):
# index = np.arange(W.shape[1])
# np.random.shuffle(index)
# grad_shuffled = grad_b_w[:,index].copy()
# W_tmp = W.copy()
# biases_shuffled = list()
# for i in range(10):
#     W_tmp += grad_shuffled * 0.1
#     biases_shuffled.append(bias_rnd(W_tmp, idx_c, idx_a, idx_b))
# biases_shuffled
# # avanzar por el gradiente hace aumentar y "Mucho"
# # avanzar por random hacer aumentar/caer monotonamente pero "poco"
# # (es monotono porque es una funcion lineal)
# ###

# ### chequeo bias vs resultados previos
# tmp = pd.read_csv("results/pkl/garg_byword_MALE_SHORT-FEMALE_SHORT.csv")
# tmp.loc[tmp['word'] == "girlfriend"]
# bias_0 = bias_rnd(W, idx_c, idx_a, idx_b)
# W_norm = W / np.linalg.norm(W, axis=1)[:,np.newaxis] # normalize by l2 norm
# bias_1 = bias_rnd(W_norm, idx_c, idx_a, idx_b)
# # coinciden cuando se normaliza la matriz por l2 por vector
# # entiendo q no hace falta esto para bias gradient
# # (no se comparan bias entre palabras)
# ###

# TODO: revisar dimension de gradient

def Li(Wi, Yi, Xi, b_wi, Uj, b_uj):
    """Pointwise loss for word i
    """
    diff = np.dot(Uj, Wi) + b_uj + b_wi - np.log(Xi - Yi)
    return sum(f_glove(Xi - Yi) * (diff**2))

f_loss = lambda w, y: Li(w, y, Xi, b_wi, Uj, b_uj)
grad_w_y = jacobian(jacobian(f_loss, 0), 1) # grad contra 1er y grad contra el 2do



V, dim = W.shape
_, Ji, Xi = find(X[i,:]) # nonzero coocs of word i (indices and values)
Wi =  W[i,:]
Uj = U[Ji,:]
Yi = np.full_like(Xi, 0)
b_uj = b_u[Ji]
b_wi = b_w[i]
H1_i = np.linalg.inv(Hi(Uj, Xi))
jacobian_w_y = grad_w_y(Wi, Yi)
grad_wi_x = 1/V * np.matmul(H1_i, jacobian_w_y)
# scipy.sparse para llegar a matriz  final VxV
out = lil_matrix((dim, V))
out[:,Ji] = grad_wi_x



#%% Gradient Bias-X

# no se puede hacer el mismo analisis para si avanzar por gradiente
# cambia el bias

# TODO: ver porque no es simetrico el gradient

bias_grad = bias_gradient(idx_c, idx_a, idx_b, W, U, b_w, b_u, X)


def top_bottom_cells(bias_gradient_matrix, k=50):
    """Return dict of indices and values of top and bottom
    K gradients of cooc matrix"""
    ii, jj, vv = scipy.sparse.find(bias_gradient_matrix) # nonzero indices, columns and values
    # top
    idx_top = np.argpartition(vv, -k)[-k:] # top n values
    idx_top = idx_top[np.argsort(-vv[idx_top])] # sort from largest to lowest
    top = {(ii[idx], jj[idx]): vv[idx] for idx in idx_top}
    # bottom
    idx_bottom = np.argpartition(vv, k)[:k] # top n values
    idx_bottom = idx_bottom[np.argsort(vv[idx_bottom])] # sort from largest to lowest
    bottom = {(ii[idx], jj[idx]): vv[idx] for idx in idx_bottom}
    return top, bottom


top_cells, bottom_cells = top_bottom_cells(bias_grad, 50)
{(idx2str[k[0]], idx2str[k[1]]): v for k, v in top_cells.items() }
{(idx2str[k[0]], idx2str[k[1]]): v for k, v in bottom_cells.items() }


X[idx_c, idx_a[0]]
X[idx_c, idx_b[0]]
X[idx_a[0], idx_c]
X[idx_b[0], idx_c]

bias_grad[idx_c, idx_a[0]]
bias_grad[idx_c, idx_b[0]]
bias_grad[idx_a[0], idx_c]
bias_grad[idx_b[0], idx_c]

X[idx_c, 2]
X[idx_c, 1]
bias_grad[idx_c, 2]
bias_grad[idx_c, 1]


X[idx_c, str2idx['vulgate']]
X[idx_b[0], str2idx['vulgate']]




# indices = [idx_c] + idx_a + idx_b
# grad_w_x = csr_matrix((dim, V)) # inicializa matriz
# for i in indices:
#     grad_wi_x = gradient_wi_x(W, U, b_w, b_u, X, i)
#     grad_w_x += grad_wi_x





### test: subir por la direccion del gradiente aumenta el bias
grad_b_w = gradient_bias_w(W, idx_c, idx_a, idx_b)
W_tmp = W.copy()
biases_grad = list()
# A. actualizaciones con gradiente:
for i in range(10):
    W_tmp += grad_b_w * 0.1
    biases_grad.append(bias_rnd(W_tmp, idx_c, idx_a, idx_b))
biases_grad
# B. actualizaciones con pesos del gradiente permutados ("random"):
index = np.arange(W.shape[1])
np.random.shuffle(index)
grad_shuffled = grad_b_w[:,index].copy()
W_tmp = W.copy()
biases_shuffled = list()
for i in range(10):
    W_tmp += grad_shuffled * 0.1
    biases_shuffled.append(bias_rnd(W_tmp, idx_c, idx_a, idx_b))
biases_shuffled
# avanzar por el gradiente hace aumentar y "Mucho"
# avanzar por random hacer aumentar/caer monotonamente pero "poco"
# (es monotono porque es una funcion lineal)
###




















### no es simetrico!!! :(
# bias_gradient_matrix[31718, 64]
# bias_gradient_matrix[64, 31718]




# ### analisis de gradients
# str2idx['he']
# str2idx['she']
# str2idx['motherhood']
# aa = grad_bias_w[16,:].toarray()[0]
# bb = grad_bias_w[64,:].toarray()[0]
# import seaborn as sns
# sns.scatterplot(x=aa, y=bb)
# sns.scatterplot(x=W[16,:], y=aa)
# sns.scatterplot(x=W[64,:], y=bb)
# sns.scatterplot(x=W[16,:], y=W[64,:])
# sns.scatterplot(x=W[16,:], y=W[31718,:])
###




### palabras que mas coocurren con word i:
i = str2idx['motherhood']
top_j = np.argsort(-X[i,:].toarray())[0]
[idx2str[j] for j in top_j[:20]]
###


# ### chequeo: Coocs mas altas:
# ii, jj, Xi = find(X)
# top_j = np.argsort(-Xi)
# [(idx2str[ii[idx]], idx2str[jj[idx]])  for idx in top_j[:20]]
# ###
