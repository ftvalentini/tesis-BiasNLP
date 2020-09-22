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
word_context = "king"
words_target_a = ['he']# ,'him','his','himself']
words_target_b = ['she']#,'her','hers','herself']
# words indices
idx_c = str2idx[word_context]
idx_a = sorted([str2idx[w] for w in words_target_a])
idx_b = sorted([str2idx[w] for w in words_target_b])

#%% Bias Gradient total

# prueba normalizando vectores en norma 1
# W /= np.linalg.norm(W, axis=1)[:,np.newaxis]
# U /= np.linalg.norm(U, axis=1)[:,np.newaxis]
bias_grad = bias_gradient(idx_c, idx_a, idx_b, W, U, b_w, b_u, X)

#%% Explora

# check simetria
bias_grad[idx_c, idx_a[0]]
bias_grad[idx_a[0], idx_c]
# NO DA SIMETRICO ("explicacion" está mas abajo)

# check other important gradients (chequear signos)
bias_grad[idx_a[0], idx_c]
bias_grad[idx_b[0], idx_c]
bias_grad[idx_c, idx_a[0]]
bias_grad[idx_c, idx_b[0]]


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

# largest gradients
top_cells, bottom_cells = top_bottom_cells(bias_grad, 300)
top = {(idx2str[k[0]], idx2str[k[1]]): v for k, v in top_cells.items() }
bottom = {(idx2str[k[0]], idx2str[k[1]]): v for k, v in bottom_cells.items() }
# solo con words frecuentes
{k:v for k,v in top.items() if str2count[k[0]] > 1000 and str2count[k[1]] > 1000}
{k:v for k,v in bottom.items() if str2count[k[0]] > 1000 and str2count[k[1]] > 1000}
# check similitud entre palabras
# entre si
aa = {(idx2str[k[0]], idx2str[k[1]]): np.dot(W[k[0]], W[k[1]]) for k in top_cells.keys() }
# entre w2 y context
bb = {(idx2str[idx_c], idx2str[k[1]]): np.dot(W[idx_c], W[k[1]]) for k in top_cells.keys() }

# DF para top
df_top = pd.DataFrame({
        'w0': [k[0] for k in list(aa.keys())]
        ,'w1': [k[1] for k in list(aa.keys())]
        ,'grad':list(top_cells.values())
        ,'sim':list(aa.values())
        ,'sim_w1_king':list(bb.values())}
        )

# dado el grad positivo, deberia suceder que sim(w1, king)>0 cuando w0 == 'he'
df_top.loc[df_['w0'] == 'he']
# dado el grad positivo, deberia suceder que sim(w1, king)<0 cuando w0 == 'she'
df_top.loc[df_['w0'] == 'she']

# check coocs
X[idx_c, idx_a[0]]
X[idx_a[0], idx_c]
X[idx_c, idx_b[0]]
X[idx_b[0], idx_c]


# X[str2idx['colombia'], str2idx['friedrich']]
# bias_grad[str2idx['colombia'], str2idx['friedrich']]
# bias_grad[str2idx['friedrich'], str2idx['colombia']]
# # "Austrian coach Friedrich Donnenfeld was the manager of Colombia"
#
# ### palabras que mas coocurren con word i:
# i = str2idx['cocaine']
# top_j = np.argsort(-X[i,:].toarray())[0]
# [idx2str[j] for j in top_j[:20]]
# ###

# no se puede hacer el mismo analisis para si avanzar por gradiente
# cambia el bias

#%% Change in X --> W --> Bias using bias gradient
# using EC 8 y el bias gradient ya calculado

from scipy.sparse import find, csr_matrix
from metrics.gradient import Hi, grad_loss_wi
indices = [idx_c] + idx_a + idx_b
V, dim = W.shape
bias0 = bias_rnd(W, idx_c, idx_a, idx_b)
W_new = W.copy() # WE que se van a perturbar
# i = 180
for i in indices:
    _, Ji, Xi = find(X[i,:])
    Wi =  W[i,:]
    Uj = U[Ji,:]
    b_uj = b_u[Ji]
    b_wi = b_w[i]
    H1_i = np.linalg.inv(Hi(Uj, Xi))
    # _, _, Yi_diff = find(bias_grad[i,:]) # perturbacion segun bias gradient
    # Yi_diff = -Yi_diff
    Yi_zero = np.full_like(Xi, 0)
    grad_L_wi = grad_loss_wi(Yi_zero, Wi, Xi, b_wi, Uj, b_uj)
    # perturbacion segun bias gradient (solo en las coocs preexistentes)
    Xi_new = np.array(Xi + bias_grad[i,Ji])[0]
    # Uj_new = U[Ji_new,:]
    # b_uj_new = b_u[Ji_new]
    # Yi_zero_new = np.full_like(Xi_new, 0)
    grad_L_wi_new = grad_loss_wi(Yi_zero, Wi, Xi_new, b_wi, Uj, b_uj)
    Wi_new = Wi - 1/V * np.dot(H1_i, grad_L_wi_new - grad_L_wi)
    W_new[i,:] = Wi_new
bias_new = bias_rnd(W_new, idx_c, idx_a, idx_b)
bias_new - bias0 # dif debe dar positiva
### da positiva pero pequeña (está ok porque es infinitesimal?)


#%% analisis de simetria

import autograd.numpy as np
from autograd import grad, jacobian, hessian
from scipy.sparse import csr_matrix, find, lil_matrix
idx_c = str2idx['king']
idx_a = [str2idx['he']]
idx_b = [str2idx['she']]
# 1. gradient bias w
f_bias = lambda w: bias_rnd(w, idx_c, idx_a, idx_b)
grad_bias_w = jacobian(f_bias)(W)
grad_bias_w = csr_matrix(grad_bias_w)
print(W.shape)
assert grad_bias_w.shape == W.shape
ii, jj, Wij = find(grad_bias_w)
# --> hay valores no nulos en los WE de las palabras
# 2. Hessian de pointwise loss en Xi wrt Wi
from metrics.gradient import Hi, grad_loss_wi
i = idx_c
_, Ji, Xi = find(X[i,:])
Uj = U[Ji,:]
H_i = Hi(Uj, Xi)
# (shape DxD --> derivadas segundas de WE i)
assert np.allclose(H_i.T, H_i) # El hessiano es simetrico :)
H1_i = np.linalg.inv(H_i)
assert np.allclose(H1_i.T, H1_i) # El hessiano es simetrico :)
# 3. grad_wi_x de cada palabra i
Wi =  W[i,:]
Yi = np.full_like(Xi, 0)
b_uj = b_u[Ji]
b_wi = b_w[i]
V, dim = W.shape
f_grad_loss_wi = lambda y: V * grad_loss_wi(y, Wi, Xi, b_wi, Uj, b_uj)
jacobian_w_y = jacobian(f_grad_loss_wi)(Yi) # cambio de W ante cambio en coocs positivas
print(jacobian_w_y.shape) # dim D x N_coocs_i
print(Ji.shape)
# 4. gradiente Wi X
# version q aumenta dim al final
grad_wi_x = 1/V * np.matmul(H1_i, jacobian_w_y)
grad_w_x = lil_matrix((dim, V))
grad_w_x[:,Ji] = grad_wi_x
grad_w_x = grad_w_x.toarray()
# version q aumenta dim al ppio
jacobian_bis = lil_matrix((dim, V)) # jacobiano con todas las columnas
jacobian_bis[:,Ji] = jacobian_w_y
grad_w_x_bis = 1/V * np.matmul(H1_i, jacobian_bis.toarray()) #premult for H "reescala" el J
# grad_w_x_bis = csr_matrix(grad_w_x_bis)
assert np.allclose(grad_w_x_bis, grad_w_x) # da igual
# chequea forma
grad_w_x = csr_matrix(grad_w_x)
ii, jj, xx = find(grad_w_x)
assert np.allclose(np.unique(jj), Ji) # en las columnas quedan coocs con i
# 5. mult final
grad_bias_w = csr_matrix(grad_bias_w)
# dejo solo el bias_w de i:
grad_bias_w_i = csr_matrix(grad_bias_w.shape)
grad_bias_w_i[i,:] = grad_bias_w[i,:]
ff = grad_bias_w_i.dot(grad_w_x)
# test symmetry
tmp = ff - ff.T
np.all(np.abs(tmp.data) < 1e-10)
### no es simetrico
ii, jj, xx = find(ff)
np.unique(ii) # solo queda i en las filas
assert np.allclose(np.unique(jj), Ji) # en las columnas quedan coocs con i
# al usar todos los bias_w en la multiplicacion:
gg = grad_bias_w.dot(grad_w_x)
ii, jj, xx = find(gg)
np.unique(ii) # quedan i del bias en las filas
ii, jj, xx = find(gg[i,:])
assert np.allclose(np.unique(jj), Ji) # en las columnas quedan coocs con i

type(bias_grad)

grad_bias_w.shape
grad_w_x.shape

i, j, x = find(X[idx_c,:])
ii, jj, xx = find(bias_grad[idx_c,:])
j_notin_x = set(jj) - set(j)
len(j)
len(jj)
jj[-1]
idx2str[272808]
X[idx_c, 272808]
bias_grad[idx_c, 272808]
X[idx_a[0], 272808]
X[idx_b[0], 272808]


top_cells, bottom_cells = top_bottom_cells(ff, 20)
{(idx2str[k[0]], idx2str[k[1]]): v for k, v in top_cells.items() }
{(idx2str[k[0]], idx2str[k[1]]): v for k, v in bottom_cells.items() }

### NOTA: esta chequeado que al cambiar targets, el bias grad de C cambia
# (ej: con king-he-she y king-argentina-spain --> los rdos para king cambian)
###

ff[str2idx['colombia'], str2idx['friedrich']]
ff[str2idx['friedrich'], str2idx['colombia']]


# entender geometria de vectores
import seaborn as sns
sns.scatterplot(W[idx_c,:], W[idx_a[0],:])
sns.scatterplot(W[idx_c,:], W[idx_b[0],:])
np.dot(W[str2idx['she'],:], W[str2idx['woman'],:])
np.dot(W[str2idx['elephant'],:], W[str2idx['woman'],:])
sns.scatterplot(W[str2idx['she'],:], W[str2idx['woman'],:])
sns.scatterplot(W[str2idx['elephant'],:], W[str2idx['woman'],:])
aa = np.array([1,2,3])
bb = np.array([3,2,1])
cc = np.array([-2,-2,-2])
np.dot(aa, bb)
np.dot(aa, cc)
np.dot(aa, aa)

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


#%% ETC

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



# indices = [idx_c] + idx_a + idx_b
# grad_w_x = csr_matrix((dim, V)) # inicializa matriz
# for i in indices:
#     grad_wi_x = gradient_wi_x(W, U, b_w, b_u, X, i)
#     grad_w_x += grad_wi_x

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

# ### chequeo: Coocs mas altas:
# ii, jj, Xi = find(X)
# top_j = np.argsort(-Xi)
# [(idx2str[ii[idx]], idx2str[jj[idx]])  for idx in top_j[:20]]
# ###
