import autograd.numpy as np
from autograd import grad, jacobian, hessian
from autograd import elementwise_grad as egrad

from scipy.sparse import lil_matrix, csr_matrix, find


def f_glove(x, x_max=100, alpha=0.75):
    """GloVe f() weighting function for coocurrences
    Parameters:
        x: np.ndarray of coocs
    """
    # out = (x/x_max)**alpha
    # out[x >= x_max] = 1.0
    ### for scipy.sparse:
    # out = (x/x_max).power(alpha).minimum(1)
    ###
    out = np.minimum((x/x_max)**alpha, 1)
    return out


def Li(Wi, Yi, Xi, b_wi, Uj, b_uj):
    """Pointwise loss for word i
    """
    diff = np.dot(Uj, Wi) + b_uj + b_wi - np.log(Xi - Yi)
    return sum(f_glove(Xi - Yi) * (diff**2))


def Hi(Uj, Xi):
    """Hessian of pointwise loss wrt Wi
    Nota: H_wi tiene dim D x D (page 4 bottom)
    """
    temp = np.sqrt(2 * f_glove(Xi)) * Uj.T
    return np.matmul(temp, temp.T)
    ### FOR scipy.sparse (REVISAR, SI LO USO!!!):
    # temp = np.sqrt(2 * f_glove(Xi)) * Uj
    # return np.matmul(temp.toarray().T, temp.toarray())
    # return temp.T.dot(temp)
    ###


def bias_rnd(W, idx_c, idx_a, idx_b):
    """Relative norm distance bias (Garg 2018) for a context word with idx_c
    and target words with idx_a and idx_b
    """
    Wi = W[idx_c,:]
    avg_t1 = np.mean(W[idx_a,:], axis=0)
    avg_t2 = np.mean(W[idx_b,:], axis=0)
    return np.linalg.norm(Wi - avg_t2) - np.linalg.norm(Wi - avg_t1)


def gradient_bias_w(W, idx_c, idx_a, idx_b):
    """
    Gradient of RND bias wrt W matrix
    Returns scipy.sparse.csr_matrix of gradient
    """
    f_bias = lambda w: bias_rnd(w, idx_c, idx_a, idx_b)
    grad_bias_w = jacobian(f_bias)(W)
    grad_bias_w = csr_matrix(grad_bias_w)
    return grad_bias_w


def gradient_wi_x(W, U, b_w, b_u, X, i):
    """
    Gradient of Wi (embedding of word i) wrt to Xi (coocurrences of word i)
    Returns scipy.sparse.csr_matrix
    Param:
        - X: coocurrence scipy.sparse matrix
        - W, U, b_w, b_u: np.array GloVe vectors and biases
        - i: idx of word
    """
    V, dim = W.shape
    _, Ji, Xi = find(X[i,:]) # nonzero coocs of word i (indices and values)
    Wi =  W[i,:]
    Uj = U[Ji,:]
    Yi = np.full_like(Xi, 0)
    b_uj = b_u[Ji]
    b_wi = b_w[i]
    H1_i = np.linalg.inv(Hi(Uj, Xi))
    f_loss = lambda w, y: Li(w, y, Xi, b_wi, Uj, b_uj)
    grad_w_y = jacobian(jacobian(f_loss, 0), 1) # grad contra 1er y grad contra el 2do
    jacobian_w_y = grad_w_y(Wi, Yi)
    grad_wi_x = 1/V * np.matmul(H1_i, jacobian_w_y)
    # scipy.sparse para llegar a matriz  final VxV
    out = lil_matrix((dim, V))
    out[:,Ji] = grad_wi_x
    return out.tocsr()


def bias_gradient(idx_c, idx_a, idx_b, W, U, b_w, b_u, X):
    V, dim = W.shape
    # GRADIENTE BIAS_W (para toda la matriz de una -- estara ok?)
    grad_bias_w = gradient_bias_w(W, idx_c, idx_a, idx_b)
    # GRADIENTES W_X (para cada palabra)
    indices = [idx_c] + idx_a + idx_b
    grad_w_x = csr_matrix((dim, V)) # inicializa matriz
    for i in indices:
        grad_wi_x = gradient_wi_x(W, U, b_w, b_u, X, i)
        grad_w_x += grad_wi_x
    out = grad_bias_w.dot(grad_w_x) # muy pesado sin scipy.sparse
    return out





# ### Chequeo autodiff equivale a manual:
# def f_glove_d(x, x_max=100, alpha=0.75):
#     """1st derivative of GloVe f() weighting function for coocurrences
#     Parameters:
#         x: np.ndarray of coocs
    # """
    # out = alpha * (x/x_max)**(alpha-1) * (1/x_max)
    # out[x >= x_max] = 0
    # return out
# import autograd.numpy as np
# from autograd import grad
# from autograd import elementwise_grad as egrad
# grad_f = egrad(f_glove)
# x = np.array(list(range(10, 150)))
# y1 = grad_f(x)
# y2 = f_glove_d(x)
# sns.lineplot(x, y1)
# sns.lineplot(x, y2)
# ###

# ### plot f_glove and derivative
# import seaborn as sns
# x = np.array(list(range(0, 200)))
# y = f_glove(x)
# y_ = f_glove_d(x)
# sns.lineplot(x, y)
# sns.lineplot(x, y_)
# ### is not differentiable in Y at Y = 0 where Xij = 0, the bias gradient is only defined at
# ### non-zero co-occurrences

# import jax.numpy as jnp # no funciona en windows
# from jax import grad # no funciona en windows



# def Li(W, X, b_w, U, b_u, i):
#     """GloVe pointwise loss
#     Parameters:
#         - X: cooc matrix (scipy.sparse)
#         - W, b_w, U, b_u: GloVe vectors and biases (numpy.ndarray)
#         - i: index of word
#     """
#     _, Ji, Xi = find(X[i,:]) # nonzero coocs of word i (indices and values)
#     if len(Xi) == 0:
#         return 0
#     diff = np.dot(U[Ji,:], W[i,:]) + b_u[Ji] + b_w[i] - np.log(Xi)
#     return sum(f_glove(Xi) * (diff**2))


# def Li(Wi, Xi, b_wi, Uj, b_uj):
#     """GloVe pointwise loss
#     Parameters:
#         - X: cooc matrix (scipy.sparse)
#         - W, b_w, U, b_u: GloVe vectors and biases (numpy.ndarray)
#         - i: index of word
#     """
#     diff = np.dot(Uj, Wi) + b_uj + b_wi - np.log(Xi)
#     return sum(f_glove(Xi) * (diff**2))


# def J(W, X, b_w, U, b_u):
#     """GloVe total loss
#     Parameters:
#         - X: cooc matrix (scipy.sparse)
#         - W, b_w, U, b_u: GloVe vectors and biases (numpy.ndarray)
#     Nota: vocab indices empiezan en 1
#     """
#     loss = 0
#     for i in range(1, V+1):
#         loss += Li(X, W, b_w, U, b_u, i)
#     return loss


# def Hi(U, X, i):
#     """Hessian of pointwise loss wrt Wi
#     Nota: H_wi tiene dim D x D (page 4 bottom)
#     """
#     _, Ji, Xi = find(X[i,:]) # nonzero coocs of word i (indices and values)
#     temp = np.sqrt(2 * f_glove(Xi)) * U[Ji,:].T
#     return np.matmul(temp, temp.T)


# def gradient_x_wi():
#     """Approximation of gradient of word vector i wrt X
#     Ecs 11 and 11bis (la que sigue a la 11)
#     """
#     aa = 2 * V * U[j,:] * (f_glove(Xij)/(Xij)) - \
#                     f_glove_d(Xij)*(W[i,:]*U[j,:] + b_w[i] + b_u[j] - np.log(Xij))
#     # hay que sumar across j
#     return




# ### H_ es simetrica
# np.allclose(H_, H_.T)
# np.allclose(np.linalg.inv(H_), np.linalg.inv(H_).T)




### nonzero rows de grad_bias_w:
# ii, jj, vv = find(grad_bias_w)
# np.unique(ii)
# sorted([i] + j_t1 + j_t2)
###


# def bias_rnd(Wi, avg_t1, avg_t2):
#     return np.linalg.norm(Wi - avg_t2) - np.linalg.norm(Wi - avg_t1)
# f_bias = lambda w: bias_rnd(w, avg_t1, avg_t2)
# grad_bias_w = jacobian(f_bias)(Wi)
### esto deja dim (1,100)


#### OLD: autograd NO FUNCIONA CON scipy.sparse :(
# # sparse matrices para conservar indices originales
# Wi = scipy.sparse.lil_matrix(W.shape)
# Uj = scipy.sparse.lil_matrix(U.shape)
# Xi = scipy.sparse.lil_matrix(X[i,:])
# Wi[i,:] = W[i,:]
# Uj[Ji,:] = U[Ji,:]
# Yi = scipy.sparse.lil_matrix(Xi.shape)
###

# ### test: gradiente de euclidean norm: manual  vs autodiff
# aa = np.array([1,2,3,6])
# ff = lambda x: np.linalg.norm(x)
# grad_manual = aa / np.linalg.norm(aa)
# grad_auto = jacobian(ff)(aa)
# ###


# ### test: jacobian es lo mismo que egrad (en este caso al menos)
# f_loss = lambda w, y: Li(w, y, Xi, b_wi, Uj, b_uj)
# # jac_loss = jacobian(f_loss, 0)
# aa = jacobian(f_loss, 0)(Wi, Yi)
# bb = egrad(f_loss, 0)(Wi, Yi)
# # aa - bb # es lo mismo
# ###

# ### test: H_wi con loop vs H funcion brunet github vs H autodiff
# # con loop
# H_rdo = 0
# for ii in range(Xi.shape[0]):
#     H_rdo += 2 * f_glove(Xi[ii]) * np.outer(Uj[ii], Uj[ii])
# # autodiff
# f_loss = lambda w, y: Li(w, y, Xi, b_wi, Uj, b_uj)
# H_auto = hessian(f_loss, 0)(Wi, Yi)
# np.isclose(H_rdo, H_auto).all()
# # manual brunet
# H_wi = Hi(Uj, Xi)
# # rdo: son iguales los tres!
# # np.isclose(H_rdo, H_auto)
# # np.isclose(H_rdo, H_wi)
# # np.isclose(H_auto, H_wi)
# ###

# ### chequeo: hessiano egrad-egrad == hessiano manual
# H_wi = Hi(Uj, Xi) # manual
# loss_i = lambda w: Li(w, Yi=Yi, Xi=Xi, b_wi=b_wi, Uj=Uj, b_uj=b_uj)
# Hi_auto = egrad(egrad(loss_i))
# H_wi2 = Hi_auto(Wi) / V # autodiff
# # rdo: no coinciden las dimensiones :(
# H_wi.shape
# H_wi2.shape
# ###

# ### test: distance manual vs numpy
# aa = np.array([1,2,3,6])
# bb = np.array([5,8,3,2])
# np.linalg.norm(aa - bb)
# np.sqrt(np.sum([i**2 for i in aa-bb]))
# ###
