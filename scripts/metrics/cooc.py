import numpy as np
import pandas as pd
import json
import scipy.sparse
import scipy.sparse.linalg
from scipy.stats import norm
import statsmodels.api as sm
from tqdm import tqdm

from .utils.coocurrence import create_cooc_matrix, create_cooc_matrix2


def pmi(cooc_matrix, words_target, words_context, str2idx, alpha=1.0):
    """Return PPMI of words_target, words_context using scipy.sparse cooc_matrix
    and str2idx
    Asume que cooc_matrix siempre tiene (i,j) y (j,i)
    """
    # handling out of vocab words
    words_outof_vocab = [w for w in words_target + words_context \
                                                            if w not in str2idx]
    if len(words_outof_vocab) == len(words_context+words_target):
        raise ValueError("ALL WORDS ARE OUT OF VOCAB")
    if words_outof_vocab:
        print(f'{", ".join(words_outof_vocab)} : not in vocab')
    # counts and probabilities
    C = cooc_matrix
    total_count = C.sum()
    ii = [str2idx[w] for w in words_target] # target rows indices
    jj = [str2idx[w] for w in words_context] # context cols indices
    Cij = C[ii,:][:,jj].sum() # frecuencia conjunta
    Ci = C[ii,:].sum() # sum of target rows
    Cj = C[:,jj].sum() # sum of context cols
    Pij = Cij / total_count # prob conjunta
    Pi = Ci / total_count # prob marginal target
    Pj = Cj / total_count # prob marginal context
    # PMI
    pmi = np.log(Pij / (Pi * Pj))
    # # PPMI
    # ppmi = max(0, pmi)
    return pmi


def log_oddsratio(counts_a, counts_not_a, counts_b, counts_not_b, ci_level=None):
    """Return log Odds Ratio of counts_a, counts_not_a, counts_b, counts_not_b
    If ci_level: return (oddsratio, lower, upper, pvalue)
    """
    table = np.array([[counts_a, counts_not_a], [counts_b, counts_not_b]])
    t22 = sm.stats.Table2x2(table, shift_zeros=True)
    log_odds_ratio = t22.log_oddsratio
    if ci_level:
        lower, upper = t22.log_oddsratio_confint(alpha=1-ci_level, method="normal")
        pvalue = t22.log_oddsratio_pvalue()
        return log_odds_ratio, lower, upper, pvalue
    return log_odds_ratio


def bias_ppmi(cooc_matrix, words_target_a, words_target_b, words_context, str2idx,
            alpha=1.0):
    """ Return PPMI(A,C)-PPMI(A,C) between 2 sets of target words (A B) and a \
    set of context words.
    """
    pmi_a = pmi(cooc_matrix, words_target_a, words_context, str2idx, alpha=alpha)
    pmi_b = pmi(cooc_matrix, words_target_b, words_context, str2idx, alpha=alpha)
    bias = max(0,pmi_a) - max(0,pmi_b)
    return bias


def contingency_counts(cooc_matrix, words_target_a, words_target_b, words_context,
                       str2idx):
    """ Return coocurrence counts used to compute OddsRatio A/B \
    between 2 sets of target words (A B) and a set of context words.
    """
    C = cooc_matrix
    # indices
    idx_a = [str2idx[w] for w in words_target_a] # target A indices
    idx_b = [str2idx[w] for w in words_target_b] # target B indices
    idx_c = [str2idx[w] for w in words_context] # context indices
    idx_notc = [str2idx[w] for w in str2idx.keys() \
                                    if w not in words_context] # not context indices
    # frecuencias
    context_a = C[idx_a,:][:,idx_c].sum()
    context_b = C[idx_b,:][:,idx_c].sum()
    notcontext_a = C[idx_a,:][:,idx_notc].sum()
        # notcontext_a = C[idx_a,:].sum() - context_a # EQUIVALENTE a linea anterior
    notcontext_b = C[idx_b,:][:,idx_notc].sum()
    return (context_a, notcontext_a, context_b, notcontext_b)


def bias_logoddsratio(cooc_matrix, words_target_a, words_target_b, words_context,
                    str2idx, ci_level=None):
    """ Return bias log(OddsRatio) A/B between 2 sets of target words (A B) \
    and a set of context words
    If ci_level: return confidence interval at ci_level (oddsr, lower, upper)
    """
    context_a, notcontext_a, context_b, notcontext_b = \
        contingency_counts(
            cooc_matrix, words_target_a, words_target_b, words_context, str2idx)
    result = log_oddsratio(
        context_a, notcontext_a, context_b, notcontext_b, ci_level=ci_level)
    return result


def dpmi_byword(cooc_matrix, words_target_a, words_target_b, words_context, str2idx
                ,ci_level=.95):
    """
    Return DataFrame with DPMI and log(OddsRatio) A/B for each word in
    words_context and the relevant coocurrence counts
    """
    # handling target words out of vocab
    words_target = words_target_a + words_target_b
    words_outof_vocab = [w for w in words_target if w not in str2idx]
    if len(words_outof_vocab) == len(words_target):
        raise ValueError("ALL WORDS ARE OUT OF VOCAB")
    if words_outof_vocab:
        print(f'{", ".join(words_outof_vocab)} \nNOT IN VOCAB')
    # matrix statistics
    C = cooc_matrix
    # words indices
    idx_a = [str2idx[w] for w in words_target_a if w not in words_outof_vocab]
    idx_b = [str2idx[w] for w in words_target_b if w not in words_outof_vocab]
    idx_c = sorted(
            [str2idx[w] for w in words_context if w not in words_outof_vocab])
        # words context siempre sorted segun indice!!!
    # frecuencias
    print("Computing counts...\n")
    total_count = C.sum() # total
    count_a = C[idx_a,:].sum() # total target A
    count_b = C[idx_b,:].sum() # total target B
    counts_context = C[:,idx_c].sum(axis=0) # totales de cada contexto
    counts_context_a = C[idx_a,:][:,idx_c].sum(axis=0) # de cada contexto con target A
    counts_context_b = C[idx_b,:][:,idx_c].sum(axis=0) # de cada contexto con target A
    counts_notcontext_a = count_a - counts_context_a # de A sin cada contexto
    counts_notcontext_b = count_b - counts_context_b # de B sin cada contexto
    # probabilidades
    print("Computing probabilities...\n")
    prob_a = count_a / total_count
    prob_b = count_b / total_count
    probs_context = counts_context / total_count
    probs_context_a = counts_context_a / total_count
    probs_context_b = counts_context_b / total_count
    # PMI
    pmi_a = np.log(probs_context_a / (prob_a * probs_context))
    pmi_b = np.log(probs_context_b / (prob_b * probs_context))
    # insert en DataFrame  segun word index
        # words context siempre sorted segun indice!!!
    print("Putting results in DataFrame...\n")
    str2idx_context = {w: str2idx[w] for w in words_context}
    df = pd.DataFrame(str2idx_context.items(), columns=['word','idx'])
    df['count_total'] = counts_context.T
    df['count_context_a'] = counts_context_a.T
    df['count_context_b'] = counts_context_b.T
    df['pmi_a'] = pmi_a.T
    df['pmi_b'] = pmi_b.T
    df['dppmi'] = np.maximum(0,df['pmi_a']) - np.maximum(0,df['pmi_b'])
    df['count_notcontext_a'] = counts_notcontext_a.T
    df['count_notcontext_b'] = counts_notcontext_b.T
    # add calculo de odds ratio
    print("Computing Odds Ratios...\n")
    def log_oddsratio_(a, b, c, d, ci_level):
        return pd.Series(log_oddsratio(a, b, c, d, ci_level=ci_level))
    df_odds = df.apply(
        lambda d: log_oddsratio_(d['count_context_a'], d['count_notcontext_a'] \
                                ,d['count_context_b'], d['count_notcontext_b']
                                , ci_level=ci_level), axis=1)
    df_odds.columns = ['log_oddsratio','lower','upper','pvalue']
    df_odds['odds_ratio'] = np.exp(df_odds['log_oddsratio'])
    # final result
    result = pd.concat([df, df_odds], axis=1)
    return result


def cosine_similarities(M, idx_target, idx_context, use_norm=True):
    """Return relative cosin similarity values between avg of words_target and
    words_context
    Param:
        - M: (V+1)x(V+1) matrix where row/column indices are indices of
        words given by str2idx (it can be cooc. matrix or PPMI matrix)
        - use_norm: divides by norm (as usual cosine) -- if False: only dot pr.
    Notes:
        - Rows of M are treated as word vectors
        - Must pass words indices (not words)
        - It works OK if len(idx_target) == 1
    """
    avg_target = M[idx_target,:].mean(axis=0)
    M_c = M[idx_context,:] # matriz de contexts words
    del M
    # similitud coseno
    productos = M_c.dot(avg_target.T)
    denominadores = np.linalg.norm(avg_target) * \
                                        scipy.sparse.linalg.norm(M_c, axis=1)
    # denominadores = scipy.sparse.linalg.norm(M_c, axis=1)
    # denominadores *= np.linalg.norm(avg_target)
    rel_sims = productos.ravel()
    if use_norm:
        rel_sims /= denominadores.ravel()
    out = np.array(rel_sims).ravel()
    return out


def relative_cosine_diffs(
    M, idx_target_a, idx_target_b, idx_context, use_norm=True, return_both=False):
    """Return relative cosine difference between A/B wrt to each Context
    Param:
        - M: (V+1)x(V+1) matrix where row/column indices are indices of
        words (it can be cooc matrix or PPMI matrix)
        - return_both: returns tuple (cos(a,c), cos(b,c))
        - use_norm: divides by norm (as usual cosine) -- if False: only dot pr.
    Notes:
        - Rows of M are treated as word vectors
        - Must pass words indices (not words)
    """
    # distancias de avg(target) cra cada context
    cos_a = cosine_similarities(M, idx_target_a, idx_context, use_norm=use_norm)
    cos_b = cosine_similarities(M, idx_target_b, idx_context, use_norm=use_norm)
    # > 0: mas cerca de A que de B --> bias "hacia" A
    if return_both:
        return cos_a, cos_b
    diffs = cos_a - cos_b
    return diffs


def order2_byword(M, words_target_a, words_target_b, words_context, str2idx
                  ,n_dim=None, only_positive=True):
    """
    Return DataFrame with
        - 2nd order coocurrence bias A/B of each Context word
        - freq of each word
    Param:
        - M: scipy.sparse matrix (Cooc. or PMI matrix)
        - n_dim: use first n_dim of each row if not None
    """
    # handling words out of vocab (asume que todas las context estan en vocab)
    words_target = words_target_a + words_target_b
    words_target_out = [w for w in words_target if w not in str2idx]
    if len(words_target_out) == len(words_target):
        raise ValueError("ALL TARGET WORDS ARE OUT OF VOCAB")
    if words_target_out:
        print(f'{", ".join(words_target_out)} \nNOT IN VOCAB')
    # words indices
    idx_a = sorted([str2idx[w] for w in words_target_a \
                                                if w not in words_target_out])
    idx_b = sorted([str2idx[w] for w in words_target_b \
                                                if w not in words_target_out])
    idx_c = sorted([str2idx[w] for w in words_context])
    # get bias for each c
    if n_dim:
        M = M[:,:n_dim+1] # +1 porque idx=0 esta vacio
    # max(celda, 0) para usar PPMI -- si se usa cooc no pasa nada
    if only_positive:
        M = M.maximum(0)
    # cosine biases
    dots_a, dots_b = relative_cosine_diffs(
                    M, idx_a, idx_b, idx_c, use_norm=False, return_both=True)
    cosines_a, cosines_b = relative_cosine_diffs(
                    M, idx_a, idx_b, idx_c, use_norm=True, return_both=True)
    # normas y NZ cells
    M_c = M[idx_c,:]
    del M
    normas = scipy.sparse.linalg.norm(M_c, axis=1)
    nnz = (M_c > 0).sum(1).A1 # contar NZ de cada palabra
    # results DataFrame (todos los resultados sorted by idx)
    str2idx_context = {w: str2idx[w] for w in words_context}
    results = pd.DataFrame(str2idx_context.items(), columns=['word','idx'])
    results['cosine_a'] = cosines_a
    results['cosine_b'] = cosines_b
    results['dot_a'] = dots_a
    results['dot_b'] = dots_b
    results['diff_cosine'] = cosines_a - cosines_b
    results['diff_dot'] = dots_a - dots_b
    results['norm'] = normas
    results['nnz'] = nnz
    return results


def differential_bias_bydoc(cooc_matrix
                            ,words_target_a, words_target_b, words_context
                            ,str2idx
                            ,metric="log_oddsratio"
                            ,alpha=1.0, window_size=8
                            ,corpus="corpora/simplewikiselect.txt"
                            ,corpus_metadata='corpora/simplewikiselect.meta.json'
                            ):
    """Return pd.DataFrame with one row by doc and columns:
        - id
        - line of corpus
        - name of document
        - differential bias (bias global - bias sin el texto)
    Param:
        - cooc_matrix: full coocurrence sparse matrix
        - metric: "pmi", "odds_ratio"
        - str2idx: el mismo that was used to build cooc_matrix
    """
    # computa coocurrence counts globales
    global_counts = contingency_counts(
            cooc_matrix, words_target_a, words_target_b, words_context, str2idx)
    # computa bias global
    if metric == "log_oddsratio":
        bias_global = log_oddsratio(*global_counts, ci_level=None)
    elif metric == "pmi":
        return # not yet implemented
    # words indices (to get them only once)
    idx_a = [str2idx[w] for w in words_target_a] # target A indices
    idx_b = [str2idx[w] for w in words_target_b] # target B indices
    idx_c = [str2idx[w] for w in words_context] # context indices
    idx_notc = [str2idx[w] for w in str2idx.keys() \
                                    if w not in words_context] # not context indices
    # sets of words to test intersection with docs
    words_list = words_target_a + words_target_b + words_context
    words_target = set(words_target_a) | set(words_target_b)
    # read docs metadata to retrieve index, name
    with open(corpus_metadata, "r", encoding="utf-8") as f:
        docs_metadata = json.load(f)['index']
    # for each doc: read metadata + read doc + cooc_matrix + differential bias
    # init results dict
    result = {'id': [], 'line': [], 'name': [], 'diff_bias': []}
    i = -1  # init doc counter
    pbar = tqdm(total=len(docs_metadata))
    with open(corpus, "r", encoding="utf-8") as f:
        while True:
            i += 1
            doc = f.readline().strip()
            pbar.update(1)
            # end when no more docs to read:
            if not doc:
                break
            tokens_doc = doc.split()
            # si no hay words target --> not parse:
            if not bool(set(tokens_doc) & words_target):
                continue
            # compute sparse cooc_matrix of same shape as full matrix
                # only with counts for words in words_list
            C_i = create_cooc_matrix2(
                doc, words_list, str2idx, window_size=window_size)
            # doc odds ratio counts
            doc_counts = (
                C_i[idx_a,:][:,idx_c].sum(), C_i[idx_b,:][:,idx_c].sum()
                , C_i[idx_a,:][:,idx_notc].sum(), C_i[idx_b,:][:,idx_notc].sum()
            )
            # global counts menos doc counts
            diff_counts = tuple(
                            [x[0]-x[1] for x in zip(global_counts, doc_counts)])
            # diferencia entre bias global y bias con diff_counts
            diff_bias = bias_global - \
                                log_oddsratio(*diff_counts, ci_level=None)
            # make results DataFrame (docs_metadata is ordered by line)
            result['id'].append(docs_metadata[i]['id'])
            result['line'].append(i)
            result['name'].append(docs_metadata[i]['name'])
            result['diff_bias'].append(diff_bias)
    pbar.close()
    return pd.DataFrame(result)
