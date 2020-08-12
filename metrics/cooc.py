import numpy as np
import pandas as pd
import json
from scipy.stats import norm, chi2_contingency
from tqdm import tqdm

from .utils.coocurrence import create_cooc_matrix


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
    return pmi


def odds_ratio(counts_a, counts_not_a, counts_b, counts_not_b, ci_level=None):
    """Return Odds Ratio of counts_a, counts_not_a, counts_b, counts_not_b
    words_context using cooc dict and str2count
    If ci_level: return (oddsratio, lower, upper, pvalue)
    """
    if (counts_a == 0) & (counts_b == 0):
        return float("nan")
    elif counts_b == 0:
        return float("inf")
    elif counts_a == 0:
        return float("-inf")
    else:
        odds_ratio = (counts_a/counts_not_a) / (counts_b/counts_not_b)
    # ODDS ratio variance, CI and pvalue
    if ci_level:
        log_odds_variance = \
            (1/counts_a) + (1/counts_not_a) + (1/counts_b) + (1/counts_not_b)
        qt = norm.ppf(1 - (1 - ci_level) / 2)
        lower = np.exp(np.log(odds_ratio) - qt * np.sqrt(log_odds_variance))
        upper = np.exp(np.log(odds_ratio) + qt * np.sqrt(log_odds_variance))
        def chisq_pvalue(a, not_a, b, not_b):
            matriz = np.array([[a, not_a], [b, not_b]])
            chi2, pv, dof, ex = chi2_contingency(matriz, correction=False)
            return pv
        pvalue = chisq_pvalue(counts_a, counts_not_a, counts_b, counts_not_b)
        return odds_ratio, lower, upper, pvalue
    return odds_ratio


def bias_pmi(cooc_matrix, words_target_a, words_target_b, words_context, str2idx,
            alpha=1.0):
    """ Return PMI(A,C)-PMI(A,C) between 2 sets of target words (A B) and a \
    set of context words.
    """
    pmi_a = pmi(cooc_matrix, words_target_a, words_context, str2idx, alpha=alpha)
    pmi_b = pmi(cooc_matrix, words_target_b, words_context, str2idx, alpha=alpha)
    bias = pmi_a - pmi_b
    return bias


def bias_odds_ratio(cooc_matrix, words_target_a, words_target_b, words_context,
                    str2idx, ci_level=None):
    """ Return bias OddsRatio A/B between 2 sets of target words (A B) and a set of \
    context words.
    If ci_level: return confidence interval at ci_level (oddsr, lower, upper)
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
    result = odds_ratio(
        context_a, notcontext_a, context_b, notcontext_b, ci_level=ci_level)
    return result


def differential_bias_bydoc(cooc_matrix
                            ,words_target_a, words_target_b, words_context
                            ,str2idx
                            ,metric="pmi"
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
    # computa bias global
    def bias_total(cooc_matrix, str2idx, metric=metric):
        if metric == "pmi":
            return bias_pmi(cooc_matrix, words_target_a, words_target_b \
                            , words_context, str2idx, alpha=alpha)
        elif metric == "odds_ratio":
            return bias_odds_ratio(cooc_matrix, words_target_a, words_target_b \
                            , words_context, str2idx)
    bias_global = bias_total(cooc_matrix, str2idx, metric)
    # sets of words to test intersection with docs
    words_list = words_target_a + words_target_b + words_context
    words_target = set(words_target_a) | set(words_target_b)
    # read docs metadata to retrieve index, name
    with open(corpus_metadata, "r", encoding="utf-8") as f:
        docs_metadata = json.load(f)['index']
    # for each doc: read metadata + read doc + cooc_matrix + differential bias
    result = {'id': [], 'line': [], 'name': [], 'diff_bias': []}  # init results dict
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
            doc_metadata = [d for d in docs_metadata if d['line'] == i][0]
            # compute sparse cooc_matrix of same shape as full matrix
                # only with counts for words in words_list
            cooc_matrix_i = create_cooc_matrix(
                doc, words_list, str2idx, window_size=window_size)
            # update cooc global
            cooc_matrix_new = cooc_matrix - cooc_matrix_i
            # compute new bias
            bias_global_new = bias_total(cooc_matrix_new, str2idx, metric)
            # Differential bias
            diff_bias = bias_global - bias_global_new
            # make results DataFrame
            result['id'].append(doc_metadata['id'])
            result['line'].append(i)
            result['name'].append(doc_metadata['name'])
            result['diff_bias'].append(diff_bias)
    pbar.close()
    return pd.DataFrame(result)


def bias_byword(cooc_matrix, words_target_a, words_target_b, words_context, str2idx
                ,ci_level=.95):
    """
    Return DataFrame with Odds Ratio A/B for each word in words_context \
    and the relevant coocurrence counts
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
    idx_c = sorted([str2idx[w] for w in words_context if w not in words_outof_vocab])
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
    pmi_a = probs_context_a / (prob_a * probs_context)
    pmi_b = probs_context_b / (prob_b * probs_context)
    # insert en DataFrame  segun word index
        # words context siempre sorted segun indice!!!
    print("Putting results in DataFrame...\n")
    str2idx_context = str2idx.copy()
    for w in words_target:
        del str2idx_context[w]
    df = pd.DataFrame(str2idx_context.items(), columns=['word','idx'])
    df['pmi_a'] = pmi_a.T
    df['pmi_b'] = pmi_b.T
    df['diff_pmi'] = df['pmi_a'] - df['pmi_b']
    df['count_context_a'] = counts_context_a.T
    df['count_notcontext_a'] = counts_notcontext_a.T
    df['count_context_b'] = counts_context_b.T
    df['count_notcontext_b'] = counts_notcontext_b.T
    # # add calculo de odds ratio
    # print("Computing Odds Ratios...\n")
    # def odds_(a, b, c, d, ci_level):
    #     return pd.Series(odds_ratio(a, b, c, d, ci_level=ci_level))
    # df_odds = df.apply(
    #     lambda d: odds_(d['count_context_a'], d['count_notcontext_a'] \
    #                     ,d['count_context_b'], d['count_notcontext_b']
    #                     , ci_level=ci_level), axis=1)
    # df_odds.columns = ['odds_ratio','upper','lower','pvalue']
    # df_odds['log_odds_ratio'] = np.log(df_odds['odds_ratio'])
    # # final result
    # result = pd.concat([df, df_odds], axis=1)
    result = df
    return result
