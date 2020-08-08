import numpy as np
import pandas as pd
import json
from scipy.stats import norm, chi2_contingency
from tqdm import tqdm

from utils.coocurrence import create_cooc_dict, CREC, sizeof

def pmi(cooc_dict, words_target, words_context, str2count, alpha=1.0):
    """Return PPMI of words_target, words_context using cooc dict and str2count
    Asume que cooc_dict siempre tiene (i,j) y (j,i)
    """
    # words_outof_vocab = [word for word in words_target + words_context
    #                         if word not in str2count]
    # if words_outof_vocab:
    #     return f'{", ".join(words_outof_vocab)} : not in vocab'
    # contexto: probabilidad marginal
    count_context_alpha = sum(
        [str2count.get(w, 0)**alpha for w in words_context])
    count_total_alpha = sum([v**alpha for v in str2count.values()])
    prob_context_alpha = count_context_alpha / count_total_alpha
    # contexto: probabilidad condicional de coocurrencia con target
    count_context_con_target = 0
    for target in words_target:
        for context in words_context:
            count_context_con_target += cooc_dict.get((target, context), 0)
    count_target_total = sum([str2count.get(w, 0) for w in words_target])
    prob_context_con_target = count_context_con_target / count_target_total
    pmi = np.log(prob_context_con_target / prob_context_alpha)
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


def bias_pmi(cooc_dict, words_target_a, words_target_b, words_context, str2count,
            alpha=1.0):
    """ Return PMI(A,C)-PMI(A,C) between 2 sets of target words (A B) and a \
    set of context words.
    """
    pmi_a = pmi(cooc_dict, words_target_a, words_context, str2count, alpha=alpha)
    pmi_b = pmi(cooc_dict, words_target_b, words_context, str2count, alpha=alpha)
    bias = pmi_a - pmi_b
    return bias


def bias_odds_ratio(cooc_dict, words_target_a, words_target_b, words_context,
                    str2count, ci_level=None):
    """ Return bias OddsRatio A/B between 2 sets of target words (A B) and a set of \
    context words.
    If ci_level: return confidence interval at ci_level (oddsr, lower, upper)
    """
    def odds_ratio_counts(words_target):
        count_target_context = 0
        for target in words_target:
            for context in words_context:
                count_target_context += cooc_dict.get((target, context), 0)
        count_target_total = sum([str2count.get(w, 0) for w in words_target])
        count_target_notcontext = count_target_total - count_target_context
        result = {'context': count_target_context,
                  'notcontext': count_target_notcontext}
        return result
    counts_a = odds_ratio_counts(words_target_a)
    counts_b = odds_ratio_counts(words_target_b)
    result = odds_ratio(
        counts_a['context'], counts_a['notcontext'], counts_b['context']
        ,counts_b['notcontext'], ci_level=ci_level)
    return result


def differential_bias_bydoc(words_target_a, words_target_b, words_context,
                            cooc_dict, str2count,
                            metric="ppmi",
                            alpha=1.0, window_size=8,
                            corpus="corpora/simplewikiselect.txt",
                            corpus_metadata='corpora/simplewikiselect.meta.json'
                            ):
    """Return pd.DataFrame with one row by doc and columns:
        - id
        - line of corpus
        - name of document
        - differential bias (bias global - bias sin el texto)
    Param:
        - metric: "pmi", "odds_ratio"
    """
    # computa bias global
    def bias_total(cooc_dict, str2count, metric=metric):
        if metric == "pmi":
            return bias_pmi(cooc_dict, words_target_a, words_target_b \
                            , words_context, str2count, alpha=alpha)
        elif metric == "odds_ratio":
            return bias_odds_ratio(cooc_dict, words_target_a, words_target_b \
                            , words_context, str2count)
    bias_global = bias_total(cooc_dict, str2count, metric)
    # sets of words to test intersection with docs
    words_list = words_target_a + words_target_b + words_context
    words_target = set(words_target_a) | set(words_target_b)
    # read docs metadata to retrieve index, name
    with open(corpus_metadata, "r", encoding="utf-8") as f:
        docs_metadata = json.load(f)['index']
    # for each doc: read metadata + read doc + cooc_matrix + differential bias
    result = {'id': [], 'line': [], 'name': [], 'diff_bias': []}  # init results dict
    i = -1  # init doc counter
    with open(corpus, "r", encoding="utf-8") as f:
        while True:
            i += 1
            doc = f.readline().strip()
            # end when no more docs to read:
            if not doc:
                break
            words_doc = doc.split()
            # si no hay words target --> not parse:
            if not bool(set(words_doc) & words_target):
                continue
            doc_metadata = [d for d in docs_metadata if d['line'] == i][0]
            # compute cooc_dict y str2count of document
            cooc_dict_i = create_cooc_dict(
                doc, words_list, window_size=window_size)
            str2count_i = pd.value_counts(words_doc).to_dict()
            # update cooc global y str2count global
            cooc_dict_new = \
                    {k: v - cooc_dict_i.get(k, 0) for k, v in cooc_dict.items()}
            str2count_new = \
                    {w: str2count.get(w, 0) - str2count_i.get(w, 0) \
                                                            for w in words_list}
            # compute new bias
            bias_global_new = bias_total(cooc_dict_new, str2count_new, metric)
            # Differential bias
            diff_bias = bias_global - bias_global_new
            # make results DataFrame
            result['id'].append(doc_metadata['id'])
            result['line'].append(i)
            result['name'].append(doc_metadata['name'])
            result['diff_bias'].append(diff_bias)
    return pd.DataFrame(result)


def bias_byword(words_target_a, words_target_b, words_context, str2idx, str2count,
                ci_level=.95, cooc_file='embeddings/cooc-C0-V20-W8-D0.bin'):
    """
    Return DataFrame with Odds Ratio A/B for each word in words_context \
    and the relevant coocurrence counts
    """
    words_outof_vocab = [w for w in words_target_a + words_target_b + \
                                            words_context if w not in str2idx]
    if words_outof_vocab:
        print(f'{", ".join(words_outof_vocab)} \nNOT IN VOCAB')
    idx2str = {str2idx[w]: w for w in words_context}
    target_indices_a = sorted([str2idx[word] for word in words_target_a])
    target_indices_b = sorted([str2idx[word] for word in words_target_b])
    target_indices = sorted(target_indices_a + target_indices_b)
    context_indices = sorted([str2idx[word] for word in words_context])
    size_crec = sizeof(CREC) # crec: structura de coocucrrencia en Glove
    # init dict of odds_ratio counts for each context word
    str2counts = {w: dict() for w in words_context}
    # open bin file sorted by idx1
    pbar = tqdm(total=target_indices[-1] * len(str2idx))
    with open(cooc_file, 'rb') as f:
        cr = CREC()
        k = 0
        current_idx = target_indices[0]
        while (f.readinto(cr) == size_crec):
            pbar.update(1)
            if cr.idx1 in target_indices:
                if cr.idx2 in context_indices:
                    word = idx2str[cr.idx2]
                    if cr.idx1 in target_indices_a:
                        str2counts[word]['context_a'] = \
                            str2counts[word].get('context_a', 0) + cr.value
                    elif cr.idx1 in target_indices_b:
                        str2counts[word]['context_b'] = \
                            str2counts[word].get('context_b', 0) + cr.value
                if cr.idx1 > current_idx:
                    k += 1
                    current_idx = target_indices[k]
            # stop if idx1 is higher than highest target idx
            if cr.idx1 > target_indices[-1]:
                break
    pbar.close()
    df_counts = pd.DataFrame.from_dict(str2counts, orient='index').fillna(0)
    # Add columnas de odds ratio
    count_target_a = sum([str2count.get(w,0) for w in words_target_a])
    count_target_b = sum([str2count.get(w,0) for w in words_target_b])
    df_counts['notcontext_a'] = count_target_a - df_counts['context_a']
    df_counts['notcontext_b'] = count_target_b - df_counts['context_b']
    def odds_(a, b, c, d, ci_level):
        return pd.Series(odds_ratio(a, b, c, d, ci_level=ci_level))
    df_odds = df_counts.apply(
        lambda d: odds_(d['context_a'], d['notcontext_a'] \
                        ,d['context_b'], d['notcontext_b'], ci_level=ci_level) \
                            ,axis=1)
    df_odds.columns = ['odds_ratio','upper','lower','pvalue']
    result = pd.concat([df_counts, df_odds], axis=1)
    return result
