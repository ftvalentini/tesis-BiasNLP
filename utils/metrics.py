import numpy as np
import pandas as pd
import json
from scipy.stats import norm

from utils.coocurrence import create_cooc_dict


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
    # ODDS ratio
    if (counts_a['context'] == 0) & (counts_a['context'] == 0):
        odds_ratio = float("nan")
    elif counts_b['context'] == 0:
        odds_ratio = float("inf")
    elif counts_a['context'] == 0:
        odds_ratio = float("-inf")
    else:
        odds_ratio = (counts_a['context']/counts_a['notcontext']) / \
            (counts_b['context']/counts_b['notcontext'])
    # ODDS ratio variance and CI
    if ci_level:
        log_odds_variance = \
            1 / counts_a['context'] + 1 / counts_a['notcontext'] + \
            1 / counts_b['context'] + 1 / counts_b['notcontext']
        qt = norm.ppf(1 - (1 - ci_level) / 2)
        lower = np.exp(np.log(odds_ratio) - qt * np.sqrt(log_odds_variance))
        upper = np.exp(np.log(odds_ratio) + qt * np.sqrt(log_odds_variance))
        return odds_ratio, lower, upper
    return odds_ratio


def bias_relative_norm_distance(vector_dict
                                ,words_target_a, words_target_b, words_context
                                ,ci_bootstrap_iters=None):
    """Return relative_norm_distance as bias metric between A/B wrt to Context
    (Garg et al 2018 Eq 3)
    """
    # TODO: handlear words out of vocab
    # normalize vecs to norm 1
    for word, vec in vector_dict.items():
        vector_dict[word] = vec / np.linalg.norm(vec)
    # average target vectors
    avg_a = np.mean(np.array([vector_dict[w] for w in words_target_a]), axis=0)
    avg_b = np.mean(np.array([vector_dict[w] for w in words_target_b]), axis=0)
    # array context vectors (dim x n_words)
    vectors_c = np.array([vector_dict[w] for w in words_context]).transpose()
    # para cada columna de C: substract avg_target y norma 2
    normdiffs_a = np.linalg.norm(np.subtract(vectors_c, avg_a[:,np.newaxis]), axis=0)
    normdiffs_b = np.linalg.norm(np.subtract(vectors_c, avg_b[:,np.newaxis]), axis=0)
    # relative norm distance: suma de las diferencias de normdiffs target para cada C
    # rel_norm_distance > 0: mas lejos de B que de A --> bias "a favor de" A
    rel_norm_distance = np.sum(normdiffs_b - normdiffs_a)
    if ci_bootstrap_iters:
        diffs = normdiffs_b - normdiffs_a
        rel_norm_distances = []
        np.random.seed(123)
        for b in range(ci_bootstrap_iters):
            indices = np.random.choice(len(diffs), len(diffs), replace=True)
            rel_norm_distances.append(np.sum(diffs[indices]))
        sd = np.std(rel_norm_distances)
        lower, upper = rel_norm_distance - sd, rel_norm_distance + sd
        return rel_norm_distance, lower, upper
    return rel_norm_distance


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
