import numpy as np
import pandas as pd
import json
from scipy.stats import norm

from utils.coocurrence import create_cooc_dict


def pmi(cooc_dict, words_target, words_context, str2count, alpha=.75):
    """Return PPMI of words_target, words_context using cooc dict and str2count
    """
    # words_outof_vocab = [word for word in words_target + words_context
    #                         if word not in str2count]
    # if words_outof_vocab:
    #     return f'{", ".join(words_outof_vocab)} : not in vocab'
    count_total = sum(str2count.values())
    count_total_alpha = sum([v**alpha for v in str2count.values()])
    count_target = sum([str2count.get(word, 0) for word in words_target])
    count_context_alpha = sum(
        [str2count.get(word, 0)**alpha for word in words_context])
    count_target_context = 0
    for target in words_target:
        for context in words_context:
            count_target_context += cooc_dict.get((target, context), 0)
    prob_target = count_target / count_total
    prob_context_alhpa = count_context_alpha / count_total_alpha
    prob_target_context = count_target_context / count_total
    pmi = np.log(prob_target_context / (prob_target * prob_context_alhpa))
    return pmi


def bias_odds_ratio(cooc_dict, words_target_a, words_target_b, words_context,
                    str2count, ci_level=None):
    """ Return bias OddsRatio A/B between 2 sets of target words (A B) and a set of \
    context words.
    If ci_level returns confidence interval at ci_level (oddsr, lower, upper)
    """
    # ODDS de words A
    count_target_a = sum([str2count[word] for word in words_target_a])
    count_target_a_context = 0
    for target in words_target_a:
        for context in words_context:
            count_target_a_context += cooc_dict.get((target, context), 0)
    odds_a = count_target_a_context / (count_target_a - count_target_a_context)
    # ODDS de words B
    count_target_b = sum([str2count[word] for word in words_target_b])
    count_target_b_context = 0
    for target in words_target_b:
        for context in words_context:
            count_target_b_context += cooc_dict.get((target, context), 0)
    odds_b = count_target_b_context / (count_target_b - count_target_b_context)
    # ODDS ratio
    if (odds_a == 0) & (odds_b == 0):
        odds_ratio = float("nan")
    elif odds_b == 0:
        odds_ratio = float("inf")
    elif odds_a == 0:
        odds_ratio = float("-inf")
    else:
        odds_ratio = odds_a / odds_b
    # ODDS ratio variance and CI
    if ci_level:
        log_odds_variance = \
            1 / count_target_a_context + 1 / count_target_b_context + \
            1 / (count_target_a - count_target_a_context) + \
            1 / (count_target_b - count_target_b_context)
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


def bias_ppmi_bydoc(words_target_a, words_target_b, words_context,
                    alpha=.75, window_size=8,
                    corpus="corpora/simplewikiselect.txt",
                    corpus_metadata='corpora/simplewikiselect.meta.json'
                    ):
    """Return pd.DataFrame with one row by doc and columns:
        - id
        - line of corpus
        - name
        - difference PPMI(A,C) - PPMI(B,C)
            - Si no hay Cooc target-context --> no se calcula
            - Si solo hay Cooc targetA-context --> return PPMI(A,C)
            - Si solo hay Cooc targetB-context --> return PPMI(B,C)
            - Si hay ambas Cooc --> return difference
        TODO: ppmi en realidad nunca puede ser 0...
    """
    word_list = words_target_a + words_target_b + words_context
    # sets of words to test intersection with docs
    words_target = set(words_target_a) | set(words_target_b)
    # read docs metadata to retrieve index, name
    with open(corpus_metadata, "r", encoding="utf-8") as f:
        docs_metadata = json.load(f)['index']
    # for each doc: read metadata + read doc + cooc_matrix + bias
    result = {'id': [], 'line': [], 'name': [],
              'diff_ppmi': []}  # init results dict
    i = -1  # init doc counter
    with open(corpus, "r", encoding="utf-8") as f:
        while True:
            i += 1
            doc = f.readline().strip()
            # end when no more docs to read:
            if not doc:
                break
            words_doc = doc.split()
            # si no hay words context --> not parse:
            if not bool(set(words_doc) & set(words_context)):
                continue
            # si no hay words target --> not parse:
            elif not bool(set(words_doc) & words_target):
                continue
            # si hay words target y context --> parse:
            doc_metadata = [d for d in docs_metadata if d['line'] == i][0]
            cooc_dict = create_cooc_dict(
                doc, word_list, window_size=window_size)
            words_target_cooc = [k[0] for k in cooc_dict.keys()
                                 if k[0] in words_target and k[1] in words_context]
            # si no hay coocurrencia target-context --> no calcular ppmi:
            if not words_target_cooc:
                continue
            pmi_a = 0 # pmi default
            pmi_b = 0 # pmi default
            str2count = pd.value_counts(words_doc).to_dict()
            # si hay coocurrencia targetA-context: get ppmi_a
            if bool(set(words_target_a) & set(words_target_cooc)):
                pmi_a = pmi(cooc_dict, words_target_a,
                            words_context, str2count, alpha=alpha)
            # si hay coocurrencia targetB-context: get ppmi_b
            if bool(set(words_target_b) & set(words_target_cooc)):
                pmi_b = pmi(cooc_dict, words_target_b,
                            words_context, str2count, alpha=alpha)
            # Bias: diff_ppmi
            diff_ppmi = max(0, pmi_a) - max(0, pmi_b)
            # make results DataFrame
            result['id'].append(doc_metadata['id'])
            result['line'].append(i)
            result['name'].append(doc_metadata['name'])
            result['diff_ppmi'].append(diff_ppmi)
    return pd.DataFrame(result)
