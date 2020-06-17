
def normalize_results(results):

    # get max and min scores
    scores = []
    for chart in results:
        for dimension in results[chart]:
            scores.append(results[chart][dimension]['score'])
    score_max = max(scores)
    score_min = min(scores)

    # normalize scores
    for chart in results:
        for dimension in results[chart]:
            score = results[chart][dimension]['score']
            score_norm = round((score - score_min) / (score_max - score_min), 4)
            results[chart][dimension]['score_norm'] = score_norm

    return results

