import heapq  # for retrieval topK
import math
import numpy as np
from Dataset import MovieLensDataset

# Global variables that are shared across processes
_model = None
_testRatings = None
_testNegatives = None
_topk = None


def evaluate_model(model, full_dataset: MovieLensDataset, topK: int):
    """
    Evaluate the performance (Hit_Ratio, NDCG) of top-K recommendation
    Return: score of each test rating.
    """
    global _model
    global _testRatings
    global _testNegatives
    global _topk
    _model = model
    _testRatings = full_dataset.testRatings
    _testNegatives = full_dataset.testNegatives
    _topk = topK

    hits, ndcgs = [], []
    for idx in range(len(_testRatings)):
        (hr, ndcg) = eval_one_rating(idx, full_dataset)
        hits.append(hr)
        ndcgs.append(ndcg)
    return (hits, ndcgs)


def eval_one_rating(idx, full_dataset: MovieLensDataset):
    rating = _testRatings[idx]
    items = _testNegatives[idx]
    u = rating[0]

    gtItem = rating[1]
    items.append(gtItem)
    # Get prediction scores
    map_item_score = {}
    users = np.full(len(items), u, dtype="int32")

    feed_dict = {
        "user_id": users,
        "item_id": np.array(items),
    }
    predictions = _model.predict(feed_dict)
    for i in range(len(items)):
        item = items[i]
        map_item_score[item] = predictions[i]

    # Evaluate top rank list
    ranklist = heapq.nlargest(_topk, map_item_score, key=map_item_score.get)
    hr = getHitRatio(ranklist, gtItem)
    ndcg = getNDCG(ranklist, gtItem)
    return (hr, ndcg)


def getHitRatio(ranklist, gtItem):
    for item in ranklist:
        if item == gtItem:
            return 1
    return 0


def getNDCG(ranklist, gtItem):
    for i in range(len(ranklist)):
        item = ranklist[i]
        if item == gtItem:
            return math.log(2) / math.log(i + 2)
    return 0
