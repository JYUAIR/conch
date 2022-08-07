import torch
from functools import partial
import inspect
import numpy as np
import random
import bottleneck as bn

class Metrics(object):
    @staticmethod
    def compute(pred_scores, ground_truth, metrics_list):
        results = {}
        for metric in metrics_list:
            try:
                if "@" in metric:
                    met, k = metric.split("@")
                    met_foo = getattr(Metrics, "%s_at_k" % met.lower())
                    results[metric] = met_foo(pred_scores, ground_truth, int(k))
                else:
                    results[metric] = getattr(Metrics, metric)(pred_scores, ground_truth)
            except AttributeError:
                print("Skipped unknown metric '%s'.", metric)
        return results

    @staticmethod
    def ndcg_at_k(pred_scores, ground_truth, k=100):
        assert pred_scores.shape == ground_truth.shape, \
            "'pred_scores' and 'ground_truth' must have the same shape."
        k = min(pred_scores.shape[1], k)
        n_users = pred_scores.shape[0]
        idx_topk_part = bn.argpartition(-pred_scores, k - 1, axis=1)
        topk_part = pred_scores[np.arange(n_users)[:, np.newaxis], idx_topk_part[:, :k]]
        idx_part = np.argsort(-topk_part, axis=1)
        idx_topk = idx_topk_part[np.arange(n_users)[:, np.newaxis], idx_part]
        tp = 1. / np.log2(np.arange(2, k + 2))
        DCG = (ground_truth[np.arange(n_users)[:, np.newaxis], idx_topk] * tp).sum(axis=1)
        IDCG = np.array([(tp[:min(int(n), k)]).sum() for n in ground_truth.sum(axis=1)])
        return DCG / IDCG

    @staticmethod
    def recall_at_k(pred_scores, ground_truth, k=100):
        assert pred_scores.shape == ground_truth.shape, \
            "'pred_scores' and 'ground_truth' must have the same shape."
        k = min(pred_scores.shape[1], k)
        idx = bn.argpartition(-pred_scores, k - 1, axis=1)
        pred_scores_binary = np.zeros_like(pred_scores, dtype=bool)
        pred_scores_binary[np.arange(pred_scores.shape[0])[:, np.newaxis], idx[:, :k]] = True
        X_true_binary = (ground_truth > 0)
        num = (np.logical_and(X_true_binary, pred_scores_binary).sum(axis=1)).astype(np.float32)
        recall = num / np.minimum(k, X_true_binary.sum(axis=1))
        return recall

    @staticmethod
    def hit_at_k(pred_scores, ground_truth, k=100):
        assert pred_scores.shape == ground_truth.shape, \
            "'pred_scores' and 'ground_truth' must have the same shape."
        k = min(pred_scores.shape[1], k)
        idx = bn.argpartition(-pred_scores, k - 1, axis=1)
        pred_scores_binary = np.zeros_like(pred_scores, dtype=bool)
        pred_scores_binary[np.arange(pred_scores.shape[0])[:, np.newaxis], idx[:, :k]] = True
        X_true_binary = (ground_truth > 0)
        num = (np.logical_and(X_true_binary, pred_scores_binary).sum(axis=1)).astype(np.float32)
        return num > 0


class ValidFunc(object):
    def __init__(self, func, **kwargs):
        self.func_name = func.__name__
        self.function = partial(func, **kwargs)

        args = inspect.getfullargspec(self.function).args
        assert args == ["model", "test_loader", "metric_list"], \
            "A (partial) validation function must have the following kwargs: model, test_loader and\
            metric_list"

    def __call__(self, model, test_loader, metric):
        return self.function(model, test_loader, [metric])[metric]

    def __str__(self):
        kwdefargs = inspect.getfullargspec(self.function).kwonlydefaults
        return "ValidFunc(fun='%s', params=%s)" % (self.func_name, kwdefargs)

    def __repr__(self):
        return str(self)


def logic_evaluate(model, test_loader, metric_list):
    results = {m: [] for m in metric_list}
    for batch_idx, batch_data in enumerate(test_loader):
        positive_pred, negative_pred = model.predict(batch_data)
        # we concatenate the positive prediction to the negative predictions
        # in each row of the final tensor we will have the positive prediction in the first column
        # and the 100 negative predictions in the last 100 columns
        positive_pred = positive_pred.view(positive_pred.size(0), 1)
        pred_scores = torch.cat((positive_pred, negative_pred), dim=1)
        # now, we need to construct the ground truth tensor
        ground_truth = np.zeros(pred_scores.size())
        ground_truth[:, 0] = 1  # the positive item is always in the first column of pred_scores, as we said before
        pred_scores = pred_scores.cpu().numpy()
        res = Metrics.compute(pred_scores, ground_truth, metric_list)
        for m in res:
            results[m].append(res[m])

    for m in results:
        results[m] = np.concatenate(results[m])
    return results


def evaluate(model, test_loader, metric_list):
    results = {m: [] for m in metric_list}
    for _, (data_tr, heldout) in enumerate(test_loader):
        data_tensor = data_tr.view(data_tr.shape[0], -1)
        recon_batch = model.predict(data_tensor)[0].cpu().numpy()
        heldout = heldout.view(heldout.shape[0], -1).cpu().numpy()
        res = Metrics.compute(recon_batch, heldout, metric_list)
        for m in res:
            results[m].append(res[m])

    for m in results:
        results[m] = np.concatenate(results[m])
    return results


def one_plus_random(model, test_loader, metric_list, r=1000):
    results = {m: [] for m in metric_list}
    for _, (data_tr, heldout) in enumerate(test_loader):
        tot = set(range(heldout.shape[1]))
        data_tensor = data_tr.view(data_tr.shape[0], -1)
        recon_batch = model.predict(data_tensor)[0].cpu().numpy()
        heldout = heldout.view(heldout.shape[0], -1).cpu().numpy()

        users, items = heldout.nonzero()
        rows = []
        for u, i in zip(users, items):
            rnd = random.sample(tot - set(list(heldout[u].nonzero()[0])), r)
            rows.append(list(recon_batch[u][[i] + list(rnd)]))

        pred = np.array(rows)
        ground_truth = np.zeros_like(pred)
        ground_truth[:, 0] = 1
        res = Metrics.compute(pred, ground_truth, metric_list)
        for m in res:
            results[m].append(res[m])

    for m in results:
        results[m] = np.concatenate(results[m])
    return results
