import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, log_loss, average_precision_score, roc_auc_score)


def classification_metrics(y, p):
    y, p = np.asarray(y).ravel(), np.clip(np.asarray(p).ravel(), 1e-7, 1-1e-7)
    pred = p >= .5
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return dict(accuracy=float(accuracy_score(y, pred)),
                precision=float(precision_score(y, pred, zero_division=0)),
                recall=float(recall_score(y, pred, zero_division=0)),
                specificity=float(tn / (tn + fp)) if tn+fp else None,
                f1=float(f1_score(y, pred, zero_division=0)),
                macro_f1=float(f1_score(y, pred, labels=[0, 1], average="macro", zero_division=0)),
                log_loss=float(log_loss(y, p, labels=[0, 1])),
                roc_auc=float(roc_auc_score(y,p)) if len(np.unique(y))==2 else None,
                average_precision=float(average_precision_score(y,p)) if np.any(y) else None,
                tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp), threshold=.5)


def top1_metrics(candidates, reference, p):
    pred = candidates[np.arange(len(candidates)), np.argmax(p, axis=1)]
    covered = (candidates == reference[:, None]).any(axis=1)
    return dict(top1_accuracy=float(np.mean(pred == reference)),
                candidate_miss_rate=float(np.mean(~covered)),
                covered_top1_accuracy=float(np.mean(pred[covered] == reference[covered])) if covered.any() else None)


def collapse(sequence):
    a = np.asarray(sequence)
    return a[np.r_[True, a[1:] != a[:-1]]]


def edit_similarity(a, b):
    a, b = list(a), list(b)
    row = list(range(len(b)+1))
    for i, x in enumerate(a, 1):
        nxt = [i]
        for j, y in enumerate(b, 1):
            nxt.append(min(nxt[-1]+1, row[j]+1, row[j-1]+(x != y)))
        row = nxt
    return 1 - row[-1] / max(len(a), len(b), 1)


def sequence_metrics(trip, predicted, reference, route):
    predicted = np.asarray(predicted)
    prev, nxt = predicted[:-1], predicted[1:]
    contiguous = (prev == nxt) | (trip.arcs[prev, 1] == trip.arcs[nxt, 0])
    # Strict adjacency penalizes valid multi-edge jumps; report separately from bridge success.
    ps, rs = set(predicted.tolist()), set(route.tolist())
    precision, recall = len(ps & rs) / max(len(ps), 1), len(ps & rs) / max(len(rs), 1)
    return dict(point_accuracy=float(np.mean(predicted == reference)),
                strict_adjacent_continuity=float(np.mean(contiguous)) if len(contiguous) else 1.,
                route_arc_precision=precision, route_arc_recall=recall,
                route_arc_f1=2*precision*recall/max(precision+recall,1e-12),
                sampled_sequence_edit_similarity=edit_similarity(collapse(predicted), collapse(reference)),
                full_route_edit_similarity=edit_similarity(collapse(predicted), collapse(route)))
