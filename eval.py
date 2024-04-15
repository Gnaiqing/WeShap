"""
Evaluate the quality of the LFs, label model, and the final labels.
"""
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix, roc_auc_score
from label_model import get_wrench_label_model
import torch
from scipy.special import comb
from sklearn.neighbors import KNeighborsClassifier
from snorkel.labeling.analysis import LFAnalysis


def evaluate_lfs(labels, L_train, lf_classes=None, n_class=2):
    # evaluate the coverage, accuracy of label functions
    n_lf = L_train.shape[1]
    n_active = np.sum(L_train != -1, axis=0)
    n_correct = np.sum(L_train == labels.reshape(-1, 1), axis=0)
    lf_covs = n_active / len(labels)
    lf_accs = np.divide(n_correct, n_active, out=np.repeat(np.nan, len(n_active)), where=n_active != 0)
    lf_cov_avg = np.mean(lf_covs)
    lf_acc_avg = np.mean(lf_accs[lf_covs != 0])
    # evaluate conflicts, overlaps
    lf_stats = LFAnalysis(L_train).lf_summary(Y=labels)
    lf_overlap_avg = lf_stats["Overlaps"].mean()
    lf_conflict_avg = lf_stats["Conflicts"].mean()
    results = {
        "n_lf": n_lf,
        "lf_cov_avg": lf_cov_avg,
        "lf_acc_avg": lf_acc_avg,
        "lf_overlap_avg": lf_overlap_avg,
        "lf_conflict_avg": lf_conflict_avg,
    }

    # evaluate the LF quality per class
    if lf_classes is not None:
        lf_num_pc = []
        lf_acc_avg_pc = []
        lf_cov_avg_pc = []
        lf_cov_total_pc = []
        for c in range(n_class):
            active_lfs = lf_classes == c
            lf_num_pc.append(np.sum(active_lfs))
            if np.sum(active_lfs) != 0:
                lf_acc_avg_pc.append(np.nanmean(lf_accs[active_lfs]))
                lf_cov_avg_pc.append(np.mean(lf_covs[active_lfs]))
                L_train_pc = L_train[:, active_lfs]
                cov_pc = np.mean(np.max(L_train_pc, axis=1) == c)
                lf_cov_total_pc.append(cov_pc)
            else:
                # no LF emits label for class c
                lf_acc_avg_pc.append(np.nan)
                lf_cov_avg_pc.append(np.nan)
                lf_cov_total_pc.append(np.nan)

        results["n_lf_per_class"] = lf_num_pc
        results["lf_acc_per_class"] = lf_acc_avg_pc
        results["lf_cov_per_class"] = lf_cov_avg_pc
        results["lf_cov_total_per_class"] = lf_cov_total_pc

    return results


def evaluate_labels(labels, preds, n_class=2):
    # Evaluate the prediction results. -1 in preds mean the label model rejects making prediction.
    covered_indices = preds != -1
    covered_labels = labels[covered_indices]
    covered_preds = preds[covered_indices]
    if -1 in covered_labels:
        # ground truth labels are missing
        results = {
            "coverage": np.sum(covered_indices) / len(preds),
            "accuracy": np.nan,
        }
        return results

    coverage = np.sum(covered_indices) / len(preds)
    accuracy = accuracy_score(covered_labels, covered_preds)

    average = "binary" if n_class == 2 else "macro"
    precision = precision_score(covered_labels, covered_preds, average=average)
    recall = recall_score(covered_labels, covered_preds, average=average)
    f1 = f1_score(covered_labels, covered_preds, average=average)

    # label distribution and predicted label distribution
    label_dist = np.zeros(n_class, dtype=float)
    pred_label_dist = np.zeros(n_class, dtype=float)
    for c in range(n_class):
        label_dist[c] = np.sum(labels == c) / len(labels)
        pred_label_dist[c] = np.sum(covered_preds == c) / len(covered_preds)
    # class-specific label quality
    precision_perclass = precision_score(covered_labels, covered_preds, average=None)
    recall_perclass = recall_score(covered_labels, covered_preds, average=None)
    f1_perclass = f1_score(covered_labels, covered_preds, average=None)
    # confusion matrix
    cm = confusion_matrix(covered_labels, covered_preds)
    results = {
        "coverage": coverage,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "label_distribution": label_dist.tolist(),
        "pred_distribution": pred_label_dist.tolist(),
        "per_class_precision": precision_perclass.tolist(),
        "per_class_recall": recall_perclass.tolist(),
        "per_class_f1": f1_perclass.tolist(),
        "confusion_matrix": cm
    }
    return results


def evaluate_disc_model(disc_model, test_dataset):
    if disc_model is None:
        return {
            "acc": np.nan,
            "auc": np.nan,
            "f1": np.nan
        }
    y_pred = disc_model.predict(test_dataset.features)
    y_probs = disc_model.predict_proba(test_dataset.features)
    test_acc = accuracy_score(test_dataset.labels, y_pred)
    if test_dataset.n_class == 2:
        test_auc = roc_auc_score(test_dataset.labels, y_probs[:, 1])
        test_f1 = f1_score(test_dataset.labels, y_pred)
    else:
        test_auc = roc_auc_score(test_dataset.labels, y_probs, average="macro", multi_class="ovo")
        test_f1 = f1_score(test_dataset.labels, y_pred, average="macro")

    results = {
        "acc": test_acc,
        "auc": test_auc,
        "f1": test_f1
    }
    return results


def calc_lf_score(pos, neg, n_class=2):
    """
    Calculate the value for a single LF using majority vote.
    pos and neg are the number of correct and wrong predictions.
    """
    if pos == 0 and neg == 0:
        return 0, 0
    if pos == 0:
        return 0, -1 / (n_class * neg)
    elif neg == 0:
        return (n_class - 1) / (n_class * pos), 0

    pos_score = 0
    for i in range(pos):
        for j in range(neg + 1):
            if i + j == 0:
                score = (n_class - 1) / n_class
            else:
                score = ((i + 1) / (i + j + 1) - i / (i + j))
            pos_score += score * comb(pos - 1, i, exact=True) * comb(neg, j, exact=True) / comb(pos + neg - 1, i + j,
                                                                                                exact=True)

    pos_score /= pos + neg
    neg_score = ((pos / (pos + neg)) - 1 / n_class - pos_score * pos) / neg
    return pos_score, neg_score


def calc_weshap_values(train_dataset, valid_dataset, k=5, distance="euclidean", weights="uniform"):
    """
    Calculate the WeShap values for each LF.
    """
    L_train = np.array(train_dataset.weak_labels)
    label_model = get_wrench_label_model("MV")
    label_model.fit(L_train)
    y_train_pred = label_model.predict(L_train)
    # calculate the k neighbors for each validation data point
    neigh = KNeighborsClassifier(n_neighbors=k, metric=distance, weights=weights)
    neigh.fit(train_dataset.features, y_train_pred)
    neigh_dist, neigh_idx = neigh.kneighbors(valid_dataset.features)
    # calculate the shapley score for each LF
    vote_count = np.zeros((len(train_dataset), train_dataset.n_class), dtype=int)
    for c in range(train_dataset.n_class):
        vote_count[:, c] = np.sum(L_train == c, axis=1)

    total_vote_count = np.sum(vote_count, axis=1)
    # record the score for an LF on each training data point
    lf_scores_pos = np.zeros((len(train_dataset), train_dataset.n_class))
    lf_scores_neg = np.zeros((len(train_dataset), train_dataset.n_class))
    lf_score_dict = {}
    for i in range(len(train_dataset)):
        for j in range(train_dataset.n_class):
            pos, neg = vote_count[i, j], total_vote_count[i] - vote_count[i, j]
            if (pos, neg) not in lf_score_dict:
                lf_score_dict[(pos, neg)] = calc_lf_score(pos, neg, n_class=train_dataset.n_class)
            pos_score, neg_score = lf_score_dict[(pos, neg)]
            lf_scores_pos[i, j] = pos_score
            lf_scores_neg[i, j] = neg_score

    # calculate the shapley score for each LF
    lf_shapeley_scores = np.zeros(L_train.shape[1])
    valid_labels = np.array(valid_dataset.labels)
    if weights == "uniform":
        for i in range(len(valid_dataset)):
            for train_idx in neigh_idx[i]:
                active_lf_idxs = np.where(L_train[train_idx] != -1)[0]
                for lf_idx in active_lf_idxs:
                    if L_train[train_idx, lf_idx] == valid_labels[i]:
                        lf_shapeley_scores[lf_idx] += lf_scores_pos[train_idx, valid_labels[i]]
                    else:
                        lf_shapeley_scores[lf_idx] += lf_scores_neg[train_idx, valid_labels[i]]

        lf_shapeley_scores /= len(valid_dataset) * k
    else:
        eps = 1e-6
        for i in range(len(valid_dataset)):
            weight_sum = np.sum(1 / (neigh_dist[i]+eps))
            for train_idx, dist in zip(neigh_idx[i], neigh_dist[i]):
                active_lf_idxs = np.where(L_train[train_idx] != -1)[0]
                weight = 1 / (dist+eps) / weight_sum
                for lf_idx in active_lf_idxs:
                    if L_train[train_idx, lf_idx] == valid_labels[i]:
                        lf_shapeley_scores[lf_idx] += lf_scores_pos[train_idx, valid_labels[i]] * weight
                    else:
                        lf_shapeley_scores[lf_idx] += lf_scores_neg[train_idx, valid_labels[i]] * weight

        lf_shapeley_scores /= len(valid_dataset)

    return lf_shapeley_scores


def approximate_label_model(L_aug, Y):
    # Convert NumPy arrays to PyTorch tensors
    L_aug = L_aug.reshape(len(L_aug), -1)
    L = torch.tensor(L_aug, dtype=torch.float32)
    Y = torch.tensor(Y, dtype=torch.float32)

    # Initialize W as a PyTorch tensor with requires_grad=True for optimization
    M, C = L.shape[1], Y.shape[1]
    W = torch.randn(M, C, requires_grad=True)

    # Define the optimizer
    optimizer = torch.optim.Adam([W], lr=0.01)

    # Number of iterations for the optimization loop
    num_epochs = 10000

    # Optimization loop with tqdm progress bar
    for epoch in range(num_epochs):
        optimizer.zero_grad()

        # Compute L @ W
        LW = torch.matmul(L, W)

        # Normalize each row of LW to sum to 1
        row_sums = LW.sum(dim=1, keepdim=True)
        normalized_LW = LW / row_sums

        # Compute the objective function (loss)
        loss = torch.norm(normalized_LW - Y)

        # Backpropagation
        loss.backward()

        # Print the loss every 100 epochs
        if epoch % 500 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}')

        # Update the weights
        optimizer.step()

    # Final optimized W
    W_optimized = W.detach().numpy().reshape(-1, C + 1, C)
    return W_optimized



