import os
import json
import numpy as np
import torch

from tqdm import tqdm

rel2id = json.load(open('data/DocRED/label_map.json', 'r'))
id2rel = {value: key for key, value in rel2id.items()}
rel2name = json.load(open('data/DocRED/rel_info.json', 'r'))


def to_official(preds, features, threshold):

    h_idx, t_idx, title = [], [], []

    for f in features:
        pairs = f["ent_pairs"]
        h_idx += [pair[0] for pair in pairs]
        t_idx += [pair[1] for pair in pairs]
        title += [f["title"] for _ in pairs]

    res = []
    with tqdm(desc="Converting to submission format", total=len(preds), ncols=100) as pbar:
        for i in range(preds.shape[0]):
            pred = preds[i]
            if threshold == -1:     # assume atlop if threhold is -1
                pred = torch.nonzero(pred, as_tuple=True)[0].tolist()
            else:
                pred = (pred >= threshold).nonzero(as_tuple=True)[0].tolist()
            for p in pred:
                res.append({
                    'title': title[i],
                    'h_idx': h_idx[i],
                    't_idx': t_idx[i],
                    'r': id2rel[p+1],   # need to skip "no relation" label
                })
            pbar.update(1)

    return res


def gen_train_facts(data_file_name, truth_dir):
    fact_file_name = data_file_name[data_file_name.find("train"):]
    fact_file_name = os.path.join(truth_dir, fact_file_name.replace(".json", ".fact"))

    if os.path.exists(fact_file_name):
        fact_in_train = set([])
        triples = json.load(open(fact_file_name))
        for x in triples:
            fact_in_train.add(tuple(x))
        return fact_in_train

    fact_in_train = set([])
    ori_data = json.load(open(data_file_name))
    for data in ori_data:
        vertexSet = data['vertexSet']
        for label in data['labels']:
            rel = label['r']
            for n1 in vertexSet[label['h']]:
                for n2 in vertexSet[label['t']]:
                    fact_in_train.add((n1['name'], n2['name'], rel))

    json.dump(list(fact_in_train), open(fact_file_name, "w"))

    return fact_in_train


def official_evaluate(tmp, data_dir, split):

    truth_dir = os.path.join(data_dir, 'ref')

    if not os.path.exists(truth_dir):
        os.makedirs(truth_dir)

    fact_in_train_annotated = gen_train_facts(os.path.join(data_dir, "train.json"), truth_dir)
    fact_in_train_distant = gen_train_facts(os.path.join(data_dir, "train_distant.json"), truth_dir)

    truth = json.load(open(os.path.join(data_dir, f"{split}.json")))

    std = {}
    tot_evidences = 0
    titleset = set([])

    title2vertexSet = {}

    for x in truth:
        title = x['title']
        titleset.add(title)

        vertexSet = x['vertexSet']
        title2vertexSet[title] = vertexSet

        for label in x['labels']:
            r = label['r']
            h_idx = label['h']
            t_idx = label['t']
            std[(title, r, h_idx, t_idx)] = set(label['evidence'])
            tot_evidences += len(label['evidence'])

    tot_relations = len(std)
    tmp.sort(key=lambda x: (x['title'], x['h_idx'], x['t_idx'], x['r']))
    submission_answer = [tmp[0]]
    for i in range(1, len(tmp)):
        x = tmp[i]
        y = tmp[i - 1]
        if (x['title'], x['h_idx'], x['t_idx'], x['r']) != (y['title'], y['h_idx'], y['t_idx'], y['r']):
            submission_answer.append(tmp[i])
    
    correct_re = 0
    correct_evidence = 0
    pred_evi = 0

    correct_in_train_annotated = 0
    correct_in_train_distant = 0
    titleset2 = set([])

    with tqdm("Calculating official scores", total=len(submission_answer), ncols=100) as pbar:
        for x in submission_answer:
            title = x['title']
            h_idx = x['h_idx']
            t_idx = x['t_idx']
            r = x['r']
            titleset2.add(title)
            if title not in title2vertexSet:
                continue
            vertexSet = title2vertexSet[title]

            if 'evidence' in x:
                evi = set(x['evidence'])
            else:
                evi = set([])
            pred_evi += len(evi)

            if (title, r, h_idx, t_idx) in std:
                correct_re += 1
                stdevi = std[(title, r, h_idx, t_idx)]
                correct_evidence += len(stdevi & evi)
                in_train_annotated = in_train_distant = False
                for n1 in vertexSet[h_idx]:
                    for n2 in vertexSet[t_idx]:
                        if (n1['name'], n2['name'], r) in fact_in_train_annotated:
                            in_train_annotated = True
                        if (n1['name'], n2['name'], r) in fact_in_train_distant:
                            in_train_distant = True

                if in_train_annotated:
                    correct_in_train_annotated += 1
                if in_train_distant:
                    correct_in_train_distant += 1
            pbar.update(1)

    re_p = 1.0 * correct_re / len(submission_answer)
    re_r = 1.0 * correct_re / tot_relations
    if re_p + re_r == 0:
        re_f1 = 0
    else:
        re_f1 = 2.0 * re_p * re_r / (re_p + re_r)

    evi_p = 1.0 * correct_evidence / pred_evi if pred_evi > 0 else 0
    evi_r = 1.0 * correct_evidence / tot_evidences
    if evi_p + evi_r == 0:
        evi_f1 = 0
    else:
        evi_f1 = 2.0 * evi_p * evi_r / (evi_p + evi_r)

    re_p_ignore_train_annotated = 1.0 * (correct_re - correct_in_train_annotated) / (len(submission_answer) - correct_in_train_annotated + 1e-5)
    re_p_ignore_train = 1.0 * (correct_re - correct_in_train_distant) / (len(submission_answer) - correct_in_train_distant + 1e-5)

    if re_p_ignore_train_annotated + re_r == 0:
        re_f1_ignore_train_annotated = 0
    else:
        re_f1_ignore_train_annotated = 2.0 * re_p_ignore_train_annotated * re_r / (re_p_ignore_train_annotated + re_r)

    if re_p_ignore_train + re_r == 0:
        re_f1_ignore_train = 0
    else:
        re_f1_ignore_train = 2.0 * re_p_ignore_train * re_r / (re_p_ignore_train + re_r)

    return re_f1, evi_f1, re_f1_ignore_train_annotated, re_f1_ignore_train


def unofficial_evaluate(preds, labels, dataset_name="DocRED"):
    score_dict = {}
    best_theta = -1

    # need to find optimal threshold where f1 is maximized
    sorted_logits, sorted_idxes = preds.flatten().sort(descending=True)
    sorted_labels = torch.gather(labels.flatten(), dim=0, index=sorted_idxes)
    predictions = torch.ones_like(sorted_logits).to(sorted_logits)
    num_preds = predictions.cumsum(0)
    num_labels = labels.sum()
    num_matches = (predictions * sorted_labels).cumsum(0)
    precisions = num_matches / num_preds
    recalls = num_matches / num_labels
    f1s = 2 * precisions * recalls / (precisions + recalls + 1e-20)

    f1, best_f1_pos = f1s.max(0)
    precision = precisions[best_f1_pos]
    recall = recalls[best_f1_pos]
    num_matches = num_matches[best_f1_pos]
    num_preds = num_preds[best_f1_pos]
    best_theta = sorted_logits[best_f1_pos].item()

    num_preds_per_class = (preds >= best_theta).sum(0)
    num_matches_per_class = (labels * (preds >= best_theta)).sum(0)
    num_labels_per_class = labels.sum(0)

    # Calculate macro F1
    precision_per_class = num_matches_per_class / (num_preds_per_class + 1e-20)
    recall_per_class = num_matches_per_class / (num_labels_per_class + 1e-20)
    f1_per_class = 2 * precision_per_class * recall_per_class / (precision_per_class + recall_per_class + 1e-20)
    macro_f1 = f1_per_class.mean()
    
    # class frequency
    if dataset_name == "DocRED":
        label_freq = torch.tensor([264, 8921, 4193, 2004, 2689, 1044, 511, 79, 475, 79, 275, 356, 172, 76, 194, 539, 35, 583, 632, 414, 1052, 1142, 621, 95, 203, 316, 805, 196, 173, 210, 596, 85, 303, 74, 273, 360, 119, 155, 150, 238, 304, 104, 406, 96, 62, 335, 298, 246, 156, 82, 188, 192, 166, 108, 208, 185, 23, 163, 144, 299, 231, 152, 79, 63, 223, 110, 51, 36, 379, 320, 48, 111, 85, 137, 119, 191, 140, 144, 33, 66, 9, 77, 103, 95, 100, 172, 83, 92, 92, 2, 75, 36, 36, 18, 2, 4]).to(f1_per_class)
    elif dataset_name == "Re-DocRED":
        label_freq = torch.tensor([263, 14401, 20402, 3369, 4665, 1172, 692, 155, 868, 181, 575, 761, 336, 178, 431, 948, 66, 923, 2313, 1299, 1773, 1621, 919, 200, 281, 503, 1000, 421, 340, 368, 2112, 178, 640, 168, 466, 703, 281, 366, 3055, 402, 460, 204, 403, 191, 102, 712, 1207, 341, 237, 152, 506, 506, 305, 191, 389, 356, 49, 370, 245, 669, 410, 264, 171, 145, 1168, 222, 105, 79, 379, 489, 83, 239, 174, 293, 249, 1168, 292, 357, 59, 107, 22, 152, 225, 192, 204, 298, 144, 230, 230, 2, 117, 65, 96, 96, 8, 8]).to(f1_per_class)
    elif dataset_name == "DWIE":
        label_freq = torch.tensor([83, 133, 470, 1403, 751, 1572, 1518, 307, 291, 211, 193, 1255, 2005, 1597, 137, 184, 170, 1703, 1206, 158, 361, 326, 68, 5, 11, 99, 18, 242, 253, 51, 57, 367, 32, 123, 30, 4, 21, 126, 87, 16, 43, 25, 7, 27, 6, 16, 16, 16, 11, 43, 30, 12, 7, 5, 9, 2, 2, 3, 0, 2, 1, 0, 0, 1, 1]).to(f1_per_class)
    
    macro_f1_at_500 = f1_per_class[label_freq < 500].mean()
    macro_f1_at_200 = f1_per_class[label_freq < 200].mean()
    macro_f1_at_100 = f1_per_class[label_freq < 100].mean()

    score_dict["P"] = precision.item()
    score_dict["R"] = recall.item()
    score_dict["F1"] = f1.item()
    score_dict["macro_F1"] = macro_f1.item()
    score_dict["macro_F1_at_500"] = macro_f1_at_500.item()
    score_dict["macro_F1_at_200"] = macro_f1_at_200.item()
    score_dict["macro_F1_at_100"] = macro_f1_at_100.item()
    score_dict["F1_per_class"] = f1_per_class.tolist()
    score_dict["num_matches"] = num_matches.long().item()
    score_dict["num_preds"] = num_preds.long().item()
    score_dict["num_labels"] = num_labels.long().item()
    score_dict["theta"] = best_theta

    return score_dict


def calibrate(logits, labels, preds=None):
    _logits = logits.flatten().cpu().numpy()
    _labels = labels.flatten().cpu().numpy()

    N = len(_logits)    # total sample size
    _, num_labels = logits.shape
    n_bins = 10
    bins = np.linspace(0.0, 1.0, n_bins + 1)

    # ECE & reliability diagram for ALL
    binids = np.searchsorted(bins[1:-1], _logits)
    bin_sums = np.bincount(binids, weights=_logits, minlength=len(bins))
    bin_true = np.bincount(binids, weights=_labels, minlength=len(bins))
    bin_total = np.bincount(binids, minlength=len(bins))
    nonzero = bin_total != 0
    prob_true = bin_true[nonzero] / bin_total[nonzero]
    prob_pred = bin_sums[nonzero] / bin_total[nonzero]
    ece = ((bin_total[nonzero] / N) * abs(prob_true - prob_pred)).sum()

    # ACE
    ace_list = []
    for k in range(num_labels):
        _class_logits = logits[:, k].cpu().numpy()
        _class_labels = labels[:, k].cpu().numpy()
        if preds is not None:
            _class_preds = preds[:, k].cpu().numpy()

        even_bins = np.percentile(_class_logits, bins * 100)
        _class_binids_even = np.searchsorted(even_bins[1:-1], _class_logits)

        if preds is not None:
            _class_bin_sums_even = np.bincount(_class_binids_even, weights=_class_preds, minlength=len(even_bins))
        else:
            _class_bin_sums_even = np.bincount(_class_binids_even, weights=_class_logits, minlength=len(even_bins))
        _class_bin_true_even = np.bincount(_class_binids_even, weights=_class_labels, minlength=len(even_bins))
        _class_bin_total_even = np.bincount(_class_binids_even, minlength=len(even_bins))
        _nonzero_even = _class_bin_total_even != 0
        _class_prob_true_even = _class_bin_true_even[_nonzero_even] / _class_bin_total_even[_nonzero_even]
        _class_prob_pred_even = _class_bin_sums_even[_nonzero_even] / _class_bin_total_even[_nonzero_even]
        _class_ace_score = abs(_class_prob_true_even - _class_prob_pred_even).mean()
        ace_list.append(_class_ace_score)

    ace = sum(ace_list) / len(ace_list)

    return ece, ace, prob_true, prob_pred
