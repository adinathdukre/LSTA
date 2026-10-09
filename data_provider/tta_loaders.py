import os
import numpy as np
from natsort import natsorted
from data_provider.uea import normalize_batch_ts


def _fraction_splits(label, train_frac, val_frac):
    """Per-class fraction split, subjects in ascending id order within each
    class -- matches ADFTDLoader / PTBLoader / PTBXLLoader.load_train_val_test_list."""
    splits = {}
    for cls in np.unique(label[:, 0]):
        ids = list(label[label[:, 0] == cls][:, 1].astype(int))
        n = len(ids)    # number of subjects in this class
        train_ids = ids[: int(train_frac * n)]
        val_ids = ids[int(train_frac * n): int(val_frac * n)]
        test_ids = ids[int(val_frac * n):]
        for sid in train_ids:
            splits[sid] = "train"
        for sid in val_ids:
            splits[sid] = "val"
        for sid in test_ids:
            splits[sid] = "test"
    return splits


def _apava_splits(label):
    """APAVA splits used in TeCh"""
    val_ids, test_ids = {15, 16, 19, 20}, {1, 2, 17, 18}
    splits = {}
    for sid in label[:, 1].astype(int).tolist():
        splits[sid] = "val" if sid in val_ids else "test" if sid in test_ids else "train"
    return splits


def _tdbrain_splits(label):
    """TDBRAIN splits"""
    train_ids = list(range(1, 18)) + list(range(29, 46))
    val_ids = [18, 19, 20, 21, 46, 47, 48, 49]
    test_ids = [22, 23, 24, 25, 50, 51, 52, 53]
    splits = {}
    for sid in train_ids:
        splits[sid] = "train"
    for sid in val_ids:
        splits[sid] = "val"
    for sid in test_ids:
        splits[sid] = "test"
    return splits


# key is the dataset dircetory base name
SPLIT_RULES = {
    "ADFTD": lambda label: _fraction_splits(label, 0.6, 0.8),
    "PTB": lambda label: _fraction_splits(label, 0.55, 0.70),
    "PTB-XL": lambda label: _fraction_splits(label, 0.6, 0.8),
    "APAVA": _apava_splits,
    "TDBRAIN": _tdbrain_splits,
}


class _ClassSubjectPool:
    """Returns each splits windows for each subject in a class, keeping temporal order"""

    def __init__(self, root, split):
        dataset_name = os.path.basename(os.path.normpath(root))

        feature_dir = os.path.join(root, "Feature")
        label = np.load(os.path.join(root, "Label", "label.npy"))
        splits = SPLIT_RULES[dataset_name](label)
        filenames = natsorted(os.listdir(feature_dir))   #natural sort

        windows, labels, subjects = [], [], []
        for j, filename in enumerate(filenames):
            subject_id = j + 1
            if splits.get(subject_id) != split:    #ignore windows from subjects not in this split
                continue
            cls = int(label[j][0])
            X = np.load(os.path.join(feature_dir, filename))  # (n_windows, seq_len, channels), original order
            for w in X:
                windows.append(w)
                labels.append(cls)
                subjects.append(subject_id)

        X = normalize_batch_ts(np.array(windows, dtype=np.float32))
        y = np.array(labels)
        subj = np.array(subjects)

        self.classes = sorted(np.unique(y).tolist())
        self.X_by_class = {c: X[y == c] for c in self.classes}
        self.subj_by_class = {c: subj[y == c] for c in self.classes}
        self.seq_len = X.shape[1]
        self.enc_in = X.shape[2]

        self.subject_windows_by_class = {
            c: {s: np.where(self.subj_by_class[c] == s)[0] for s in np.unique(self.subj_by_class[c])}
            for c in self.classes
        }
        self.subjects_by_class = {c: list(self.subject_windows_by_class[c].keys()) for c in self.classes}

    def sample_one(self, c, rng):
        subject = self.subjects_by_class[c][rng.integers(len(self.subjects_by_class[c]))]
        window_idx = self.subject_windows_by_class[c][subject]
        idx = window_idx[rng.integers(len(window_idx))]
        return self.X_by_class[c][idx]
