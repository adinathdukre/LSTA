import argparse
import math
import os


os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")    #need to set before import torch

import numpy as np
import torch
torch.use_deterministic_algorithms(True)
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, average_precision_score)

from data_provider.tta_loaders import _ClassSubjectPool
from data_provider.subject_windows import SubjectIndex
from models.TeCh import Model as RawTeChModel


def compute_metrics(trues, preds, probs, num_class):
    trues = np.array(trues)
    preds = np.array(preds)
    probs = np.array(probs)
    trues_onehot = np.eye(num_class)[trues]

    metrics = {
        "Accuracy": accuracy_score(trues, preds),
        "Precision": precision_score(trues, preds, average="macro", zero_division=0),
        "Recall": recall_score(trues, preds, average="macro", zero_division=0),
        "F1": f1_score(trues, preds, average="macro", zero_division=0)}
    metrics["AUROC"] = roc_auc_score(trues_onehot, probs, multi_class="ovr")
    metrics["AUPRC"] = average_precision_score(trues_onehot, probs, average="macro")
    metrics["avg"] = np.mean([v for k, v in metrics.items() if k != "avg"])
    metrics["n_windows"] = len(trues)
    return metrics

_NORM_AFFINE = {
    "LayerNorm":   ("weight", "bias"),
    "RMSNorm":     ("weight",),        # scale only, no shift
    "LayerNorm1D": ("weight", "bias"),
    "TTTLinear":   ("ln_w", "ln_b"),   # inner-loop LN of the fast-weight update
}


def build_ln_metadata(model):
    """ find norm-layer affine parameters in the model in a deterministic order"""
    names = []
    pd_all = dict(model.named_parameters())
    for name, m in model.named_modules():
        attrs = _NORM_AFFINE.get(type(m).__name__)
        if attrs is None:
            continue
        for a in attrs:
            full = f"{name}.{a}" if name else a
            if full in pd_all:
                names.append(full)
    pd = dict(model.named_parameters())
    seen_ids = set()
    metas = []
    offset = 0
    for pname in names:
        p = pd[pname]
        if id(p) in seen_ids:
            raise RuntimeError(f"tied/shared LN parameter detected: {pname}")
        seen_ids.add(id(p))
        length = p.numel()
        metas.append(dict(name=pname, shape=tuple(p.shape), offset=offset, length=length))
        offset += length
    return metas, offset


def flatten_ln(model, metas):
    '''flatten all layer norm parameters in to a single vector'''
    pd = dict(model.named_parameters())
    parts = [pd[m["name"]].detach().reshape(-1) for m in metas]
    return torch.cat(parts)


# reconstruct layer norm parameters from a single vector phi, using the metadata to know where each parameter is
# for forward pass
unflatten_ln = lambda phi, metas: {m["name"]: phi[m["offset"]: m["offset"] + m["length"]].reshape(m["shape"]) for m in metas}  
effective_basis = lambda W: math.sqrt(W.shape[1]) * W / W.norm().clamp_min(1e-12)  #normalize W


def qr_retract(W):
    """ take back W to orthonormal columns by QR decompostions"""
    Q, R = torch.linalg.qr(W)
    return Q * torch.sign(torch.diagonal(R)).unsqueeze(0)


def forward_with_phi(model, x, phi, metas, base_cache):
    '''forward pass with a given phi vector (layer norm parameters)'''
    merged = dict(base_cache)
    merged.update(unflatten_ln(phi, metas))
    return functional_call(model, merged, (x,))    # returns the model with updated parameters(functional_call)


def entropy_from_logits(logits, T=1.0):
    '''compute the mean entropy of the softmax distribution from logits'''
    logp = F.log_softmax(logits / T, dim=-1)
    return -(logp.exp() * logp).sum(dim=-1).mean()



def source_entropy_gradient(model, metas, base_cache, phi0, support_x, T=1.0):
    '''compute the gradient of the mean entropy of the support set with respect to phi (the layer norm parameters) at phi0.'''
    with torch.enable_grad():
        phi_leaf = phi0.detach().clone().requires_grad_(True)
        logits = forward_with_phi(model, support_x, phi_leaf, metas, base_cache)
        loss_s = entropy_from_logits(logits, T)
        (g,) = torch.autograd.grad(loss_s, phi_leaf, create_graph=False)   # gradients for phi_leaf
    return g.detach()



def episode_loss(model, metas, base_cache, W, eta, phi0, support_x, query_x, query_y, T=1.0):
    '''one SGD adaptation step on support set and compute the loss on the query set'''
    g = source_entropy_gradient(model, metas, base_cache, phi0, support_x, T)  # gradient wrt to phi0 (D x 1)
    B = effective_basis(W)   #normalize learned basis matrix(B)
    z_plus = -eta * (B.T @ g)   # update in low rank | B_T(r x D) @ g(D x 1) = r x 1
    delta_phi = B @ z_plus     # move update to full space D dimension
    query_logits = forward_with_phi(model, query_x, phi0 + delta_phi, metas, base_cache)
    loss = F.cross_entropy(query_logits, query_y)
    return loss, g, z_plus, delta_phi


def sample_support_indices(rng, pool_size, k, exclude_first=False):
    n = pool_size
    if n <= 1:
        eligible = np.arange(n)
    elif exclude_first:
        eligible = np.arange(1, n)
    else:
        eligible = np.arange(0, n - 1)
    kk = min(k, len(eligible))
    return eligible[rng.permutation(len(eligible))[:kk]]


def sample_train_episode(train_idx, rng, k, max_query, device):
    '''for a given training index, randomly select a subject and sample k support windows and max_query query windows.
    Randomly swap support and query pools.'''
    s = train_idx.subjects[rng.integers(len(train_idx.subjects))]  # randomly select a subject from the training index
    c = train_idx.subject_to_class[s]
    sp, qp = train_idx.support_pool[s], train_idx.query_pool[s]
    swap = rng.integers(2) == 0
    support_pool_x, query_pool_x = (sp, qp) if swap else (qp, sp)  #swap support and query pool randomly
    if support_pool_x.shape[0] < 1 or query_pool_x.shape[0] < 1:  #remember for PTB-XL only 10 windos per patient
        return None
    sel_s = sample_support_indices(rng, support_pool_x.shape[0], k, exclude_first=not swap)
    nq = min(max_query, query_pool_x.shape[0])
    sel_q = rng.permutation(query_pool_x.shape[0])[:nq]
    support_x = torch.from_numpy(support_pool_x[sel_s]).float().to(device)
    query_x = torch.from_numpy(query_pool_x[sel_q]).float().to(device)
    query_y = torch.full((query_x.shape[0],), c, dtype=torch.long, device=device)
    return support_x, query_x, query_y


class Evaluator:

    def __init__(self, data_root, ckpt_path, d_model, num_class, v_layer, t_layer,
                 patch_len, augmentations, gpu=0, model_cls=None, extra_cfg=None):
        self.device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
        self.data_root = data_root
        self.num_class = num_class

        train_pool = _ClassSubjectPool(data_root, "train")
        self.model_cls = model_cls if model_cls is not None else RawTeChModel                # To follow the same approach on medts-ttt
        self.configs = argparse.Namespace(
            seq_len=train_pool.seq_len, enc_in=train_pool.enc_in, d_model=d_model,
            t_layer=t_layer, v_layer=v_layer, dropout=0.0, patch_len=patch_len,
            num_class=num_class, augmentations=augmentations,
        )
        for k, v in (extra_cfg or {}).items():    # backbone-specific fields (e.g. MedTS-TTT patch)
            setattr(self.configs, k, v)
        self.base_state = torch.load(ckpt_path, map_location="cpu")
        self._pool_cache = {}    # to avoid reloading _ClassSubjectPool/SubjectIndex per split
        self._baseline_cache = {}  # to avoid recomputing baseline metrics for the same split (no TTA)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        self.frozen = self._fresh_model()
        self.frozen.eval()
        for p in self.frozen.parameters():
            p.requires_grad = False

        self.metas, self.D = build_ln_metadata(self.frozen)
        self.phi0 = flatten_ln(self.frozen, self.metas).to(self.device)
        self.base_cache = {**dict(self.frozen.named_parameters()), **dict(self.frozen.named_buffers())}

    def _fresh_model(self):
        m = self.model_cls(self.configs).to(self.device)
        m.load_state_dict(self.base_state)
        return m

    def _pool(self, split):
        '''Get the _ClassSubjectPool and SubjectIndex for a given split, caching them to avoid reloading.'''
        if split not in self._pool_cache:
            pool = _ClassSubjectPool(self.data_root, split)
            idx = SubjectIndex(pool)
            self._pool_cache[split] = (pool, idx)
        return self._pool_cache[split]

    def restrict_val_subjects(self, n_subjects):
        '''restirct val set to faster grid search on large datasets'''
        pool, idx = self._pool("val")
        if len(idx.subjects) <= n_subjects:
            return
        rng = np.random.default_rng(0)
        idx.subjects = sorted(rng.choice(idx.subjects, size=n_subjects, replace=False).tolist())

    def _score(self, model, X):
        preds, probs = [], []
        with torch.no_grad():
            for i in range(0, len(X), 512):   # batching to avoid OOM on long recordings
                xb = torch.from_numpy(X[i:i + 512]).float().to(self.device)
                logits = model(xb)
                pr = F.softmax(logits, dim=-1).cpu().numpy()
                preds.extend(logits.argmax(-1).cpu().numpy().tolist())
                probs.extend(pr.tolist())
        return preds, probs

    def baseline(self, split):
        """Obtain the results for the baseline (no TTA)"""
        if split in self._baseline_cache:    #if results already computed use those
            return self._baseline_cache[split]
        pool, idx = self._pool(split)

        trues_a, preds_a, probs_a = [], [], []
        for c in pool.classes:
            X = pool.X_by_class[c]
            p, pr = self._score(self.frozen, X)
            preds_a += p; probs_a += pr; trues_a += [c] * len(X)
        full = compute_metrics(trues_a, preds_a, probs_a, self.num_class)

        result = dict(full=full)
        self._baseline_cache[split] = result
        return result

    # -- meta_ln_basis: outer (meta) training ------------------------------
    def _bidirectional_eval(self, split, W, eta, k, n_draws, adapt_steps=1, T=1.0, subjects=None):
        """Evaluate the meta-trained basis on a given split.
              adapt_steps: number of adaptation steps.
              subjects: if given, restrict evaluation to this subset of the split's subjects
                (e.g. a fixed subsample used during the grid-search phase on a large val set
                like PTB-XL's); defaults to the full split's subject list."""
        pool, idx = self._pool(split)
        subj_list = subjects if subjects is not None else idx.subjects
        draw_metrics = []
        for draw in range(n_draws):
            rng = np.random.default_rng(1000 + draw)   # 1000+draw to stop collide with training episode seeds
            trues, preds, probs = [], [], []
            for s in subj_list:
                sp, qp = idx.support_pool[s], idx.query_pool[s]
                c = idx.subject_to_class[s]
                for support_pool_x, query_x, excl_first in ((sp, qp, False), (qp, sp, True)):  #run interchangebky
                    sel = sample_support_indices(rng, support_pool_x.shape[0], k, exclude_first=excl_first)
                    support_x = torch.from_numpy(support_pool_x[sel]).float().to(self.device)
                    with torch.no_grad():
                        B = effective_basis(W)
                        phi_cur = self.phi0
                        for _ in range(adapt_steps):
                            g = source_entropy_gradient(self.frozen, self.metas, self.base_cache,
                                                        phi_cur, support_x, T)
                            z_plus = -eta * (B.T @ g)
                            delta_phi = B @ z_plus
                            phi_cur = phi_cur + delta_phi
                        logits = forward_with_phi(self.frozen, torch.from_numpy(query_x).float().to(self.device),
                                                   phi_cur, self.metas, self.base_cache)
                        pr = F.softmax(logits, dim=-1).cpu().numpy()
                        pd_ = logits.argmax(-1).cpu().numpy()
                    preds.extend(pd_.tolist()); probs.extend(pr.tolist()); trues.extend([c] * len(query_x))
            draw_metrics.append(compute_metrics(trues, preds, probs, self.num_class))
        keys = [key for key in draw_metrics[0] if key != "n_windows"]
        return {key: (float(np.mean([d[key] for d in draw_metrics])),
                      float(np.std([d[key] for d in draw_metrics]))) for key in keys}

    def train_basis(self, rank, k, eta, outer_steps, outer_lr=1e-3, outer_batch=4,
                         eval_every=50, n_val_draws=5, basis_seed=0, episode_seed=1,
                         adapt_steps_grid=(1,), log_fn=print, T=1.0, retract=True):

        if self.D < rank:
            raise ValueError(f"D={self.D} < rank={rank}")
        train_pool, train_idx = self._pool("train")

        torch.manual_seed(basis_seed)
        W_raw = torch.randn(self.D, rank, device=self.device)
        Wq, _ = torch.linalg.qr(W_raw)
        W = nn.Parameter(Wq[:, :rank].clone())     # learable basis matrix (D x rank) -> D is num layer norm params
        W_init = W.detach().clone()

        opt = torch.optim.Adam([W], lr=outer_lr, weight_decay=0.0)
        ep_rng = np.random.default_rng(episode_seed)

        def eval_all_adapt_steps(W_eval):
            """val results per adapt_steps"""
            val_out = {}
            for a in adapt_steps_grid:
                val_out[a] = self._bidirectional_eval("val", W_eval, eta, k, n_val_draws, a, T=T)
            return val_out


        trajectory = []
        best_W, best_step, best_adapt_steps = W_init.clone(), None, adapt_steps_grid[0]
        best_f1, best_score = None, None  # both set from val F1 below; 

        for step in range(1, outer_steps + 1):
            opt.zero_grad(set_to_none=True)
            n_ok, total_loss = 0, 0.0
            for _ in range(outer_batch):
                ep = sample_train_episode(train_idx, ep_rng, k, 32, self.device)
                if ep is None:
                    continue
                support_x, query_x, query_y = ep
                loss, *_ = episode_loss(self.frozen, self.metas, self.base_cache, W, eta,
                                         self.phi0, support_x, query_x, query_y, T)
                (loss / outer_batch).backward()
                total_loss += loss.item()
                n_ok += 1
            torch.nn.utils.clip_grad_norm_([W], max_norm=1.0)
            opt.step()
            if retract:                      # keep W on the Stiefel manifold -> gain <= 1
                with torch.no_grad():
                    W.copy_(qr_retract(W.detach()))

            if step % eval_every == 0 or step == outer_steps:   #do evaluate in this many outer steps

                val_res = eval_all_adapt_steps(W.detach())
                avg_loss = total_loss / max(n_ok, 1)
                trajectory.append(dict(
                    step=step, outer_loss=avg_loss,
                    val=[dict(adapt_steps=a,
                              metrics={m: [float(v[0]), float(v[1])] for m, v in met.items()})
                         for a, met in val_res.items()]))
                line = "  ".join(f"[adapt_steps={a}] val_F1={met['F1'][0]:.4f}+/-{met['F1'][1]:.4f}"
                                 for a, met in val_res.items())
                log_fn(f"    step {step}/{outer_steps}  outer_loss={avg_loss:.5f}  {line}")

                for a, met in val_res.items():
                    m, s = met["F1"]
                    if best_score is None or m > best_score:   # tie -> earlier step (first found wins)
                        best_score, best_f1, best_step, best_adapt_steps = m, m, step, a
                        best_W = W.detach().clone()

        return dict(n_val_draws=n_val_draws,
                    val_subjects=len(self._pool("val")[1].subjects),
                    W=W.detach(), W_init=W_init, W_best=best_W, best_step=best_step,
                    best_adapt_steps=best_adapt_steps, best_f1=best_f1, best_score=best_score,
                    trajectory=trajectory, rank=rank, k=k, eta=eta, basis_seed=basis_seed,
                    episode_seed=episode_seed, outer_steps=outer_steps, T=T, retract=retract)

    def eval_meta_basis(self, split, W, eta, k, n_draws=5, adapt_steps=1, T=1.0):
        return self._bidirectional_eval(split, W, eta, k, n_draws, adapt_steps, T)
