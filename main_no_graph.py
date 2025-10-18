#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time, math, random
import numpy as np

# ============================================================
# Utils & Reproducibility
# ============================================================
def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)

def timestamp():
    return time.strftime("%Y%m%d-%H%M%S")

# ============================================================
# Data: โหลดชุดข้อมูล + แบ่ง k-fold แบบรักษาสัดส่วนคลาส
# ============================================================
def load_wdbc(filepath: str):
    """
    โหลดไฟล์ wdbc.data (UCI WDBC)
    - คอลัมน์ที่ 1: ID (ข้าม)
    - คอลัมน์ที่ 2: Diagnosis 'M'/'B' -> map เป็น 1/0
    - คอลัมน์ที่ 3-32: 30 ฟีเจอร์
    คืนค่า: X (N,30), y (N,)
    """
    feats, labels = [], []
    with open(filepath, "r") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 32:
                continue
            y = 1 if parts[1].strip() == "M" else 0
            x = list(map(float, parts[2:]))
            feats.append(x); labels.append(y)
    X = np.array(feats, dtype=float)
    y = np.array(labels, dtype=int)
    return X, y

def stratified_kfold_split(X, y, k=10, seed=42):
    rng = np.random.default_rng(seed)
    idx_pos = np.where(y == 1)[0]
    idx_neg = np.where(y == 0)[0]
    rng.shuffle(idx_pos); rng.shuffle(idx_neg)

    pos_folds = np.array_split(idx_pos, k)
    neg_folds = np.array_split(idx_neg, k)

    folds = []
    for i in range(k):
        fold_idx = np.concatenate([pos_folds[i], neg_folds[i]])
        rng.shuffle(fold_idx)
        folds.append(fold_idx)
    return folds

def standardize_train_test(X_train, X_test):
    mu = X_train.mean(axis=0)
    sigma = X_train.std(axis=0); sigma[sigma == 0] = 1.0
    Xtr = (X_train - mu) / sigma
    Xte = (X_test  - mu) / sigma
    return Xtr, Xte

# ============================================================
# Metrics (no plotting libs)
# ============================================================
def simple_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray):
    """
    สร้าง confusion matrix 2x2 (labels {0,1})
    [[TN, FP],
     [FN, TP]]
    """
    y_true = y_true.astype(int).ravel()
    y_pred = y_pred.astype(int).ravel()
    TN = int(np.sum((y_true == 0) & (y_pred == 0)))
    FP = int(np.sum((y_true == 0) & (y_pred == 1)))
    FN = int(np.sum((y_true == 1) & (y_pred == 0)))
    TP = int(np.sum((y_true == 1) & (y_pred == 1)))
    return np.array([[TN, FP],
                     [FN, TP]], dtype=int)

def balanced_accuracy_from_cm(cm: np.ndarray) -> float:
    TN, FP = cm[0,0], cm[0,1]
    FN, TP = cm[1,0], cm[1,1]
    rec0 = TN / max(TN + FP, 1)
    rec1 = TP / max(TP + FN, 1)
    return float((rec0 + rec1) / 2.0)

# ============================================================
# MLP (tanh hidden, sigmoid output)
# ============================================================
class MLP:
    def __init__(self, layer_sizes):
        assert layer_sizes[-1] == 1, "Binary output (1 unit) expected."
        self.layers = layer_sizes
        self.W, self.b = [], []
        for i in range(len(layer_sizes)-1):
            fan_in, fan_out = layer_sizes[i], layer_sizes[i+1]
            limit = math.sqrt(6.0/(fan_in+fan_out))
            Wi = np.random.uniform(-limit, limit, size=(fan_in, fan_out))
            bi = np.zeros((1, fan_out))
            self.W.append(Wi); self.b.append(bi)

    # genome handling
    def to_genome(self) -> np.ndarray:
        flat = []
        for Wi, bi in zip(self.W, self.b):
            flat.append(Wi.ravel()); flat.append(bi.ravel())
        return np.concatenate(flat)

    def from_genome(self, g: np.ndarray):
        pos = 0
        for i in range(len(self.W)):
            Wi = self.W[i]; szW = Wi.size
            self.W[i] = g[pos:pos+szW].reshape(Wi.shape); pos += szW
            bi = self.b[i]; szb = bi.size
            self.b[i] = g[pos:pos+szb].reshape(bi.shape); pos += szb

    # forward
    @staticmethod
    def _tanh(x): return np.tanh(x)
    @staticmethod
    def _sigmoid(x): return 1.0/(1.0 + np.exp(-x))

    def forward(self, X: np.ndarray) -> np.ndarray:
        a = X
        for i in range(len(self.W)-1):
            a = self._tanh(a @ self.W[i] + self.b[i])
        z_out = a @ self.W[-1] + self.b[-1]
        return self._sigmoid(z_out)  # (N,1)

    def predict(self, X: np.ndarray, thr: float = 0.5) -> np.ndarray:
        p = self.forward(X).reshape(-1)
        return (p >= thr).astype(int)

# ============================================================
# GA (Genetic Algorithm) สำหรับเทรนพารามิเตอร์ของ MLP
# ============================================================
class GAConfig:
    def __init__(self,
                 pop_size=30,
                 gens=80,
                 tournament_k=3,
                 elite_frac=0.2,
                 blx_alpha=0.3,
                 mut_prob=0.1,
                 mut_sigma=None,
                 early_stop_patience=15):
        self.pop_size = pop_size
        self.gens = gens
        self.tournament_k = tournament_k
        self.elite_frac = elite_frac
        self.blx_alpha = blx_alpha
        self.mut_prob = mut_prob
        self.mut_sigma = mut_sigma
        self.early_stop_patience = early_stop_patience

class GeneticMLPTrainer:
    def __init__(self, layer_sizes, cfg: GAConfig, seed=42, l2_reg=0.0):
        self.layer_sizes = layer_sizes
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        self.l2_reg = l2_reg
        self.hist_best = []

    @staticmethod
    def balanced_accuracy(y_true, y_pred):
        cm = simple_confusion_matrix(y_true, y_pred)
        return balanced_accuracy_from_cm(cm)

    def init_population(self):
        pop = []
        for _ in range(self.cfg.pop_size):
            m = MLP(self.layer_sizes)
            pop.append(m.to_genome())
        return np.array(pop)

    def tournament_select(self, pop, fitnesses):
        k = self.cfg.tournament_k
        idx = self.rng.choice(len(pop), size=k, replace=False)
        best = idx[0]; best_fit = fitnesses[best]
        for i in idx[1:]:
            if fitnesses[i] > best_fit:
                best, best_fit = i, fitnesses[i]
        return pop[best].copy()

    def blx_alpha_crossover(self, g1, g2):
        alpha = self.cfg.blx_alpha
        lo = np.minimum(g1, g2)
        hi = np.maximum(g1, g2)
        diff = hi - lo
        low_ext  = lo - alpha * diff
        high_ext = hi + alpha * diff
        return self.rng.uniform(low=low_ext, high=high_ext)

    def mutate(self, g):
        if self.cfg.mut_prob <= 0:
            return g
        if self.cfg.mut_sigma is None:
            scale = np.median(np.abs(g)) if np.any(g) else 1.0
            sigma = 0.1 * (scale if scale > 1e-8 else 1.0)
        else:
            sigma = self.cfg.mut_sigma
        mask = self.rng.random(g.shape) < self.cfg.mut_prob
        return g + mask * self.rng.normal(0.0, sigma, size=g.shape)

    def genome_to_model(self, genome):
        m = MLP(self.layer_sizes)
        m.from_genome(genome)
        return m

    def fitness(self, genome, Xtr, ytr, Xval, yval):
        m = self.genome_to_model(genome)
        ypred = m.predict(Xval)
        bal_acc = self.balanced_accuracy(yval, ypred)
        if self.l2_reg > 0:
            wnorm = sum(float(np.sum(W*W)) for W in m.W)
            bal_acc -= self.l2_reg * wnorm
        return bal_acc

    def evolve(self, Xtr, ytr, Xval, yval):
        pop = self.init_population()
        fitnesses = np.array([self.fitness(g, Xtr, ytr, Xval, yval) for g in pop])

        elite_k = max(2, int(self.cfg.elite_frac * self.cfg.pop_size))
        best_so_far = np.max(fitnesses); self.hist_best = [best_so_far]
        last_improve_gen = 0

        for gen in range(1, self.cfg.gens+1):
            # เก็บ elite
            elite_idx = np.argsort(-fitnesses)[:elite_k]
            elites = pop[elite_idx].copy()

            # ผสม + กลายพันธุ์
            children = []
            while len(children) + elite_k < self.cfg.pop_size:
                p1 = self.tournament_select(pop, fitnesses)
                p2 = self.tournament_select(pop, fitnesses)
                child = self.blx_alpha_crossover(p1, p2)
                child = self.mutate(child)
                children.append(child)
            pop = np.vstack([elites] + [np.array(children)]) if children else elites

            # ประเมินรุ่นใหม่
            fitnesses = np.array([self.fitness(g, Xtr, ytr, Xval, yval) for g in pop])
            best = np.max(fitnesses)
            self.hist_best.append(best)

            if best > best_so_far + 1e-12:
                best_so_far = best
                last_improve_gen = gen

            print(f"[GA] Gen {gen:03d}/{self.cfg.gens} | Best (balanced acc): {best:.4f}")

            # early stop
            if gen - last_improve_gen >= self.cfg.early_stop_patience:
                print(f"[GA] Early stop at gen {gen} (patience={self.cfg.early_stop_patience})")
                break

        best_idx = int(np.argmax(fitnesses))
        return self.genome_to_model(pop[best_idx]), self.hist_best

# ============================================================
# Training & Evaluation (k-fold) — พิมพ์ผลในเทอร์มินัลเท่านั้น
# ============================================================
def run_experiment():
    set_global_seed(SEED)
    X, y = load_wdbc(DATA_PATH)

    folds = stratified_kfold_split(X, y, k=K_FOLDS, seed=SEED)
    accs, histories, best_models = [], [], []

    layer_sizes = [X.shape[1]] + list(HIDDEN_SIZES) + [1]

    cfg = GAConfig(
        pop_size=POP_SIZE, gens=GENERATIONS, tournament_k=TOURNAMENT_K,
        elite_frac=ELITE_FRAC, blx_alpha=BLX_ALPHA, mut_prob=MUT_PROB,
        mut_sigma=MUT_SIGMA, early_stop_patience=PATIENCE
    )

    print(f"== Start GA-MLP (WDBC) | {K_FOLDS}-fold | hidden={HIDDEN_SIZES} ==")

    for fi in range(K_FOLDS):
        val_idx = folds[fi]
        tr_idx = np.concatenate([folds[j] for j in range(K_FOLDS) if j != fi])

        Xtr, ytr = X[tr_idx], y[tr_idx]
        Xva, yva = X[val_idx], y[val_idx]
        Xtr, Xva = standardize_train_test(Xtr, Xva)

        print(f"\n--- Fold {fi+1}/{K_FOLDS} ---")
        trainer = GeneticMLPTrainer(layer_sizes, cfg, seed=SEED + fi, l2_reg=L2_REG)
        best_model, hist = trainer.evolve(Xtr, ytr, Xva, yva)
        histories.append(hist)

        yhat = best_model.predict(Xva)
        bal_acc = GeneticMLPTrainer.balanced_accuracy(yva, yhat)
        accs.append(bal_acc)
        best_models.append(best_model)

        # พิมพ์สรุปของโฟลด์นี้
        last = hist[-1] if len(hist) > 0 else float('nan')
        print(f"[Fold {fi+1}] Balanced Acc (val): {bal_acc:.4f} | "
              f"Best fitness @last gen: {last:.4f}")

    # ค่าเฉลี่ย k-fold
    avg = float(np.mean(accs))
    print("\n==================== RESULTS ====================")
    for i, a in enumerate(accs, 1):
        print(f"Fold {i:02d}: Balanced Acc = {a:.4f}")
    print(f"AVERAGE Balanced Acc over {K_FOLDS} folds = {avg:.4f}")

    # Confusion Matrix บนทั้งชุดข้อมูล ด้วยสเกลจากโฟลด์ที่ดีที่สุด
    best_fold = int(np.argmax(accs))
    tr_idx = np.concatenate([folds[j] for j in range(K_FOLDS) if j != best_fold])
    Xtr_all = X[tr_idx]
    _, X_all_std = standardize_train_test(Xtr_all, X)
    y_all_pred = best_models[best_fold].predict(X_all_std)

    cm = simple_confusion_matrix(y, y_all_pred)
    TN, FP = cm[0,0], cm[0,1]
    FN, TP = cm[1,0], cm[1,1]
    bal_all = balanced_accuracy_from_cm(cm)

    print("\nConfusion Matrix (all data via best-fold scaler)")
    print("            Pred=0   Pred=1")
    print(f"True=0    | {TN:6d} | {FP:6d}")
    print(f"True=1    | {FN:6d} | {TP:6d}")
    print(f"Balanced Accuracy (all) = {bal_all:.4f}")
    print(f"(Best fold index, 0-based) = {best_fold}")

# ============================================================
# main
# ============================================================
if __name__ == "__main__":
    # -------- USER CONFIG --------
    DATA_PATH      = "wdbc.data"  # ไฟล์ข้อมูล
    K_FOLDS        = 10
    HIDDEN_SIZES   = [12]         # ตัวอย่าง: [16,8], [8], [] (ไม่มี hidden)
    POP_SIZE       = 36
    GENERATIONS    = 120
    TOURNAMENT_K   = 3
    ELITE_FRAC     = 0.22
    BLX_ALPHA      = 0.35
    MUT_PROB       = 0.08
    MUT_SIGMA      = None         # None = ให้โปรแกรมคำนวณ sigma อัตโนมัติ
    PATIENCE       = 20           # early stopping patience
    L2_REG         = 0.0
    SEED           = 2025

    run_experiment()
