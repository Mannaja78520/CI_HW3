#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, time, math, random
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# ============================================================
# Utils & Reproducibility (ฟังก์ชันช่วยเหลือ + ทำให้สุ่มซ้ำได้)
# ============================================================
def set_global_seed(seed: int):
    """ตั้งค่า seed ให้ทั้ง random และ numpy เพื่อให้ผลลัพธ์ทำซ้ำได้"""
    random.seed(seed)
    np.random.seed(seed)

def timestamp():
    """คืนสตริงเวลาปัจจุบันใช้ตั้งชื่อโฟลเดอร์/ไฟล์ผลลัพธ์ เช่น 20251019-153012"""
    return time.strftime("%Y%m%d-%H%M%S")

# ============================================================
# Data: โหลดชุดข้อมูล + แบ่ง k-fold แบบรักษาสัดส่วนคลาส (stratified)
# ============================================================
def load_wdbc(filepath: str):
    """
    โหลดไฟล์ wdbc.data (UCI WDBC)
    - คอลัมน์ที่ 1: ID (ข้าม)
    - คอลัมน์ที่ 2: Diagnosis 'M'/'B' -> map เป็น 1/0
    - คอลัมน์ที่ 3-32: 30 ฟีเจอร์แบบตัวเลขลอย
    คืนค่า:
      X: np.ndarray (N,30)
      y: np.ndarray (N,)
    """
    feats, labels = [], []
    with open(filepath, "r") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 32:            # เผื่อบรรทัดไม่ครบ 32 คอลัมน์ให้ข้าม
                continue
            y = 1 if parts[1].strip() == "M" else 0  # M=1 (malignant), B=0 (benign)
            x = list(map(float, parts[2:]))          # 30 ฟีเจอร์
            feats.append(x); labels.append(y)
    X = np.array(feats, dtype=float)
    y = np.array(labels, dtype=int)
    return X, y

def stratified_kfold_split(X, y, k=10, seed=42):
    """
    แบ่ง index ของข้อมูลเป็น k โฟลด์แบบ stratified (รักษาสัดส่วนคลาสเท่า ๆ กัน)
    คืนค่า:
      folds: list ความยาว k โดยแต่ละสมาชิกคือ np.ndarray ของดัชนีในโฟลด์นั้น
    """
    rng = np.random.default_rng(seed)
    # แยกดัชนีแต่ละคลาส
    idx_pos = np.where(y == 1)[0]
    idx_neg = np.where(y == 0)[0]
    rng.shuffle(idx_pos); rng.shuffle(idx_neg)     # สุ่มลำดับ

    # แบ่งแต่ละคลาสเป็น k ส่วน แล้วค่อยรวมกันเป็นโฟลด์
    pos_folds = np.array_split(idx_pos, k)
    neg_folds = np.array_split(idx_neg, k)

    folds = []
    for i in range(k):
        fold_idx = np.concatenate([pos_folds[i], neg_folds[i]])
        rng.shuffle(fold_idx)                      # สุ่มสลับภายในโฟลด์อีกรอบ
        folds.append(fold_idx)
    return folds

def standardize_train_test(X_train, X_test):
    """
    ทำมาตรฐานแบบ z-score โดยใช้ mean/std จาก TRAIN เท่านั้น (ป้องกัน data leakage)
    คืนค่า:
      Xtr: X_train ที่ถูกสเกลแล้ว
      Xte: X_test  ที่ถูกสเกลด้วยสถิติจาก train
    """
    mu = X_train.mean(axis=0)
    sigma = X_train.std(axis=0); sigma[sigma == 0] = 1.0  # กันหารด้วยศูนย์
    Xtr = (X_train - mu) / sigma
    Xte = (X_test  - mu) / sigma
    return Xtr, Xte

# ============================================================
# MLP (โครงข่ายประสาทเทียมอย่างง่าย)
# - hidden ใช้ tanh
# - output ใช้ sigmoid (ปัญหาทูคลาส -> ออก 1 ยูนิต)
# - รองรับการแปลงพารามิเตอร์เป็นจีโนม/คืนจากจีโนมเพื่อใช้กับ GA
# ============================================================
class MLP:
    def __init__(self, layer_sizes):
        """
        layer_sizes: ลิสต์ขนาดแต่ละชั้น เช่น [30, 12, 1]
          - input  = 30
          - hidden = 12
          - output = 1 (ต้องเป็น 1 สำหรับ binary)
        """
        assert layer_sizes[-1] == 1, "Binary output (1 unit) expected."
        self.layers = layer_sizes
        self.W, self.b = [], []

        # Xavier/Glorot (uniform-ish) เพื่อให้สเกลของสัญญาณสมดุลตอนเริ่ม
        for i in range(len(layer_sizes)-1):
            fan_in, fan_out = layer_sizes[i], layer_sizes[i+1]
            limit = math.sqrt(6.0/(fan_in+fan_out))
            Wi = np.random.uniform(-limit, limit, size=(fan_in, fan_out))
            bi = np.zeros((1, fan_out))
            self.W.append(Wi); self.b.append(bi)

    # ---------- genome handling ----------
    def to_genome(self) -> np.ndarray:
        """รวม W และ b ของทุกเลเยอร์เป็นเวกเตอร์เดียว (ใช้เป็นจีโนมใน GA)"""
        flat = []
        for Wi, bi in zip(self.W, self.b):
            flat.append(Wi.ravel()); flat.append(bi.ravel())
        return np.concatenate(flat)

    def from_genome(self, g: np.ndarray):
        """คืนค่าเวกเตอร์จีโนมกลับเป็น W และ b ตามรูปร่างเดิมของแต่ละเลเยอร์"""
        pos = 0
        for i in range(len(self.W)):
            Wi = self.W[i]; szW = Wi.size
            self.W[i] = g[pos:pos+szW].reshape(Wi.shape); pos += szW
            bi = self.b[i]; szb = bi.size
            self.b[i] = g[pos:pos+szb].reshape(bi.shape); pos += szb

    # ---------- forward ----------
    @staticmethod
    def _tanh(x): return np.tanh(x)
    @staticmethod
    def _sigmoid(x): return 1.0/(1.0 + np.exp(-x))

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        ส่งสัญญาณหน้าผ่านชั้นซ่อน (tanh) ไปยังเอาต์พุต (sigmoid)
        คืนค่า: ความน่าจะเป็น class=1 ขนาด (N,1)
        """
        a = X
        for i in range(len(self.W)-1):
            a = self._tanh(a @ self.W[i] + self.b[i])
        z_out = a @ self.W[-1] + self.b[-1]
        return self._sigmoid(z_out)  # (N,1)

    def predict(self, X: np.ndarray, thr: float = 0.5) -> np.ndarray:
        """แปลงความน่าจะเป็นเป็นคลาสด้วย threshold (ดีฟอลต์ 0.5) -> เวกเตอร์ 0/1"""
        p = self.forward(X).reshape(-1)
        return (p >= thr).astype(int)

# ============================================================
# GA (Genetic Algorithm) สำหรับเทรนพารามิเตอร์ของ MLP
# - เลือกด้วย Tournament
# - Crossover แบบ BLX-α (ขยายช่วงระหว่างพ่อแม่)
# - Mutation เป็น Gaussian ต่อจีโนมบางตำแหน่ง
# - Fitness = Balanced Accuracy (เฉลี่ย recall ของแต่ละคลาส)
# - มี Early Stopping ตาม patience
# ============================================================
class GAConfig:
    def __init__(self,
                 pop_size=30,          # ขนาดประชากร
                 gens=80,              # จำนวนเจนเนอเรชันสูงสุด
                 tournament_k=3,       # ขนาดทัวร์นาเมนต์
                 elite_frac=0.2,       # สัดส่วน elite ที่เก็บข้ามเจน
                 blx_alpha=0.3,        # พารามิเตอร์ BLX-α
                 mut_prob=0.1,         # ความน่าจะเป็น mutation ต่อยีน
                 mut_sigma=None,       # ส่วนเบี่ยงเบนมาตรฐานของ noise (None=ปรับอัตโนมัติ)
                 early_stop_patience=15 # หยุดถ้าไม่ดีขึ้นติดต่อกันเท่านี้เจน
                 ):
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
        """
        layer_sizes: สถาปัตยกรรมของ MLP
        cfg        : ค่าพารามิเตอร์ของ GA
        seed       : seed ของตัวสุ่ม GA
        l2_reg     : ค่าลงโทษ L2 (ช่วยลด overfitting ถ้าตั้ง > 0)
        """
        self.layer_sizes = layer_sizes
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        self.l2_reg = l2_reg
        self.hist_best = []  # เก็บค่าฟิตเนสที่ดีที่สุดของแต่ละเจน

    # ---------- ตัวชี้วัด ----------
    @staticmethod
    def balanced_accuracy(y_true, y_pred):
        """macro recall (balanced accuracy) -> เฉลี่ย recall ของคลาส 0 และ 1"""
        cm = confusion_matrix(y_true, y_pred, labels=[0,1])
        with np.errstate(divide='ignore', invalid='ignore'):
            recalls = np.diag(cm) / cm.sum(axis=1)
        recalls = np.nan_to_num(recalls, nan=0.0)
        return float(recalls.mean())

    # ---------- ขั้นตอนของ GA ----------
    def init_population(self):
        """สร้างประชากรเริ่มต้น (สุ่ม MLP แล้วแปลงเป็นจีโนม)"""
        pop = []
        for _ in range(self.cfg.pop_size):
            m = MLP(self.layer_sizes)
            pop.append(m.to_genome())
        return np.array(pop)

    def tournament_select(self, pop, fitnesses):
        """เลือกพ่อแม่ด้วยทัวร์นาเมนต์: สุ่ม k ตัว เอาตัวที่ fitness สูงสุด"""
        k = self.cfg.tournament_k
        idx = self.rng.choice(len(pop), size=k, replace=False)
        best = idx[0]; best_fit = fitnesses[best]
        for i in idx[1:]:
            if fitnesses[i] > best_fit:
                best, best_fit = i, fitnesses[i]
        return pop[best].copy()

    def blx_alpha_crossover(self, g1, g2):
        """
        BLX-α crossover:
        สำหรับแต่ละยีน สุ่มค่าระหว่าง [min-α*diff, max+α*diff]
        ทำให้ลูกมีโอกาสออกนอกช่วงพ่อแม่เล็กน้อยเพื่อกระตุ้นการสำรวจ
        """
        alpha = self.cfg.blx_alpha
        lo = np.minimum(g1, g2)
        hi = np.maximum(g1, g2)
        diff = hi - lo
        low_ext  = lo - alpha * diff
        high_ext = hi + alpha * diff
        return self.rng.uniform(low=low_ext, high=high_ext)

    def mutate(self, g):
        """
        กลายพันธุ์แบบ Gaussian: เลือกบางยีนตาม prob แล้วเติม noise ~ N(0, sigma^2)
        ถ้าไม่กำหนด sigma จะคำนวณจาก median(|gene|) เพื่อให้สเกลเหมาะสมอัตโนมัติ
        """
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
        """แปลงจีโนมกลับเป็นโมเดล MLP (ใส่ค่า W/b ให้ครบ)"""
        m = MLP(self.layer_sizes)
        m.from_genome(genome)
        return m

    def fitness(self, genome, Xtr, ytr, Xval, yval):
        """
        วัดความฟิตของ genome:
        - เอา genome -> โมเดล -> ทำนายบน validation -> คำนวณ balanced accuracy
        - (ถ้าตั้ง L2) ลบโทษตาม norm(W) เพื่อลดโมเดลที่น้ำหนักสูงเกิน
        """
        m = self.genome_to_model(genome)
        ypred = m.predict(Xval)
        bal_acc = self.balanced_accuracy(yval, ypred)
        if self.l2_reg > 0:
            wnorm = sum(float(np.sum(W*W)) for W in m.W)
            bal_acc -= self.l2_reg * wnorm
        return bal_acc

    def evolve(self, Xtr, ytr, Xval, yval):
        """
        วงจรหลักของ GA:
        1) สร้างประชากรเริ่มต้น
        2) วัด fitness
        3) เก็บ elite
        4) ผสมพ่อแม่ (BLX-α) + กลายพันธุ์
        5) วนซ้ำจนถึง gens หรือหยุดก่อนด้วย early stopping
        คืนค่า:
          - โมเดลที่ดีที่สุด (สร้างจาก genome ที่ดีที่สุด)
          - ประวัติ best fitness รายเจน
        """
        pop = self.init_population()
        fitnesses = np.array([self.fitness(g, Xtr, ytr, Xval, yval) for g in pop])

        elite_k = max(2, int(self.cfg.elite_frac * self.cfg.pop_size))  # เก็บอย่างน้อย 2 ตัว
        best_so_far = np.max(fitnesses); self.hist_best = [best_so_far]
        last_improve_gen = 0

        for gen in range(1, self.cfg.gens+1):
            # ----- เก็บ elite -----
            elite_idx = np.argsort(-fitnesses)[:elite_k]
            elites = pop[elite_idx].copy()

            # ----- สร้างลูกจนเต็มประชากร -----
            children = []
            while len(children) + elite_k < self.cfg.pop_size:
                p1 = self.tournament_select(pop, fitnesses)
                p2 = self.tournament_select(pop, fitnesses)
                child = self.blx_alpha_crossover(p1, p2)
                child = self.mutate(child)
                children.append(child)
            pop = np.vstack([elites] + [np.array(children)]) if children else elites

            # ----- ประเมินเจนใหม่ -----
            fitnesses = np.array([self.fitness(g, Xtr, ytr, Xval, yval) for g in pop])
            best = np.max(fitnesses)
            self.hist_best.append(best)

            # ปรับสถานะการดีขึ้นล่าสุด (ใช้กับ early stop)
            if best > best_so_far + 1e-12:
                best_so_far = best
                last_improve_gen = gen

            print(f"[GA] Gen {gen:03d}/{self.cfg.gens} | Best (balanced acc): {best:.4f}")

            # ----- Early stopping: ถ้าไม่ดีขึ้นติดต่อกันตาม patience ให้หยุด -----
            if gen - last_improve_gen >= self.cfg.early_stop_patience:
                print(f"[GA] Early stop at gen {gen} (patience={self.cfg.early_stop_patience})")
                break

        # สร้างโมเดลจาก genome ที่ดีที่สุดของรุ่นสุดท้าย
        best_idx = int(np.argmax(fitnesses))
        return self.genome_to_model(pop[best_idx]), self.hist_best

# ============================================================
# Training & Evaluation (k-fold)
# - รัน GA แยกตามโฟลด์
# - เก็บคะแนน/ประวัติ
# - พล็อต กราฟและ Confusion Matrix (แสดง + เซฟ)
# ============================================================
def run_experiment():
    # สร้างโฟลเดอร์ผลลัพธ์ตามเวลา เพื่อไม่ทับของเดิม
    outdir = Path("outputs") / f"wdbc_ga_{timestamp()}"
    outdir.mkdir(parents=True, exist_ok=True)

    # ตั้ง seed ให้ผลทำซ้ำได้ + โหลดข้อมูล
    set_global_seed(SEED)
    X, y = load_wdbc(DATA_PATH)

    # สร้างโฟลด์แบบ stratified
    folds = stratified_kfold_split(X, y, k=K_FOLDS, seed=SEED)
    accs, histories, best_models = [], [], []

    # นิยามสถาปัตยกรรมของ MLP จากพารามิเตอร์ผู้ใช้
    layer_sizes = [X.shape[1]] + list(HIDDEN_SIZES) + [1]

    # กำหนดคอนฟิก GA
    cfg = GAConfig(
        pop_size=POP_SIZE, gens=GENERATIONS, tournament_k=TOURNAMENT_K,
        elite_frac=ELITE_FRAC, blx_alpha=BLX_ALPHA, mut_prob=MUT_PROB,
        mut_sigma=MUT_SIGMA, early_stop_patience=PATIENCE
    )

    # วน k-fold
    for fi in range(K_FOLDS):
        val_idx = folds[fi]                                           # โฟลด์ที่ใช้วัดผล
        tr_idx = np.concatenate([folds[j] for j in range(K_FOLDS) if j != fi])  # ที่เหลือใช้เทรน

        # แตกชุดเทรน/วาล และทำมาตรฐานด้วยสถิติจากเทรน
        Xtr, ytr = X[tr_idx], y[tr_idx]
        Xva, yva = X[val_idx], y[val_idx]
        Xtr, Xva = standardize_train_test(Xtr, Xva)

        # เทรนด้วย GA แล้วบันทึกประวัติและโมเดลดีที่สุดของโฟลด์นี้
        trainer = GeneticMLPTrainer(layer_sizes, cfg, seed=SEED + fi, l2_reg=L2_REG)
        best_model, hist = trainer.evolve(Xtr, ytr, Xva, yva)
        histories.append(hist)

        # วัดผลบนวาลิเดชัน
        yhat = best_model.predict(Xva)
        bal_acc = GeneticMLPTrainer.balanced_accuracy(yva, yhat)
        accs.append(bal_acc)
        best_models.append(best_model)

        print(f"[Fold {fi+1}/{K_FOLDS}] Balanced Acc = {bal_acc:.4f}")

    # ค่าเฉลี่ย k-fold
    avg = float(np.mean(accs))
    print(f"\n== Stratified {K_FOLDS}-fold | Avg Balanced Acc: {avg:.4f} ==")

    # -------------------- Visualization --------------------
    # 1) กราฟความคืบหน้า GA ต่อโฟลด์
    rows = 2
    cols = (len(histories) + 1) // 2
    fig1, axes1 = plt.subplots(rows, cols, figsize=(14, 8))
    axes1 = np.array(axes1).reshape(-1)
    for i, h in enumerate(histories):
        axes1[i].plot(h)
        axes1[i].set_title(f"Fold {i+1}")
        axes1[i].set_xlabel("Generation")
        axes1[i].set_ylabel("Best Balanced Acc")
    # ปิด subplot ที่ไม่ได้ใช้ (กรณีจำนวนกราฟไม่พอดีกริด)
    for j in range(i+1, len(axes1)):
        axes1[j].axis('off')
    fig1.suptitle("GA Progress per Fold")
    fig1.tight_layout()
    if SAVE_PLOTS:
        fig1.savefig(outdir / "ga_progress_per_fold.png", dpi=160)

    # 2) กราฟแท่งคะแนนต่อโฟลด์ + เส้นค่าเฉลี่ย
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    xs = np.arange(1, len(accs) + 1)
    ax2.bar(xs, accs)
    ax2.axhline(avg, linestyle="--", label=f"Avg = {avg:.4f}")
    ax2.set_xlabel("Fold"); ax2.set_ylabel("Balanced Accuracy")
    ax2.set_title(f"K-Fold Balanced Accuracy (k={K_FOLDS})")
    ax2.legend()
    fig2.tight_layout()
    if SAVE_PLOTS:
        fig2.savefig(outdir / "balanced_accuracy_per_fold.png", dpi=160)

    # 3) Confusion Matrix บนทั้งชุดข้อมูล โดยใช้สเกลจากเทรนของ "โฟลด์ที่ดีที่สุด"
    best_fold = int(np.argmax(accs))                                # หาโฟลด์ที่ทำคะแนนดีที่สุด
    tr_idx = np.concatenate([folds[j] for j in range(K_FOLDS) if j != best_fold])
    Xtr_all = X[tr_idx]
    _, X_all_std = standardize_train_test(Xtr_all, X)               # ใช้สถิติของ train มาสเกลทั้งชุด
    y_all_pred = best_models[best_fold].predict(X_all_std)

    cm = confusion_matrix(y, y_all_pred, labels=[0, 1])
    fig3, ax3 = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["B(0)", "M(1)"])
    disp.plot(ax=ax3, colorbar=False)
    ax3.set_title("Confusion Matrix (all data via best-fold scaler)")
    fig3.tight_layout()
    if SAVE_PLOTS:
        fig3.savefig(outdir / "confusion_matrix_best_fold_on_all.png", dpi=160)

    # แสดงทุกรูปพร้อมกัน (ถ้า RUN ในสภาพแวดล้อมมี GUI)
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close('all')

    # path ไฟล์ที่เซฟ
    if SAVE_PLOTS:
        print(f"[SAVE] {outdir/'genetic algorithms_progress_per_fold.png'}")
        print(f"[SAVE] {outdir/'balanced_accuracy_per_fold.png'}")
        print(f"[SAVE] {outdir/'confusion_matrix_best_fold_on_all.png'}")

# ============================================================
# main
# ============================================================
if __name__ == "__main__":
    # -------- USER CONFIG (แก้ค่าตรงนี้) --------
    DATA_PATH      = "wdbc.data"  # ไฟล์ข้อมูล
    K_FOLDS        = 10
    HIDDEN_SIZES   = [12]         # ปรับเป็น [16,8], [8], [] ได้ตามต้องการ
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
    SHOW_PLOTS     = True         # True = โชว์กราฟ
    SAVE_PLOTS     = True         # True = เซฟรูป PNG ลงโฟลเดอร์ outputs/

    run_experiment()
