"""
Train / Validation / Test cho GRU trong đề tài NIDS IoT-23.

Đầu vào: file .npz gồm
    X_train, y_train, X_val, y_val, X_test, y_test
    X: (N, T, F) float32   |   y: (N,) int
    (tuỳ chọn) class_names: mảng tên lớp theo thứ tự nhãn 0..K-1

Chạy:
    python train_gru.py --data data/iot23_seq.npz --out runs/gru_h64
"""
import argparse
import copy
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, precision_recall_fscore_support)
from torch.utils.data import DataLoader, TensorDataset


# ----------------------------------------------------------------- 1. Tiện ích
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_data(path):
    d = np.load(path, allow_pickle=True)
    out = {k: d[k] for k in ["X_train", "y_train", "X_val", "y_val", "X_test", "y_test"]}
    out["X_train"], out["X_val"], out["X_test"] = [
        out[k].astype(np.float32) for k in ["X_train", "X_val", "X_test"]]
    for k in ["y_train", "y_val", "y_test"]:
        out[k] = out[k].astype(np.int64)
    out["class_names"] = list(d["class_names"]) if "class_names" in d.files else None
    return out


def make_loader(X, y, batch, shuffle):
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    return DataLoader(ds, batch_size=batch, shuffle=shuffle)


# ----------------------------------------------------------------- 2. Model
class GRUNIDS(nn.Module):
    """GRU nhỏ: (B,T,F) -> hidden state bước cuối -> Linear -> logits (B,K)."""

    def __init__(self, n_feat, n_class, hidden=64, layers=1, dropout=0.2):
        super().__init__()
        self.gru = nn.GRU(n_feat, hidden, num_layers=layers, batch_first=True,
                          dropout=dropout if layers > 1 else 0.0)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden, n_class)

    def forward(self, x):
        _, h = self.gru(x)          # h: (layers, B, hidden)
        return self.fc(self.drop(h[-1]))


# ----------------------------------------------------------------- 3. Đánh giá
@torch.no_grad()
def run_eval(model, loader, criterion, device):
    model.eval()
    total_loss, preds, labels = 0.0, [], []
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        total_loss += criterion(logits, yb).item() * len(yb)
        preds.append(logits.argmax(1).cpu().numpy())
        labels.append(yb.cpu().numpy())
    preds, labels = np.concatenate(preds), np.concatenate(labels)
    return total_loss / len(labels), preds, labels


def metrics(y_true, y_pred, n_class):
    labs = list(range(n_class))
    p, r, f, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labs, average="macro", zero_division=0)
    pw, rw, fw, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labs, average="weighted", zero_division=0)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": p, "recall_macro": r, "f1_macro": f,
        "precision_weighted": pw, "recall_weighted": rw, "f1_weighted": fw,
    }


# ----------------------------------------------------------------- 4. Train
def train(model, tr, va, y_train, n_class, args, device):
    # Trọng số lớp tính CHỈ từ train (xử lý mất cân bằng)
    counts = np.bincount(y_train, minlength=n_class).astype(np.float64)
    w = np.zeros(n_class)
    nz = counts > 0
    w[nz] = (counts[nz].sum() / (nz.sum() * counts[nz])) ** args.weight_power
    w = w / w[nz].mean()
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32, device=device))
    eval_criterion = nn.CrossEntropyLoss()  # val/test: loss không trọng số cho dễ so sánh

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=2)

    best_f1, best_state, bad, history = -1.0, None, 0, []
    for ep in range(1, args.epochs + 1):
        model.train()
        run_loss, n = 0.0, 0
        t0 = time.time()
        for xb, yb in tr:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # chống bùng nổ gradient
            opt.step()
            run_loss += loss.item() * len(yb)
            n += len(yb)

        val_loss, vp, vl = run_eval(model, va, eval_criterion, device)
        m = metrics(vl, vp, n_class)
        sched.step(m["f1_macro"])
        history.append({"epoch": ep, "train_loss": run_loss / n, "val_loss": val_loss, **m})
        print(f"ep {ep:02d} | train {run_loss/n:.4f} | val {val_loss:.4f} | "
              f"acc {m['accuracy']:.4f} | F1-macro {m['f1_macro']:.4f} | {time.time()-t0:.1f}s")

        # Early stopping theo F1-macro của val (không theo accuracy)
        if m["f1_macro"] > best_f1:
            best_f1, best_state, bad = m["f1_macro"], copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= args.patience:
                print(f"Early stop tại epoch {ep}")
                break
    model.load_state_dict(best_state)
    return history, best_f1


# ----------------------------------------------------------------- 5. Đo độ phức tạp
def complexity(model, x_sample, device, out_path):
    n_params = sum(p.numel() for p in model.parameters())
    torch.save(model.state_dict(), out_path)
    size_kb = Path(out_path).stat().st_size / 1024

    model_cpu = copy.deepcopy(model).to("cpu").eval()
    x1 = x_sample[:1].cpu()
    with torch.no_grad():
        for _ in range(50):
            model_cpu(x1)
        t0 = time.perf_counter()
        for _ in range(300):
            model_cpu(x1)
        lat_ms = (time.perf_counter() - t0) / 300 * 1000
    return {"n_params": n_params, "model_size_kb_fp32": size_kb,
            "latency_ms_batch1_cpu": lat_ms}


# ----------------------------------------------------------------- 6. Main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="runs/gru")
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--weight_power", type=float, default=0.5,
                    help="0 = không dùng class weight, 1 = nghịch đảo tần suất đầy đủ")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    D = load_data(args.data)
    n_feat = D["X_train"].shape[2]
    n_class = int(max(D["y_train"].max(), D["y_val"].max(), D["y_test"].max())) + 1
    names = D["class_names"] or [str(i) for i in range(n_class)]
    print(f"Train {D['X_train'].shape} | Val {D['X_val'].shape} | Test {D['X_test'].shape} "
          f"| K={n_class} | device={device}")

    # Kiểm tra nhanh dữ liệu
    for k in ["X_train", "X_val", "X_test"]:
        assert np.isfinite(D[k]).all(), f"{k} còn NaN/inf"

    tr = make_loader(D["X_train"], D["y_train"], args.batch, True)
    va = make_loader(D["X_val"], D["y_val"], args.batch, False)
    te = make_loader(D["X_test"], D["y_test"], args.batch, False)

    model = GRUNIDS(n_feat, n_class, args.hidden, args.layers, args.dropout).to(device)
    print(model)

    history, best_val_f1 = train(model, tr, va, D["y_train"], n_class, args, device)

    # Test: chỉ chạy MỘT lần, sau khi đã chọn xong model theo val
    _, tp, tl = run_eval(model, te, nn.CrossEntropyLoss(), device)
    test_m = metrics(tl, tp, n_class)
    cm = confusion_matrix(tl, tp, labels=list(range(n_class)))
    report = classification_report(tl, tp, labels=list(range(n_class)),
                                   target_names=names, zero_division=0, digits=4)
    cx = complexity(model, torch.from_numpy(D["X_test"][:1]), device, out / "gru_fp32.pt")

    print("\n=== TEST ===")
    print(json.dumps(test_m, indent=2))
    print(report)
    print(cx)

    np.save(out / "confusion_matrix.npy", cm)
    (out / "classification_report.txt").write_text(report)
    (out / "results.json").write_text(json.dumps({
        "args": vars(args), "best_val_f1_macro": best_val_f1,
        "test": test_m, "complexity": cx, "history": history}, indent=2))


if __name__ == "__main__":
    main()
