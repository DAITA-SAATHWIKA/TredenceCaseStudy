"""
Self-Pruning Neural Network on CIFAR-10
========================================
Tredence AI Engineering Internship — Case Study Solution

Author: Candidate Submission
Description:
    Implements a feed-forward neural network with learnable gate parameters
    that dynamically prune themselves during training via L1 sparsity regularization.
    Evaluated on CIFAR-10 across three lambda (λ) values to analyze the
    sparsity-vs-accuracy trade-off.

Usage:
    python self_pruning_network.py

Requirements:
    pip install torch torchvision matplotlib numpy tqdm
"""

import os
import math
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless backend – works without a display
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm


# ──────────────────────────────────────────────────────────────────────────────
# 1.  PRUNABLE LINEAR LAYER
# ──────────────────────────────────────────────────────────────────────────────

class PrunableLinear(nn.Module):
    """
    A drop-in replacement for nn.Linear that associates every weight with a
    learnable *gate score*.  During the forward pass the gate scores are passed
    through a Sigmoid to obtain values in (0, 1), and then multiplied
    element-wise with the weights.  An L1 penalty on the gate values (computed
    outside this module in the training loop) encourages most gates — and
    therefore most weights — to collapse to zero, effectively pruning them.

    Gradient flow
    -------------
    Both ``self.weight`` and ``self.gate_scores`` are ``nn.Parameter`` tensors,
    so autograd tracks every operation involving them.  The chain rule gives:

        ∂L/∂gate_scores  =  ∂L/∂output  ·  ∂output/∂pruned_weights
                             · ∂pruned_weights/∂gates
                             · ∂gates/∂gate_scores

    where  ∂pruned_weights/∂gates = weight   (element-wise)
    and    ∂gates/∂gate_scores    = sigmoid′ = gates * (1 - gates)

    Gradients also flow into ``self.weight`` through the same pruned_weights
    term, so both parameters are jointly optimised.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features

        # ── Trainable weights (same as nn.Linear) ──
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias   = nn.Parameter(torch.zeros(out_features)) if bias else None

        # ── Gate scores: one scalar per weight, initialised near 0.5 ──
        # Initialising to 0 (sigmoid → 0.5) gives balanced starting gates
        # while still allowing the optimiser full range in both directions.
        self.gate_scores = nn.Parameter(torch.zeros(out_features, in_features))

        # Kaiming uniform initialisation for weights (matches nn.Linear default)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    # ── Convenience property ──────────────────────────────────────────────────
    @property
    def gates(self) -> torch.Tensor:
        """Sigmoid-activated gate values ∈ (0, 1)."""
        return torch.sigmoid(self.gate_scores)

    # ── Forward pass ─────────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gates         = self.gates                          # (out, in)  ∈ (0,1)
        pruned_weights = self.weight * gates                # element-wise multiply
        return F.linear(x, pruned_weights, self.bias)

    # ── Utility methods ───────────────────────────────────────────────────────
    def sparsity(self, threshold: float = 1e-2) -> float:
        """Fraction of weights whose gate value is below *threshold*."""
        with torch.no_grad():
            g = self.gates
            return (g < threshold).float().mean().item()

    def extra_repr(self) -> str:
        return (f"in_features={self.in_features}, "
                f"out_features={self.out_features}, "
                f"bias={self.bias is not None}")


# ──────────────────────────────────────────────────────────────────────────────
# 2.  SELF-PRUNING NETWORK ARCHITECTURE
# ──────────────────────────────────────────────────────────────────────────────

class SelfPruningNet(nn.Module):
    """
    A three-hidden-layer feed-forward network built with PrunableLinear layers.

    Architecture
    ------------
    Input  : 3 × 32 × 32 CIFAR-10 images  (flattened → 3072)
    Hidden : 1024 → 512 → 256
    Output : 10 classes

    Batch-normalisation and ReLU activations are used between layers.
    Note: BN and activations are *not* prunable — only the linear projections
    carry gate parameters, consistent with the problem specification.
    """

    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            PrunableLinear(3072, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),

            PrunableLinear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),

            PrunableLinear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),

            PrunableLinear(256, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)   # flatten
        return self.net(x)

    # ── Helpers ───────────────────────────────────────────────────────────────
    def prunable_layers(self):
        """Yield all PrunableLinear submodules."""
        for m in self.modules():
            if isinstance(m, PrunableLinear):
                yield m

    def sparsity_loss(self) -> torch.Tensor:
        """
        L1 norm of *all* gate values across all PrunableLinear layers.

        Why L1?  The L1 norm penalises each gate proportionally to its
        magnitude without distinguishing very small values from zero.  This
        creates a constant gradient (-λ or +λ) that drives small gates all
        the way to exactly zero, unlike L2 which only asymptotically approaches
        zero.  In conjunction with the sigmoid squashing, this reliably
        produces a bimodal gate distribution with a sharp spike at 0.
        """
        return sum(layer.gates.sum() for layer in self.prunable_layers())

    def overall_sparsity(self, threshold: float = 1e-2) -> float:
        """Global sparsity: fraction of all prunable weights below *threshold*."""
        total = pruned = 0
        with torch.no_grad():
            for layer in self.prunable_layers():
                g = layer.gates
                total   += g.numel()
                pruned  += (g < threshold).sum().item()
        return pruned / total if total else 0.0

    def all_gate_values(self) -> np.ndarray:
        """Return a flat NumPy array of all gate values (for plotting)."""
        vals = []
        with torch.no_grad():
            for layer in self.prunable_layers():
                vals.append(layer.gates.cpu().numpy().ravel())
        return np.concatenate(vals)


# ──────────────────────────────────────────────────────────────────────────────
# 3.  DATA LOADING
# ──────────────────────────────────────────────────────────────────────────────

def get_cifar10_loaders(batch_size: int = 256, data_root: str = "./data"):
    """Return (train_loader, test_loader) for CIFAR-10."""
    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2470, 0.2435, 0.2616)

    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_set = datasets.CIFAR10(data_root, train=True,  download=True, transform=train_tf)
    test_set  = datasets.CIFAR10(data_root, train=False, download=True, transform=test_tf)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False,
                              num_workers=2, pin_memory=True)
    return train_loader, test_loader


# ──────────────────────────────────────────────────────────────────────────────
# 4.  TRAINING & EVALUATION HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, lambda_: float,
                    device: torch.device, scaler=None) -> tuple[float, float]:
    """
    One training epoch.

    Total Loss = CrossEntropy(logits, labels)  +  λ · Σ gates

    Returns (avg_classification_loss, avg_total_loss).
    """
    model.train()
    cls_loss_sum = total_loss_sum = 0.0
    n_batches = len(loader)

    for images, labels in loader:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        if scaler is not None:                    # AMP path
            with torch.autocast(device_type="cuda"):
                logits   = model(images)
                cls_loss = F.cross_entropy(logits, labels)
                sp_loss  = model.sparsity_loss()
                loss     = cls_loss + lambda_ * sp_loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:                                     # standard path (CPU / MPS)
            logits   = model(images)
            cls_loss = F.cross_entropy(logits, labels)
            sp_loss  = model.sparsity_loss()
            loss     = cls_loss + lambda_ * sp_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        cls_loss_sum   += cls_loss.item()
        total_loss_sum += loss.item()

    return cls_loss_sum / n_batches, total_loss_sum / n_batches


@torch.no_grad()
def evaluate(model, loader, device: torch.device) -> float:
    """Return top-1 test accuracy (0–100)."""
    model.eval()
    correct = total = 0
    for images, labels in loader:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        preds    = model(images).argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)
    return 100.0 * correct / total


# ──────────────────────────────────────────────────────────────────────────────
# 5.  MAIN EXPERIMENT LOOP
# ──────────────────────────────────────────────────────────────────────────────

def run_experiment(lambda_: float, train_loader, test_loader,
                   device: torch.device, epochs: int = 30,
                   lr: float = 1e-3) -> dict:
    """
    Train a fresh SelfPruningNet with the given λ and return result dict.
    """
    print(f"\n{'='*60}")
    print(f"  λ = {lambda_:.4f}  |  device = {device}  |  epochs = {epochs}")
    print(f"{'='*60}")

    model     = SelfPruningNet().to(device)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    scaler    = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    history = {"cls_loss": [], "total_loss": [], "sparsity": []}
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        cls_loss, tot_loss = train_one_epoch(
            model, train_loader, optimizer, lambda_, device, scaler)
        scheduler.step()
        sparsity = model.overall_sparsity()

        history["cls_loss"].append(cls_loss)
        history["total_loss"].append(tot_loss)
        history["sparsity"].append(sparsity * 100)

        if epoch % 5 == 0 or epoch == 1:
            acc = evaluate(model, test_loader, device)
            print(f"  Epoch {epoch:3d}/{epochs} | "
                  f"cls_loss={cls_loss:.4f} | "
                  f"tot_loss={tot_loss:.4f} | "
                  f"sparsity={sparsity*100:.1f}% | "
                  f"acc={acc:.2f}%")

    final_acc      = evaluate(model, test_loader, device)
    final_sparsity = model.overall_sparsity() * 100
    elapsed        = time.time() - t0

    print(f"\n  ✓  Final acc={final_acc:.2f}%  |  sparsity={final_sparsity:.1f}%  "
          f"|  time={elapsed:.0f}s")

    return {
        "lambda":      lambda_,
        "accuracy":    final_acc,
        "sparsity":    final_sparsity,
        "gate_values": model.all_gate_values(),
        "history":     history,
        "model":       model,
    }


# ──────────────────────────────────────────────────────────────────────────────
# 6.  PLOTTING
# ──────────────────────────────────────────────────────────────────────────────

def make_gate_distribution_plot(results: list[dict], save_path: str):
    """
    Figure 1 — Gate value distributions for all λ values.
    Shows the bimodal (spike at 0, cluster away from 0) pattern that
    characterises successful self-pruning.
    """
    n    = len(results)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), sharey=False)
    if n == 1:
        axes = [axes]

    colors = ["#E74C3C", "#2ECC71", "#3498DB"]   # low / medium / high λ

    for ax, res, color in zip(axes, results, colors):
        g = res["gate_values"]
        ax.hist(g, bins=80, color=color, alpha=0.82, edgecolor="none",
                log=True)                     # log-y to see the spike at 0 clearly
        ax.axvline(0.01, color="black", linestyle="--", linewidth=1,
                   label="prune threshold (0.01)")
        ax.set_title(
            f"λ = {res['lambda']}\n"
            f"Sparsity: {res['sparsity']:.1f}%  |  Acc: {res['accuracy']:.1f}%",
            fontsize=11, fontweight="bold")
        ax.set_xlabel("Gate value", fontsize=10)
        ax.set_ylabel("Count (log scale)", fontsize=10)
        ax.legend(fontsize=8)
        ax.set_xlim(-0.02, 1.02)
        frac_zero = (g < 0.01).mean() * 100
        ax.text(0.5, 0.92, f"{frac_zero:.1f}% gates < 0.01",
                transform=ax.transAxes, ha="center", fontsize=9,
                color="dimgrey")

    fig.suptitle("Self-Pruning Network — Gate Value Distributions per λ",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [saved] {save_path}")


def make_training_curves_plot(results: list[dict], save_path: str):
    """
    Figure 2 — Training dynamics: classification loss and sparsity over epochs.
    """
    fig = plt.figure(figsize=(14, 5))
    gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35)
    ax_loss = fig.add_subplot(gs[0])
    ax_sp   = fig.add_subplot(gs[1])

    colors = ["#E74C3C", "#2ECC71", "#3498DB"]
    lstyle = ["-", "--", "-."]

    for res, color, ls in zip(results, colors, lstyle):
        lbl    = f"λ={res['lambda']}"
        epochs = range(1, len(res["history"]["cls_loss"]) + 1)
        ax_loss.plot(epochs, res["history"]["cls_loss"], color=color,
                     linestyle=ls, linewidth=1.8, label=lbl)
        ax_sp.plot(epochs, res["history"]["sparsity"], color=color,
                   linestyle=ls, linewidth=1.8, label=lbl)

    ax_loss.set_title("Classification Loss vs Epoch", fontweight="bold")
    ax_loss.set_xlabel("Epoch"); ax_loss.set_ylabel("Cross-Entropy Loss")
    ax_loss.legend(); ax_loss.grid(alpha=0.3)

    ax_sp.set_title("Sparsity Level vs Epoch", fontweight="bold")
    ax_sp.set_xlabel("Epoch"); ax_sp.set_ylabel("Sparsity (%)")
    ax_sp.set_ylim(0, 105); ax_sp.legend(); ax_sp.grid(alpha=0.3)

    fig.suptitle("Training Dynamics — Self-Pruning Network", fontsize=13,
                 fontweight="bold")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [saved] {save_path}")


def make_summary_bar_plot(results: list[dict], save_path: str):
    """Figure 3 — Summary bar chart: accuracy & sparsity per λ."""
    lambdas   = [str(r["lambda"]) for r in results]
    accs      = [r["accuracy"]   for r in results]
    sparsities= [r["sparsity"]   for r in results]

    x   = np.arange(len(lambdas))
    w   = 0.35
    fig, ax1 = plt.subplots(figsize=(7, 4))

    bars1 = ax1.bar(x - w/2, accs,      w, label="Test Accuracy (%)",  color="#3498DB", alpha=0.85)
    ax2   = ax1.twinx()
    bars2 = ax2.bar(x + w/2, sparsities, w, label="Sparsity (%)",      color="#E74C3C", alpha=0.85)

    ax1.set_xlabel("λ (sparsity weight)", fontsize=11)
    ax1.set_ylabel("Test Accuracy (%)", color="#3498DB", fontsize=11)
    ax2.set_ylabel("Sparsity (%)",      color="#E74C3C", fontsize=11)
    ax1.set_xticks(x); ax1.set_xticklabels([f"λ={l}" for l in lambdas])
    ax1.set_ylim(0, 100); ax2.set_ylim(0, 100)

    lines  = [bars1, bars2]
    labels = [b.get_label() for b in lines]
    ax1.legend(lines, labels, loc="upper left", fontsize=9)

    for bar in bars1:
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f"{bar.get_height():.1f}%", ha="center", fontsize=8)
    for bar in bars2:
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f"{bar.get_height():.1f}%", ha="center", fontsize=8)

    ax1.set_title("Accuracy vs Sparsity Trade-off", fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [saved] {save_path}")


# ──────────────────────────────────────────────────────────────────────────────
# 7.  ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────

LAMBDAS   = [1e-4, 1e-3, 5e-3]   # low / medium / high sparsity pressure
EPOCHS    = 30
BATCH     = 256
OUTPUT_DIR = "./outputs"

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── device ────────────────────────────────────────────────────────────────
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"\n[Device] Using: {device}")

    # ── data ──────────────────────────────────────────────────────────────────
    train_loader, test_loader = get_cifar10_loaders(batch_size=BATCH)

    # ── experiments ───────────────────────────────────────────────────────────
    results = []
    for lam in LAMBDAS:
        res = run_experiment(lam, train_loader, test_loader, device,
                             epochs=EPOCHS)
        results.append(res)

    # ── summary table ─────────────────────────────────────────────────────────
    print("\n" + "="*55)
    print(f"  {'Lambda':<12} {'Test Accuracy':>15} {'Sparsity Level':>16}")
    print("  " + "-"*51)
    for r in results:
        print(f"  {r['lambda']:<12.4f} {r['accuracy']:>14.2f}% {r['sparsity']:>15.1f}%")
    print("="*55)

    # ── plots ─────────────────────────────────────────────────────────────────
    make_gate_distribution_plot(results,
        os.path.join(OUTPUT_DIR, "gate_distributions.png"))
    make_training_curves_plot(results,
        os.path.join(OUTPUT_DIR, "training_curves.png"))
    make_summary_bar_plot(results,
        os.path.join(OUTPUT_DIR, "summary_bar.png"))

    print(f"\n[Done]  All outputs saved to '{OUTPUT_DIR}/'")
    print("  Figures:")
    print("    gate_distributions.png — gate value histograms per λ")
    print("    training_curves.png    — loss and sparsity over epochs")
    print("    summary_bar.png        — accuracy vs sparsity comparison")


if __name__ == "__main__":
    main()
