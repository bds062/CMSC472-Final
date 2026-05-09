"""
model_training.py
─────────────────
End-to-end training script for EEGNet with joint classification +
contrastive learning on SEED / SEED-IV DE features.

Usage
-----
    # use defaults from config.yaml
    python experiments/model_training.py

    # override any field via CLI
    python experiments/model_training.py --config config.yaml --run_name Exp01 --loss contrastive --epochs 100
    python experiments/model_training.py --leave_one_out --val_subject 3
    python experiments/model_training.py --loss prototype --warmup_epochs 5 --lr 1e-3
"""

import argparse
import os
import sys
import random

import numpy as np
import torch
import matplotlib.pyplot as plt
import yaml
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

sys.path.append('./src')
from dataloader import build_loaders
from model import build_model, EEGDataset, BalancedBatchSampler, Trainer
from losses import ClassificationLoss, ContrastiveLoss, ContrastivePrototype, LeaveOneOutContrastiveLearning


# ─────────────────────────────────────────────────────────────────────────────
# 1. CONFIG
# ─────────────────────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def parse_args():
    parser = argparse.ArgumentParser(description='EEGNet training script')

    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to config.yaml (default: config.yaml)')

    # ── overrides (all optional; take precedence over config.yaml) ────────
    # logging
    parser.add_argument('--run_name', type=str)
    parser.add_argument('--log_dir',  type=str)

    # data
    parser.add_argument('--data_root',     type=str)
    parser.add_argument('--dataset',       type=str, choices=['SEED-IV', 'SEED'])
    parser.add_argument('--leave_one_out', action='store_true', default=None)
    parser.add_argument('--val_subject',   type=int)
    parser.add_argument('--val_fraction',  type=float)
    parser.add_argument('--n_per_class',   type=int)
    parser.add_argument('--strategy',      type=str,
                        choices=['cross_subject', 'within_subject'])

    # training
    parser.add_argument('--epochs',        type=int)
    parser.add_argument('--lr',            type=float)
    parser.add_argument('--weight_decay',  type=float)
    parser.add_argument('--warmup_epochs', type=int)
    parser.add_argument('--lambda_con',    type=float)
    parser.add_argument('--loss',          type=str,
                        choices=['none', 'contrastive', 'prototype', 'locl'])
    parser.add_argument('--temperature',   type=float)
    parser.add_argument('--seed',          type=int)
    parser.add_argument('--device',        type=str)

    # model
    parser.add_argument('--nb_classes',  type=int)
    parser.add_argument('--Chans',       type=int)
    parser.add_argument('--Samples',     type=int)
    parser.add_argument('--dropoutRate', type=float)

    return parser.parse_args()


def merge_config(cfg: dict, args) -> dict:
    """Override config values with any CLI arguments that were explicitly set."""
    overrides = {
        # logging
        'run_name'     : ('logging', 'run_name'),
        'log_dir'      : ('logging', 'log_dir'),
        # data
        'data_root'    : ('data', 'data_root'),
        'dataset'      : ('data', 'dataset'),
        'leave_one_out': ('data', 'leave_one_out'),
        'val_subject'  : ('data', 'val_subject'),
        'val_fraction' : ('data', 'val_fraction'),
        'n_per_class'  : ('data', 'n_per_class'),
        'strategy'     : ('data', 'strategy'),
        # training
        'epochs'       : ('train', 'epochs'),
        'lr'           : ('train', 'lr'),
        'weight_decay' : ('train', 'weight_decay'),
        'warmup_epochs': ('train', 'warmup_epochs'),
        'lambda_con'   : ('train', 'lambda_con'),
        'loss'         : ('train', 'loss'),
        'temperature'  : ('train', 'temperature'),
        'seed'         : ('train', 'seed'),
        'device'       : ('train', 'device'),
        # model
        'nb_classes'   : ('model', 'nb_classes'),
        'Chans'        : ('model', 'Chans'),
        'Samples'      : ('model', 'Samples'),
        'dropoutRate'  : ('model', 'dropoutRate'),
    }
    for arg_key, (section, cfg_key) in overrides.items():
        val = getattr(args, arg_key, None)
        if val is not None:
            cfg[section][cfg_key] = val
    return cfg


# ─────────────────────────────────────────────────────────────────────────────
# 2. REPRODUCIBILITY
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ─────────────────────────────────────────────────────────────────────────────
# 3. PLOTTING
# ─────────────────────────────────────────────────────────────────────────────

def plot_history(history, warmup_epochs, save_path):
    epochs = range(warmup_epochs + 1, len(history['train']) + 1)

    train_loss     = [e['loss']     for e in history['train']][warmup_epochs:]
    val_loss       = [e['loss']     for e in history['val']][warmup_epochs:]
    train_cls_loss = [e['cls_loss'] for e in history['train']][warmup_epochs:]
    val_cls_loss   = [e['cls_loss'] for e in history['val']][warmup_epochs:]
    train_con_loss = [e['con_loss'] for e in history['train']][warmup_epochs:]
    val_con_loss   = [e['con_loss'] for e in history['val']][warmup_epochs:]
    train_acc      = [e['acc']      for e in history['train']][warmup_epochs:]
    val_acc        = [e['acc']      for e in history['val']][warmup_epochs:]

    fig, axes = plt.subplots(1, 4, figsize=(22, 4))
    ax1, ax2, ax3, ax4 = axes

    ax1.plot(epochs, train_loss,     label='Train'); ax1.plot(epochs, val_loss,     label='Val')
    ax1.set_title('Total Loss');     ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss')
    ax1.legend(); ax1.grid(True)

    ax2.plot(epochs, train_cls_loss, label='Train'); ax2.plot(epochs, val_cls_loss, label='Val')
    ax2.set_title('Classification Loss'); ax2.set_xlabel('Epoch'); ax2.set_ylabel('Loss')
    ax2.legend(); ax2.grid(True)

    ax3.plot(epochs, train_con_loss, label='Train'); ax3.plot(epochs, val_con_loss, label='Val')
    ax3.set_title('Contrastive Loss'); ax3.set_xlabel('Epoch'); ax3.set_ylabel('Loss')
    ax3.legend(); ax3.grid(True)

    ax4.plot(epochs, train_acc,      label='Train'); ax4.plot(epochs, val_acc,      label='Val')
    ax4.set_title('Accuracy');       ax4.set_xlabel('Epoch'); ax4.set_ylabel('Accuracy')
    ax4.legend(); ax4.grid(True)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    print(f"  Plot saved to {save_path}")
    plt.show()


def plot_confusion_matrix(model, loader, device, save_path,
                          class_names=None):
    if class_names is None:
        class_names = ['Neutral', 'Sad', 'Fear', 'Happy']

    model.to(device)
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            x, labels = batch[0].to(device), batch[1].to(device)
            logits, _ = model(x)
            all_preds.extend(logits.argmax(dim=1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    cm   = confusion_matrix(all_labels, all_preds, normalize='true')
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)\
        .plot(ax=ax, cmap='Blues', values_format='.2f', colorbar=True)
    ax.set_title('Normalized Confusion Matrix')

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  Confusion matrix saved to {save_path}")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# 4. MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    cfg  = merge_config(load_config(args.config), args)

    # ── unpack config sections ────────────────────────────────────────────
    dcfg  = cfg['data']
    mcfg  = cfg['model']
    tcfg  = cfg['train']
    lcfg  = cfg['logging']

    run_name      = lcfg['run_name']
    log_dir       = lcfg['log_dir']
    chkpt_dir     = os.path.join(log_dir, run_name)
    warmup_epochs = tcfg['warmup_epochs']
    device        = tcfg['device']

    set_seed(tcfg['seed'])
    print(f"\n{'='*60}")
    print(f"  Run : {run_name}")
    print(f"  Loss: {tcfg['loss']}  |  LOO: {dcfg['leave_one_out']}")
    print(f"{'='*60}\n")

    # ── data ─────────────────────────────────────────────────────────────
    train_loader, val_loader = build_loaders(
        data_root     = dcfg['data_root'],
        dataset       = dcfg['dataset'],
        feature       = dcfg['feature'],
        n_subjects    = dcfg['n_subjects'],
        n_per_class   = dcfg['n_per_class'],
        num_workers   = dcfg['num_workers'],
        augment_train = dcfg['augment_train'],
        strategy      = dcfg['strategy'],
        leave_one_out = dcfg['leave_one_out'],
        val_subject   = dcfg['val_subject'],
        val_fraction  = dcfg['val_fraction'],
    )

    # ── model ─────────────────────────────────────────────────────────────
    model = build_model(
        nb_classes  = mcfg['nb_classes'],
        Chans       = mcfg['Chans'],
        Samples     = mcfg['Samples'],
        F1          = mcfg['F1'],
        D           = mcfg['D'],
        dropoutRate = mcfg['dropoutRate'],
        proj_hidden = mcfg['proj_hidden'],
        proj_out    = mcfg['proj_out'],
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr           = tcfg['lr'],
        weight_decay = tcfg['weight_decay'],
    )

    # ── loss function ─────────────────────────────────────────────────────
    loss_map = {
        'none'        : None,
        'contrastive' : ContrastiveLoss(temperature=tcfg['temperature']),
        'prototype'   : ContrastivePrototype(num_classes=mcfg['nb_classes']),
        'locl'        : LeaveOneOutContrastiveLearning(temperature=tcfg['temperature']),
    }
    con_loss_fn = loss_map[tcfg['loss']]

    # ── trainer ───────────────────────────────────────────────────────────
    trainer = Trainer(
        model,
        ClassificationLoss(),
        con_loss_fn,
        optimizer,
        lambda_con    = tcfg['lambda_con'],
        warmup_epochs = warmup_epochs,
        device        = device,
    )

    history = trainer.fit(train_loader, val_loader, epochs=tcfg['epochs'])

    # ── save final model ──────────────────────────────────────────────────
    final_path = os.path.join(chkpt_dir, 'model_final.pt')
    os.makedirs(chkpt_dir, exist_ok=True)
    torch.save({
        'model_state_dict'    : model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'history'             : history,
        'config'              : cfg,
    }, final_path)
    print(f"\n  Final model saved to {final_path}")

    # ── plots ─────────────────────────────────────────────────────────────
    plot_history(history, warmup_epochs,
                 save_path=os.path.join(chkpt_dir, 'plots.png'))

    plot_confusion_matrix(model, val_loader, device,
                          save_path=os.path.join(chkpt_dir, 'confusion_matrix.png'))


if __name__ == '__main__':
    main()