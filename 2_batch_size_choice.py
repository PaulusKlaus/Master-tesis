import matplotlib.pyplot as plt
from utils.log_parser import parse_training_log, summarize

import seaborn as sns
import matplotlib.pyplot as plt

import re
from collections import defaultdict
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path


def batch_size_test(paths, group_by_blocks=False):
    summary_batch_size = summarize(paths, "batch_size", group_by_blocks)
    return summary_batch_size

def plot_bs(df):

    sns.boxplot(data=df, x="batch_size", y="binary_acc")
    plt.title("Binary Accuracy by batch_size")
    plt.show()

    sns.boxplot(data=df, x="batch_size", y="test_acc")
    plt.title("Test Accuracy by batch_size")
    plt.show()

    sns.scatterplot(data=df, x="test_acc", y="binary_acc", hue="batch_size")
    plt.title("Trade-off: Test vs Binary Accuracy")
    plt.show()





paths =[#"checkpoint/SSF_PU_0416-122923/training.log",
        #"checkpoint/SSF_PU_0416-163602/training.log" # 128 batch size
        "checkpoint/SSF_CWRU_0416-155803/training.log",
       ########## "checkpoint/SSF_CWRU_0416-120355/training.log",  # not sure if itis the same augmentetion
        ]
batch_size_test(paths)


df = parse_training_log("checkpoint/SSF_CWRU_0416-105859/training.log")
#plot_bs(df)





log_path = Path('checkpoint/SSF_PU_0416-122923/training.log')
out_pdf = Path('figures/batch_size_loss_comparison.pdf')
text = log_path.read_text(errors='ignore').splitlines()

runs=[]
current=None
run_id=-1
batch_pat=re.compile(r'batch_size:\s*(\d+)')
epoch_pat=re.compile(r'Epoch\s+(\d+)\s+Train loss\s+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s+\|\s+Val loss\s+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)')
run_pat=re.compile(r'RUN:')

for line in text:
    if run_pat.search(line):
        if current and current.get('epochs'):
            runs.append(current)
        run_id += 1
        current={'run_id': run_id, 'batch_size': None, 'epochs': [], 'train': [], 'val': []}
    m=batch_pat.search(line)
    if m:
        if current is None:
            run_id += 1
            current={'run_id': run_id, 'batch_size': None, 'epochs': [], 'train': [], 'val': []}
        current['batch_size']=int(m.group(1))
    m=epoch_pat.search(line)
    if m:
        if current is None:
            run_id += 1
            current={'run_id': run_id, 'batch_size': None, 'epochs': [], 'train': [], 'val': []}
        current['epochs'].append(int(m.group(1)))
        current['train'].append(float(m.group(2)))
        current['val'].append(float(m.group(3)))
if current and current.get('epochs'):
    runs.append(current)

# Remove incomplete or no batch runs
runs=[r for r in runs if r['batch_size'] is not None and len(r['epochs'])>0]
by_batch=defaultdict(list)
for r in runs:
    by_batch[r['batch_size']].append(r)

# Summary table
print(f'Parsed {len(runs)} runs')
for bs in sorted(by_batch):
    vals=[r['val'][-1] for r in by_batch[bs]]
    bests=[min(r['val']) for r in by_batch[bs]] # lower is better because loss is negative
    print(f'batch {bs}: n={len(vals)}, final val mean={np.mean(vals):.6f}, best val mean={np.mean(bests):.6f}')

plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 13,
    'axes.labelsize': 15,
    'legend.fontsize': 13,
    'figure.titlesize': 14,
})

styles=['-', '--', '-.', ':']
markers=['o','s','^','D','v','P','X']

def aggregate(metric):
    data={}
    for bs, rr in by_batch.items():
        max_epoch=min(max(r['epochs']) for r in rr)
        epochs=np.arange(0,max_epoch+1)
        mat=[]
        for r in rr:
            d=dict(zip(r['epochs'], r[metric]))
            mat.append([d[e] for e in epochs])
        mat=np.asarray(mat,float)
        data[bs]=(epochs, mat.mean(axis=0), mat.std(axis=0), mat)
    return data

val_data=aggregate('val')
train_data=aggregate('train')

with PdfPages(Path('figures/Validation_loss_by_batch_size_pu.pdf')) as pdf:
    fig, ax = plt.subplots(figsize=(7,7))
    for idx,bs in enumerate(sorted(val_data)):
        epochs, mean, std, mat = val_data[bs]
        line, = ax.plot(epochs, mean, linestyle=styles[idx%len(styles)], marker=markers[idx%len(markers)], markevery=2, linewidth=1.8, label=f'Batch {bs} val mean (n={mat.shape[0]})')
        ax.fill_between(epochs, mean-std, mean+std, alpha=0.12)
   # ax.set_title('Validation loss by batch size (mean +/- std across runs)')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation loss')
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)
print(Path('figures/Validation_loss_by_batch_size.pdf'))

with PdfPages(Path('figures/Training_loss_by_batch_size_pu.pdf')) as pdf:
    fig, ax = plt.subplots(figsize=(7,7))
    for idx,bs in enumerate(sorted(train_data)):
        epochs, mean, std, mat = train_data[bs]
        ax.plot(epochs, mean, linestyle=styles[idx%len(styles)], marker=markers[idx%len(markers)], markevery=2, linewidth=1.8, label=f'Batch {bs} train mean (n={mat.shape[0]})')
        ax.fill_between(epochs, mean-std, mean+std, alpha=0.12)
    #ax.set_title('Training loss by batch size (mean +/- std across runs)')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Training loss')
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

print(Path('figures/Training_loss_by_batch_size.pdf'))

with PdfPages(Path('figures/Train_vs_validation_loss_by_batch_size.pdf')) as pdf:

    fig, ax = plt.subplots(figsize=(10.5,6.5))
    for idx,bs in enumerate(sorted(val_data)):
        epochs, vmean, vstd, _ = val_data[bs]
        _, tmean, tstd, _ = train_data[bs]
        ax.plot(epochs, tmean, linestyle=styles[idx%len(styles)], linewidth=1.5, label=f'Batch {bs} train')
        ax.plot(epochs, vmean, linestyle=styles[idx%len(styles)], linewidth=2.2, marker=markers[idx%len(markers)], markevery=7, label=f'Batch {bs} val')
 #   ax.set_title('Train vs validation loss by batch size (means)')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

print(Path('figures/Train_vs_validation_loss_by_batch_size.pdf'))