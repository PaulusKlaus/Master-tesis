from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # needed for 3D projection
import torch
import os

def tsne(device, encoder, loader):
    all_features = []
    all_labels = []

    encoder.eval()
    with torch.no_grad():
        for x1, x2, y in loader:
            z1, z2, p1, p2 = encoder(x1.to(device), x2.to(device))
            all_features.append(z1)
            all_labels.append(y)

    features = torch.cat(all_features).cpu().numpy()
    labels = torch.cat(all_labels).cpu().numpy()

    # Run t-SNE in 3D
    tsne = TSNE(n_components=3, perplexity=40, random_state=42)
    features_3d = tsne.fit_transform(features)

    # Make sure output folder exists
    path = "figures/tsne"
    os.makedirs(path, exist_ok=True)

    # 3D plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    pairs = [(0, 1), (0, 2), (1, 2)]
    titles = ["Dim 1 vs 2", "Dim 1 vs 3", "Dim 2 vs 3"]

    for ax, (i, j), title in zip(axes, pairs, titles):
        sc = ax.scatter(features_3d[:, i], features_3d[:, j],
                        c=labels, cmap="tab10", s=20)
        ax.set_title(title)


    plt.tight_layout()
    plt.savefig(f"{path}/tsne_3d.pdf", format="pdf", dpi=300)
    plt.close()

    print(f"3D t-SNE plot saved to {path}")



