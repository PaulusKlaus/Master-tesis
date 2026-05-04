from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # needed for 3D projection
import torch
import os


def tsne(device, encoder, loader, plot="1_axis"): # plot = " 3_axis"
    all_features = []
    all_labels = []

    encoder.eval()
    with torch.no_grad():
        for batch in loader:
            if len(batch) == 3:
                x1, x2, y = batch
                x1 = x1.to(device)
                x2 = x2.to(device)
                f, z2, p1, p2 = encoder(x1, x2)
            elif len(batch) == 2:
                x1, y = batch
                x1 = x1.to(device)
                f = encoder(x1)
            else:
                raise ValueError(f"Expected batch of length 2 or 3, got {len(batch)}")

            all_features.append(f.detach().cpu())
            all_labels.append(y.detach().cpu())

    features = torch.cat(all_features).cpu().numpy()
    labels = torch.cat(all_labels).cpu().numpy()

    # Run t-SNE in 3D
    tsne = TSNE(n_components=3, perplexity=40, random_state=42)
    features_3d = tsne.fit_transform(features)

    # Make sure output folder exists
    path = "figures/tsne"
    os.makedirs(path, exist_ok=True)

    if plot == "3_axis":
        # 3D plot
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        pairs = [(0, 1), (0, 2), (1, 2)]
        titles = ["Dim 1 vs 2", "Dim 1 vs 3", "Dim 2 vs 3"]

        for ax, (i, j), title in zip(axes, pairs, titles):
            sc = ax.scatter(features_3d[:, i], features_3d[:, j],
                            c=labels, cmap="tab10", s=20)
            ax.set_title(title)
            legend = ax.legend(*sc.legend_elements(), title="Classes")
            ax.add_artist(legend)


        plt.tight_layout()
        plt.savefig(f"{path}/tsne_3d.pdf", format="pdf", dpi=300)
        plt.close()

        print(f"3D t-SNE plot saved to {path}")
    else:
        # 3D plot
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")

        sc = ax.scatter(
            features_3d[:, 0],
            features_3d[:, 1],
            features_3d[:, 2],
            c=labels,
            cmap="tab10",
            s=20
        )

        ax.set_title("3D t-SNE")
        ax.set_xlabel("Dim 1")
        ax.set_ylabel("Dim 2")
        ax.set_zlabel("Dim 3")

        legend = ax.legend(*sc.legend_elements(), title="Classes")
        ax.add_artist(legend)

        plt.tight_layout()
        plt.savefig(f"{path}/tsne_3d.pdf", format="pdf", dpi=300)
        plt.close()



