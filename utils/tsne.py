import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def tsne(device, encoder, loader):

    all_features = []
    all_labels = []
    encoder.eval()
    with torch.no_grad():
        for x1,x2, y in loader:
            z1, z2, p1, p2 = encoder(x1.to(device),x2.to(device))      # get latent features
            x = x1.view(x1.size(0), -1)   # (B, C*L)

            all_features.append(z1)
            all_labels.append(y)


    features = torch.cat(all_features).cpu().numpy()
    labels = torch.cat(all_labels).cpu().numpy()
    # Run t-SNE
    tsne = TSNE(n_components=2, perplexity=50, random_state=42)
    features_2d = tsne.fit_transform(features)

    # Plot
    plt.figure()
    plt.scatter(features_2d[:, 0], features_2d[:, 1], c=labels)
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.title("t-SNE Feature Visualization")
    plt.colorbar(label="Class")
    plt.show()




