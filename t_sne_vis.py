import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch

from utils.loss_SSL import SimSiamLoss

from utils.train_ML import Trainer





trainer = Trainer(args, save_dir)
                        encoder = trainer.train(pretrained=False)
                        #train_loader = trainer.train_loader

                        trainer.train_classifier(encoder)

model.eval()
all_features = []
all_labels = []

with torch.no_grad():
    for x, y in dataloader:
        z = model.encoder(x)      # get latent features
        all_features.append(z)
        all_labels.append(y)

features = torch.cat(all_features).cpu().numpy()
labels = torch.cat(all_labels).cpu().numpy()
# Run t-SNE
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
features_2d = tsne.fit_transform(features)

# Plot
plt.figure()
plt.scatter(features_2d[:, 0], features_2d[:, 1], c=labels)
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.title("t-SNE Feature Visualization")
plt.colorbar(label="Class")
plt.show()