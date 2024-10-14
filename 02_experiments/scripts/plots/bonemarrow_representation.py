import torch
import numpy as np
import pandas as pd
import anndata as ad
import multiDGD
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data = ad.read_h5ad("./01_data/human_bonemarrow.h5ad")

model = multiDGD.DGD.load(data=data, save_dir="./03_results/models/", model_name="human_bonemarrow_l20_h2-3_test50e")

reps = model.representation.z.detach()

# create a pca of the reps
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

pca = PCA(n_components=2)
pca.fit(reps.cpu().numpy())
# plot the representations
reps_transformed = pca.transform(reps.cpu().numpy())
clusters = model.gmm.clustering(torch.tensor(reps)).detach().cpu().numpy()

# plot the pca
data = data[data.obs["train_val_test"] == "train"]

df_celltypes = pd.DataFrame(reps_transformed, columns=["PC 1", "PC 2"])
df_celltypes["type"] = "original"
df_celltypes["component"] = clusters
df_celltypes["component"] = df_celltypes["component"].astype(str)
df_celltypes["celltype"] = data.obs["cell_type"].values
unique_values = data.obs["cell_type"].cat.categories

# specify the font size
plt.rcParams.update({"font.size": 10})

fig, axs = plt.subplots(1, 1, figsize=(9, 4))

# remove the white lines arount the dots
sns.scatterplot(data=df_celltypes, x="PC 1", y="PC 2", hue="celltype", s=2, alpha=0.5, ec=None, ax=axs)
axs.set_title("Bone marrow representations")
# change the legend labels so that they show the index of the celltype from the unique_values list
handles, labels = axs.get_legend_handles_labels()
axs.legend(
    handles,
    [str(np.where(unique_values == label)[0][0]) + " (" + label + ")" for label in labels],
    title="Cell type",
    bbox_to_anchor=(1, 1.05),
    loc="upper left",
    markerscale=4,
    fontsize=10,
    ncol=2,
    frameon=False,
)
# remove the ticks
axs.set_xticks([])
axs.set_yticks([])

# annotate the plot with the celltype names (on means)
for i, celltype in enumerate(unique_values):
    mean = np.mean(reps_transformed[data.obs["cell_type"] == celltype], axis=0)
    plt.annotate(i, (mean[0], mean[1]), fontsize=10)

plt.tight_layout()
# save the plot
plt.savefig("./03_results/figures/bonemarrow_representation.pdf")
