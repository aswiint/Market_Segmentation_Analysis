import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import LabelEncoder
from yellowbrick.features.pca import PCADecomposition


# R> library("MSA")
# R> data("mcdonalds", package = "MSA")
df = pd.read_csv("mcdonalds.csv")
df.loc[(df["Like"] == 'I love it!+5'), "Like"] = 5
df.loc[(df["Like"] == 'I hate it!-5'), "Like"] = -5
# R> names(mcdonalds)
print("Dataframe Columns: \n", df.columns)
# R> dim(mcdonalds)
print("Dataframe Shape: ", df.shape)
# R> head(mcdonalds, 3)
pd.set_option("display.max_columns", None)
print("Dataframe Head: \n", df.head())
# R> MD.x <- as.matrix(mcdonalds[, 1:11])
# R> MD.x <- (MD.x == "Yes") + 0
frame = df.drop(columns=["Like", "Age", "VisitFrequency", "Gender"])
for col in frame.columns:
    frame[col] = frame[col].map({"Yes": 0, "No": 1})
# R> round(colMeans(MD.x), 2)
print("DataFrame Mean: \n", frame.mean())
print("Binary Matrix: \n", frame.head())

df_s = frame
df_s = pd.DataFrame(preprocessing.normalize(df_s), columns=frame.columns)
df_s = df_s.join(df["Like"])
corr = df_s.corr()
sns.heatmap(corr, cmap=sns.color_palette("RdBu_r", 7), center=0, vmin=-1, vmax=1)
plt.title("Correlation Heatmap -1")
plt.show()

c = df_s.corr().values
d = sch.distance.pdist(c)
L = sch.linkage(d, method="complete")
ind = sch.fcluster(L, 0.5*d.max(), "distance")
columns = [df_s.columns.tolist()[i] for i in list((np.argsort(ind)))]
df_s = df_s.reindex(columns, axis=1)
print(df_s.head())
corr = df_s.corr()
sns.heatmap(corr, cmap=sns.color_palette("RdBu_r", 7), center=0, vmin=-1, vmax=1)
plt.title("Correlation Ordered by Clusters Heatmap -2")
plt.show()

df_m = df_s.groupby(["Like"]).mean()
sns.heatmap(df_m, cmap="YlOrRd", linewidth=0.4)
plt.title("Grouped Features Heatmap -3")
plt.show()
sns.clustermap(df_m, cmap="YlOrRd", linewidth=0.4)
plt.title("Grouped Features ClusterMap -4")
plt.show()
# R> MD.pca <- prcomp(MD.x)
# R> summary(MD.pca)
# R> print(MD.pca, digits = 1)
df_new = frame.values
pca = PCA(n_components=11)
principalComponents = pca.fit_transform(df_new)
principalDf = pd.DataFrame(data=principalComponents, columns=["PC1", "PC2", "PC3", "PC4", "PC5", "PC6", "PC7", "PC8",
                                                              "PC9", "PC10", "PC11",])
loadings = pca.components_
num_pc = pca.n_features_in_
pc_list = ["PC"+str(i) for i in list(range(1, num_pc+1))]
loadings_df = pd.DataFrame.from_dict(dict(zip(pc_list, loadings)))
loadings_df["variable"] = frame.columns.values
loadings_df = loadings_df.set_index("variable")
print("Rotation (n x k) = (11 x 11): \n", loadings_df)
print("Explained Variance Ratio: \n", np.cumsum(pca.explained_variance_ratio_))
# R> library("flexclust")
# R> plot(predict(MD.pca), col = "grey")
# R> projAxes(MD.pca)
x = df_s.drop(columns=["Like"])
df_s["Like"] = df_s["Like"].astype("int")
y = df_s["Like"]
visualizer = PCADecomposition(scale=True, proj_features=True)
visualizer.ax.set_title("PCA Bi-Plot -5")
visualizer.fit_transform(x, y)
visualizer.show()
# R> set.seed(1234)
# R> MD.km28 <- stepFlexclust(MD.x, 2:8, nrep = 10, + verbose = FALSE)
# R> MD.km28 <- relabel(MD.km28)
# R> plot(MD.km28, xlab = "number of segments")
wcss = {}
for i in range(1, 9):
    kmeans = KMeans(n_clusters=i).fit(principalDf)
    wcss[i] = kmeans.inertia_
plt.plot(list(wcss.keys()), list(wcss.values()), "-o", color="black")
plt.title("Scree Plot -6")
plt.xlabel("Number of Segments")
plt.ylabel("wcss")
plt.xticks(range(1, 9))
plt.show()
# R> set.seed(1234)
# R> MD.b28 <- bootFlexclust(MD.x, 2:8, nrep = 10, + nboot = 100)
# R> plot(MD.b28, xlab = "number of segments", + ylab = "adjusted Rand index")
x = df_s.drop(columns=["Like"])
np.random.seed(1234)
nboot = 100
n_clusters_range = range(2, 9)
# Perform bootstrapping
ari_scores_all = []
for n_clusters in n_clusters_range:
    ari_scores = []
    cluster_percentages = []
    for j in range(nboot):
        kmeans = KMeans(n_clusters=n_clusters, random_state=j).fit(x)
        true_labels = LabelEncoder().fit_transform(df_s["Like"])
        ari_scores.append(adjusted_rand_score(true_labels, kmeans.labels_))
        unique_labels, cluster_counts = np.unique(kmeans.labels_, return_counts=True)
        percentages = cluster_counts / len(x)
        cluster_percentages.append(percentages)
    ari_scores_all.append(ari_scores)

plt.boxplot(ari_scores_all)
plt.title("Global Stability of k-means Segmentation Solutions -7")
plt.xlabel("Number of segments")
plt.ylabel("Adjusted Rand Index")
plt.xticks(range(1, len(n_clusters_range)+1), list(n_clusters_range))
plt.show()
# R> histogram(MD.km28[["4"]], data = MD.x, xlim = 0:1)
plt.figure(figsize=(12, 8))
for i in range(4):
    plt.subplot(2, 2, i+1)
    plt.hist([s[i] for s in cluster_percentages], bins=20, color="skyblue", edgecolor="black", alpha=0.7)
    plt.xlabel("Similarity")
    plt.ylabel("Percentage of Total")
    plt.title(f"""Cluster-{i+1}""")
plt.suptitle("Gorge Plot of Four Segment K-means Solution -8")
plt.tight_layout()
plt.show()
# R> library("flexmix")
# R> set.seed(1234)
# R> MD.m28 <- stepFlexmix(MD.x ~ 1, k = 2:8, nrep = 10, + model = FLXMCmvbinary(), verbose = FALSE)
# R> MD.m28
# R> plot(MD.m28, + ylab = "value of information criteria (AIC, BIC, ICL)")
# R> MD.m4 <- getModel(MD.m28, which = "4")
# R> table(kmeans = clusters(MD.k4), + mixture = clusters(MD.m4))
models = [GaussianMixture(n_components=n, random_state=42).fit(x) for n in range(2, 9)]
plt.plot(range(2, 9), [m.bic(x) for m in models], label="BIC")
plt.plot(range(2, 9), [m.aic(x) for m in models], label="AIC")
plt.legend()
plt.xlabel("Number of Component")
plt.ylabel("Value of information criteria (AIC, BIC)")
plt.title("Information Criteria for the Binary Distributions Mixture Models -9")
plt.show()

