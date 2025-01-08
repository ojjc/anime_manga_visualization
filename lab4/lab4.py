import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances
from sklearn.manifold import MDS
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

num_attr = [
    'Anime_Score','Anime_Rank','Anime_Popularity','Anime_Members',
    'Manga_Score','Manga_Rank','Manga_Popularity','Manga_Members'
]

cat_attr = ['Genres', 'Stream_type']

df = pd.read_csv('top1000_animanga.csv')

# handle any missing NaN values
df = df[num_attr + cat_attr].dropna()

# get primary genre (first genre from list)
df['Primary_Genre'] = df['Genres'].apply(lambda x: x.split(',')[0].strip())

# limit to the top 4 genres: action, comedy, drama, adventure. 
## if it doesn't fit, classify into 'Other'
top_genres = ['Action', 'Comedy', 'Drama', 'Adventure', 'Fantasy',
                'Mystery', 'Horror', 'Slice of Life', 'Romance']
df['Primary_Genre'] = df['Primary_Genre'].apply(lambda x: x if x in top_genres else 'Other')

# standardize
scaler = StandardScaler()
standardized_data = scaler.fit_transform(df[num_attr])

# perform PCA (8 components, top 2 go to PCA projection)
pca = PCA(n_components=8)
principal_components = pca.fit_transform(standardized_data)

# PCA projection + top genre for coloring in d3
pca_df = pd.DataFrame({
    'PC1': principal_components[:, 0],
    'PC2': principal_components[:, 1],
    'Genres': df['Primary_Genre'],
    'Stream_type': df['Stream_type'],
    'Anime_Score': df['Anime_Score'],
    'Anime_Rank': df['Anime_Rank'],
    'Anime_Popularity': df['Anime_Popularity'],
    'Anime_Members': df['Anime_Members'],
    'Manga_Score': df['Manga_Score'],
    'Manga_Rank': df['Manga_Rank'],
    'Manga_Popularity': df['Manga_Popularity'],
    'Manga_Members': df['Manga_Members']
})

pca_df.to_csv('pca_projection_genres_stream.csv', index=False)
print("pca_projection_genres_8.csv saved successfully")

# ====================================================
# MDS w/ Euclidean Distance (data points)

num_attr = [
    'Anime_Score', 'Anime_Rank', 'Anime_Popularity', 'Anime_Members',
    'Manga_Score', 'Manga_Rank', 'Manga_Popularity', 'Manga_Members'
]
cat_attr = ['Genres', 'Stream_type']

df = pd.read_csv('top1000_animanga.csv')

df = df[num_attr + cat_attr].dropna()

# limit Genres to the top 8 genres, classify others as "Other"
top_genres = ['Action', 'Comedy', 'Drama', 'Adventure', 'Fantasy',
              'Mystery', 'Horror', 'Slice of Life', 'Romance']
df['Genres'] = df['Genres'].apply(lambda x: x.split(',')[0].strip())
df['Genres'] = df['Genres'].apply(lambda x: x if x in top_genres else 'Other')

# compute Euclidean distance matrix
euclidean_dist = pairwise_distances(df[num_attr], metric='euclidean')

# perform MDS (Metric Multidimensional Scaling)
mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
mds_euclidean_result = mds.fit_transform(euclidean_dist)

# create a DataFrame with MDS results and corresponding categorical values
mds_euclidean_df = pd.DataFrame(mds_euclidean_result, columns=['MDS1', 'MDS2'])
mds_euclidean_df['Genres'] = df['Genres'].values
mds_euclidean_df['Stream_type'] = df['Stream_type'].values
mds_euclidean_df['Anime_Score'] = df['Anime_Score'].values
mds_euclidean_df['Anime_Rank'] = df['Anime_Rank'].values
mds_euclidean_df['Anime_Popularity'] = df['Anime_Popularity'].values
mds_euclidean_df['Anime_Members'] = df['Anime_Members'].values

mds_euclidean_df['Manga_Score'] = df['Manga_Score'].values
mds_euclidean_df['Manga_Rank'] = df['Manga_Rank'].values
mds_euclidean_df['Manga_Popularity'] = df['Manga_Popularity'].values
mds_euclidean_df['Manga_Members'] = df['Manga_Members'].values

# save to CSV
mds_euclidean_df.to_csv('mds_euclidean_with_categories.csv', index=False)
print("mds_euclidean_with_categories.csv saved successfully")

# ==========================================================
# k means clustering via elbow method
# getting rid of null

df = pd.read_csv('top1000_animanga.csv')

num_attr = [
    'Anime_Score', 'Anime_Rank', 'Anime_Popularity', 'Anime_Members',
    'Manga_Score', 'Manga_Rank', 'Manga_Popularity', 'Manga_Members'
]
cat_attr = ['Genres', 'Stream_type']

# drop rows with missing values in the specified columns
df = df[num_attr + cat_attr].dropna()

# scale numerical data for clustering
scaler = StandardScaler()
standardized_data = scaler.fit_transform(df[num_attr])

# kmeans parameters args
kmeans_args = {
    "init": "random",
    "n_init": 10,  
    "random_state": 42,
}

# elbow method
distortions = []
K = range(1, 11)
for k in K:
    kmeans = KMeans(n_clusters=k, **kmeans_args)
    kmeans.fit(standardized_data)
    distortions.append(kmeans.inertia_)

# plot elbow
plt.figure(figsize=(8, 5))
plt.plot(K, distortions, 'bx-')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Distortion')
plt.title('Elbow Method for Optimal k')
plt.savefig('elbow_plot_null.png')
plt.show()

optimal_k = 4

# perform kmeans clustering
kmeans = KMeans(init="random", n_clusters=optimal_k, n_init=10, random_state=42)
kmeans.fit(standardized_data)
df['Cluster'] = kmeans.labels_ 

df.to_csv('clustered_animanga_null.csv', index=False)
print("clustered_animanga_null.csv saved successfully")


# ==========================================================
# k means clustering via elbow method
# using null values and imputing numerical columns with the mean

# df = pd.read_csv('top1000_animanga.csv')

# num_attr = [
#     'Anime_Score', 'Anime_Rank', 'Anime_Popularity', 'Anime_Members',
#     'Manga_Score', 'Manga_Rank', 'Manga_Popularity', 'Manga_Members'
# ]
# cat_attr = ['Genres', 'Stream_type']

#  # impute numerical columns with the mean
# df[num_attr] = df[num_attr].fillna(df[num_attr].mean()) 
# # fill categorical columns with 'Unknown'
# df[cat_attr] = df[cat_attr].fillna('Unknown')  

# # limit genres to top 8 categories, classify others as "Other"
# top_genres = ['Action', 'Comedy', 'Drama', 'Adventure', 'Fantasy',
#               'Mystery', 'Horror', 'Slice of Life', 'Romance']
# df['Genres'] = df['Genres'].apply(lambda x: x.split(',')[0].strip() if isinstance(x, str) else 'Unknown')
# df['Genres'] = df['Genres'].apply(lambda x: x if x in top_genres else 'Other')

# # standardize numerical attributes
# scaler = StandardScaler()
# standardized_data = scaler.fit_transform(df[num_attr])

# #initialize kmeans parameters
# kmeans_args = {
#     "init": "random",
#     "n_init": 10,
#     "random_state": 42,
# }

# # elbow method to find optimal k
# distortions = []
# K = range(1, 11)
# for k in K:
#     kmeans = KMeans(n_clusters=k, **kmeans_args)
#     kmeans.fit(standardized_data)
#     distortions.append(kmeans.inertia_)

# # elbow method
# plt.figure(figsize=(8, 5))
# plt.plot(K, distortions, 'bx-')
# plt.xlabel('Number of Clusters (k)')
# plt.ylabel('Distortion')
# plt.title('Elbow Method for Optimal k')
# plt.savefig('elbow_plot_impute.png')
# plt.show()

# optimal_k = 4 

# # perform K-Means clustering
# kmeans = KMeans(init="random", n_clusters=optimal_k, n_init=10, random_state=42)
# kmeans.fit(standardized_data)
# df['Cluster'] = kmeans.labels_

# # save clustered data
# df.to_csv('clustered_animanga_imp.csv', index=False)
# print("clustered_animanga_imp.csv saved successfully")

#############################################################################################

# PCA

df = pd.read_csv('clustered_animanga_null.csv')

# get primary genre (first genre from list)
df['Primary_Genre'] = df['Genres'].apply(lambda x: x.split(',')[0].strip())

# limit Genres to the top 8 genres, classify others as "Other"
top_genres = ['Action', 'Comedy', 'Drama', 'Adventure', 'Fantasy',
              'Mystery', 'Horror', 'Slice of Life', 'Romance']
df['Genres'] = df['Genres'].apply(lambda x: x.split(',')[0].strip())
df['Genres'] = df['Genres'].apply(lambda x: x if x in top_genres else 'Other')

# standardize
scaler = StandardScaler()
standardized_data = scaler.fit_transform(df[num_attr])

# perform PCA (8 components, top 2 go to PCA projection)
pca = PCA(n_components=8)
principal_components = pca.fit_transform(standardized_data)

# PCA projection + top genre for coloring in d3
pca_df = pd.DataFrame({
    'PC1': principal_components[:, 0],
    'PC2': principal_components[:, 1],
    'Genres': df['Primary_Genre'],
    'Stream_type': df['Stream_type'],
    'Cluster': df['Cluster']
})

pca_df.to_csv('clu_pca_projection_genres.csv', index=False)
print("clu_pca_projection_genres.csv saved successfully")

#############################################################################################

# getting loadings for biplot 
# from https://bioturing.medium.com/how-to-read-pca-biplots-and-scree-plots-186246aae063

# get loadings (projections of original attributes on the PC1-PC2 space)
loadings = pca.components_[:2].T 

loadings_df = pd.DataFrame(loadings, columns=['PC1_loading', 'PC2_loading'])
loadings_df['Attribute'] = num_attr
loadings_df.to_csv('clu_biplot_loadings.csv', index=False)
print("clu_biplot_loadings.csv saved successfully")

#############################################################################################

# MDS w/ Euclidean Distance (data points)

euclidean_dist = pairwise_distances(df[num_attr], metric='euclidean')

# compute metric MDS (Euclidean)
mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
mds_euclidean_result = mds.fit_transform(euclidean_dist)

mds_euclidean_df = pd.DataFrame(mds_euclidean_result, columns=['MDS1', 'MDS2'])
mds_euclidean_df['Cluster'] = df['Cluster'].values

mds_euclidean_df.to_csv('clu_mds_euclidean.csv', index=False)
print("clu_mds_euclidean.csv saved successfully")