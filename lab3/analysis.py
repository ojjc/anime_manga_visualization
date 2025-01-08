import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances
from sklearn.manifold import MDS

import matplotlib.pyplot as plt
import seaborn as sns


# cleaning original animanga.csv so that it converts string data to int data
data = pd.read_csv('animanga.csv')

# function to convert string values to int
def clean_member_count(value):
    if pd.isna(value):  # check for null and don't do nothing to it :3
        return value
    return int(value.replace(',', '').strip()) 

# apply function to some columns
data['Anime_Members'] = data['Anime_Members'].apply(clean_member_count)
data['Manga_Members'] = data['Manga_Members'].apply(clean_member_count)

data.to_csv('numerical_animanga.csv', index=False)

#############################################################################################

# correlation matrix from numerical attributes
data = pd.read_csv('numerical_animanga.csv')

num_attr = [
    'Anime_Score', 'Anime_Rank', 'Anime_Popularity', 'Anime_Members',
    'Manga_Score', 'Manga_Rank', 'Manga_Popularity', 'Manga_Members'
]

# handling NaN values
def clean_numerical_column(value):
    try:
        return float(value) 
    except (ValueError, TypeError):  
        return np.nan 

for attr in num_attr:
    data[attr] = data[attr].apply(clean_numerical_column)

# if NaN val, drop instead 
data = data.dropna(subset=num_attr)

# corr matrix + save matrix
correlation_matrix = data[num_attr].corr()
correlation_matrix.to_csv('correlation_matrix.csv', index=False)
print("corr matrix saved successfully")

#############################################################################################

# realizing now that 8000> data points can make visualization difficult
# now limiting to top 1000 ranked anime and their respective manga

df = pd.read_csv('numerical_animanga.csv')

df['Anime_Rank'] = pd.to_numeric(df['Anime_Rank'], errors='coerce')
df = df.dropna(subset=['Anime_Rank'])

# filter out rows where Anime_Rank is less than or equal to 1000
filtered_df = df[df['Anime_Rank'] <= 1000]

filtered_df.to_csv('top1000_animanga.csv', index=False)

print("top 1000 ranked animanga saved successfully")
print(f"original dataset had {len(df)} rows.")
print(f"filtered dataset has {len(filtered_df)} rows.")

num_attr = [
    'Anime_Score','Anime_Rank','Anime_Popularity','Anime_Members',
    'Manga_Score','Manga_Rank','Manga_Popularity','Manga_Members'
]

#############################################################################################

# PCA

df = pd.read_csv('top1000_animanga.csv')

# handle any missing NaN values
df = df[num_attr + ['Genres']].dropna()

# get primary genre (first genre from list)
df['Primary_Genre'] = df['Genres'].apply(lambda x: x.split(',')[0].strip())

# limit to the top 4 genres: action, comedy, drama, adventure. 
## if it doesn't fit, classify into 'Other'
top_genres = ['Action', 'Comedy', 'Drama', 'Adventure']
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
    'Primary_Genre': df['Primary_Genre']
})
pca_df.to_csv('pca_projection_genres.csv', index=False)
print("pca_projection_genres.csv saved successfully")

# saving explained variance to scree_data.csv
variance_df = pd.DataFrame({
    'Component': [f'PC{i+1}' for i in range(8)],
    'ExplainedVariance': pca.explained_variance_ratio_[:8]
})
variance_df.to_csv('scree_data.csv', index=False)
print("scree_data.csv saved successfully")

#############################################################################################

# getting loadings for biplot 
# from https://bioturing.medium.com/how-to-read-pca-biplots-and-scree-plots-186246aae063

# get loadings (projections of original attributes on the PC1-PC2 space)
loadings = pca.components_[:2].T 

loadings_df = pd.DataFrame(loadings, columns=['PC1_loading', 'PC2_loading'])
loadings_df['Attribute'] = num_attr
loadings_df.to_csv('biplot_loadings.csv', index=False)
print("biplot_loadings.csv saved successfully")

#############################################################################################

# MDS w/ Euclidean Distance (data points)

euclidean_dist = pairwise_distances(df[num_attr], metric='euclidean')

# compute metric MDS (Euclidean)
mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
mds_euclidean_result = mds.fit_transform(euclidean_dist)

mds_euclidean_df = pd.DataFrame(mds_euclidean_result, columns=['MDS1', 'MDS2'])
mds_euclidean_df.to_csv('mds_euclidean.csv', index=False)
print("mds_euclidean.csv saved successfully")

#############################################################################################

# MDS with 1-|Correlation| Distance (attributes)

dissimilarity_matrix = 1 - np.abs(correlation_matrix)

mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
mds_coordinates = mds.fit_transform(dissimilarity_matrix)

# saving MDS (correlation based) to CSV
mds_correlation_df = pd.DataFrame(
    mds_coordinates, columns=['MDS1', 'MDS2'], index=num_attr
)
mds_correlation_df['Attribute'] = num_attr  
mds_correlation_df.to_csv('mds_correlation.csv', index=False)
print("mds_correlation.csv saved successfully")

print("========== all csv files produces successfully! ==========")