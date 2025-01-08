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

df = pd.read_csv('incomplete.csv')

cf = df.dropna(subset=['Genres'])

cf.to_csv('no_null.csv', index=False)

genre_split = cf['Genres'].str.split(',', expand=True)
cf['Genre_1'] = genre_split[0]
cf['Genre_2'] = genre_split[1]
cf['Genre_3'] = genre_split[2]

updated_file_path = 'animanga.csv'
cf.to_csv(updated_file_path, index=False)

# hahahahahahahahahahahahahahahahahahahahahahahahahahahahahahahahahahahahahahaha
# month mapping

df = pd.read_csv('animanga.csv')

month_mapping = {
    'Jan': '01', 'Feb': '02', 'Mar': '03', 'Apr': '04', 'May': '05', 'Jun': '06',
    'Jul': '07', 'Aug': '08', 'Sep': '09', 'Oct': '10', 'Nov': '11', 'Dec': '12'
}

# 'Start date' format as 'MM/YYYY'
df['Anime_StartDate'] = df['Start date'].str.strip()
df['Anime_StartDate'] = df['Anime_StartDate'].replace(month_mapping, regex=True)
df['Anime_StartDate'] = df['Anime_StartDate'].str.replace(r'(\d+)\s+(\d+)', r'\1/\2', regex=True)

# 'End date' format as 'MM/YYYY'
df['Anime_EndDate'] = df['End date'].str.strip()
df['Anime_EndDate'] = df['Anime_EndDate'].replace(month_mapping, regex=True)
df['Anime_EndDate'] = df['Anime_EndDate'].str.replace(r'(\d+)\s+(\d+)', r'\1/\2', regex=True)

df.drop(columns=['Start date', 'End date'], inplace=True)

df.to_csv('new_dates_ani.csv', index=False)
print('new_dates_ani.csv created successfully!')

# hahahahahahahahahhahahahahahahahahahahahahahaha

def manga_published_start(date_range):
    if pd.isna(date_range):
        return None, None
    
    parts = date_range.split(" to ")
    start_date = parts[0].strip()
    end_date = parts[1].strip() if len(parts) > 1 else None

    # format start date
    try:
        # e.g., "Dec 27, 2017"
        if len(start_date.split()) == 3:  
            start_date = pd.to_datetime(start_date, format='%b %d, %Y').strftime('%m/%d/%Y')
        # e.g., "Nov 1982"
        elif len(start_date.split()) == 2:  
            start_date = pd.to_datetime(start_date, format='%b %Y').strftime('%m/%Y')
        # e.g., "4, 1974"
        elif len(start_date.split(",")) == 2:  
            start_date = pd.to_datetime(start_date, format='%m, %Y').strftime('%m/%Y')
        # e.g., "1992" (Year only)
        elif len(start_date.split()) == 1:  
            # default to January
            start_date = f"01/{start_date}"  
        else:
            start_date = None
    except ValueError:
        start_date = None

    # format end date
    try:
        # what ? end date
        if end_date is None or end_date == "?":  
            end_date = "Ongoing or Hiatus"
        # e.g., "Feb 27, 2019"
        elif len(end_date.split()) == 3:  
            end_date = pd.to_datetime(end_date, format='%b %d, %Y').strftime('%m/%d/%Y')
        # e.g., "Nov 1987"
        elif len(end_date.split()) == 2:  
            end_date = pd.to_datetime(end_date, format='%b %Y').strftime('%m/%Y')
        # e.g., "4, 1987"
        elif len(end_date.split(",")) == 2:  
            end_date = pd.to_datetime(end_date, format='%m, %Y').strftime('%m/%Y')
        # e.g., "1999" (Year only)
        elif len(end_date.split()) == 1: 
            # Default to January 
            end_date = f"01/{end_date}"  
        else:
            end_date = None
    except ValueError:
        end_date = None

    return start_date, end_date

df[['Manga_StartDate', 'Manga_EndDate']] = df['Manga_Published'].apply(
    lambda x: pd.Series(manga_published_start(x))
)

updated_file_path = 'true_animanga.csv'
df.to_csv(updated_file_path, index=False)
print('true_animanga.csv created successfully!')

# hahahahahahahahahahahahahahahahahahahahahahahahahahahahahahahahahahahahahahahahahahahahahahahahahahaha
# PCA

num_attr = [
    'Anime_Score', 'Anime_Rank', 'Anime_Popularity', 'Anime_Members',
    'Manga_Score', 'Manga_Rank', 'Manga_Popularity', 'Manga_Members'
]

df = pd.read_csv('true_animanga.csv')

for col in num_attr:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df = df.dropna(subset=num_attr)  

genre_columns = ['Genre_1', 'Genre_2', 'Genre_3']

scaler = StandardScaler()
standardized_data = scaler.fit_transform(df[num_attr])

pca = PCA(n_components=8)
principal_components = pca.fit_transform(standardized_data)

pca_df = pd.DataFrame({
    'PC1': principal_components[:, 0],
    'PC2': principal_components[:, 1],
    'Title': df['Title'],
    'Anime_Score': df['Anime_Score'],
    'Anime_Rank': df['Anime_Rank'],
    'Anime_Popularity': df['Anime_Popularity'],
    'Anime_Members': df['Anime_Members'],
    'Manga_Score': df['Manga_Score'],
    'Manga_Rank': df['Manga_Rank'],
    'Manga_Popularity': df['Manga_Popularity'],
    'Manga_Members': df['Manga_Members'],
    'Genre_1': df['Genre_1'],
    'Genre_2': df['Genre_2'],
    'Genre_3': df['Genre_3']
})

pca_df.to_csv('pca.csv', index=False)
print("pca.csv saved successfully")

# shahhashdkahdashkhaskfgkasgfkjgsdhjkfgajhsdgfjhasgfkhgadsjkgfhjgasjdgjfasgdfsa
# changing format of dates

df = pd.read_csv('true_animanga.csv')

# convert the specified date columns to proper date formats
date_columns = ['Anime_StartDate', 'Anime_EndDate', 'Manga_StartDate', 'Manga_EndDate']

def convert_to_date(value):
    try:
        return pd.to_datetime(value, errors='coerce', dayfirst=False).date()
    except:
        return value
    
for column in date_columns:
    if column == 'Manga_EndDate':
        df[column] = df[column].apply(lambda x: x if x in ["Ongoing or Hiatus"] else convert_to_date(x))
    else:
        df[column] = df[column].apply(convert_to_date)

df.to_csv('real_animanga.csv', index=False)
print("real_animanga.csv saved successfully")
