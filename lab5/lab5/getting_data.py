import pandas as pd

#### DATA CLEANING

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# anime_startdate

df = pd.read_csv("real_animanga.csv")

df['All_Genres'] = df[['Genre_1', 'Genre_2', 'Genre_3']].apply(
    lambda x: ','.join(x.dropna()), axis=1
)

df['Date'] = pd.to_datetime(df['Anime_StartDate'], errors='coerce').dt.to_period('M')

# drop rows where Anime_StartYearMonth or All_Genres is n/a
genre_year_data = df[['Date', 'All_Genres']].dropna()

# split the combined genres into separate rows
genre_year_data = genre_year_data.assign(All_Genres=genre_year_data['All_Genres'].str.split(','))
genre_year_data = genre_year_data.explode('All_Genres')

# count occurrences of each genre by year-month
genre_trends_by_year_month = genre_year_data.groupby(['Date', 'All_Genres']).size().unstack(fill_value=0)

top_genres = genre_trends_by_year_month.sum().sort_values(ascending=False).head(18).index
top_genres_trends = genre_trends_by_year_month[top_genres]

top_genres_trends.to_csv("top_genres_anime_start_trends.csv")
print("top_genres_anime_start_trends.csv generated successfully!!!!!")

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# anime_enddate

df = pd.read_csv("real_animanga.csv")

df['Anime_EndDate'] = df['Anime_EndDate'].fillna("Ongoing or Hiatus")

df['All_Genres'] = df[['Genre_1', 'Genre_2', 'Genre_3']].apply(
    lambda x: ','.join(x.dropna()), axis=1
)

df['Date'] = pd.to_datetime(df['Anime_EndDate'], errors='coerce').dt.to_period('M')

genre_end_year_data = df[['Date', 'All_Genres']].dropna()

genre_end_year_data = genre_end_year_data.assign(All_Genres=genre_end_year_data['All_Genres'].str.split(','))
genre_end_year_data = genre_end_year_data.explode('All_Genres')

genre_trends_by_end_year_month = genre_end_year_data.groupby(['Date', 'All_Genres']).size().unstack(fill_value=0)

top_genres_end = genre_trends_by_end_year_month.sum().sort_values(ascending=False).head(18).index

top_genres_end_trends = genre_trends_by_end_year_month[top_genres_end]

top_genres_end_trends.to_csv("top_genres_anime_end_trends.csv")
print("top_genres_anime_end_trends.csv generated successfully!!!!!")

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# manga_startdate
df = pd.read_csv("real_animanga.csv")

df['All_Genres'] = df[['Genre_1', 'Genre_2', 'Genre_3']].apply(
    lambda x: ','.join(x.dropna()), axis=1
)

df['Date'] = pd.to_datetime(df['Manga_StartDate'], errors='coerce').dt.to_period('M')

genre_year_data = df[['Date', 'All_Genres']].dropna()

genre_year_data = genre_year_data.assign(All_Genres=genre_year_data['All_Genres'].str.split(','))
genre_year_data = genre_year_data.explode('All_Genres')

genre_trends_by_year_month = genre_year_data.groupby(['Date', 'All_Genres']).size().unstack(fill_value=0)

top_genres = genre_trends_by_year_month.sum().sort_values(ascending=False).head(18).index

top_genres_trends = genre_trends_by_year_month[top_genres]

top_genres_trends.to_csv("top_genres_manga_start_trends.csv")
print("top_genres_manga_start_trends.csv generated successfully!!!!!")


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# manga_enddate
df = pd.read_csv("real_animanga.csv")

df['All_Genres'] = df[['Genre_1', 'Genre_2', 'Genre_3']].apply(
    lambda x: ','.join(x.dropna()), axis=1
)

df['Date'] = pd.to_datetime(df['Manga_EndDate'], errors='coerce').dt.to_period('M')

genre_end_year_data = df[['Date', 'All_Genres']].dropna()

genre_end_year_data = genre_end_year_data.assign(All_Genres=genre_end_year_data['All_Genres'].str.split(','))
genre_end_year_data = genre_end_year_data.explode('All_Genres')

genre_trends_by_end_year_month = genre_end_year_data.groupby(['Date', 'All_Genres']).size().unstack(fill_value=0)

top_genres_end = genre_trends_by_end_year_month.sum().sort_values(ascending=False).head(18).index

top_genres_end_trends = genre_trends_by_end_year_month[top_genres_end]

top_genres_end_trends.to_csv("top_genres_manga_end_trends.csv")
print("top_genres_manga_end_trends.csv generated successfully!!!!!")