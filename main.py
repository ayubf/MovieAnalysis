import streamlit as st
import pandas as pd
import numpy as np

"""
# :movie_camera: Analysis on MovieLens Data

In this notebook, we explore the MovieLens latest small dataset to understand patterns in movie ratings, genres, and rater behavior. 

Dataset: [MovieLens Latest Small](https://grouplens.org/datasets/movielens/latest/)

### Questions We Will Answer

1. **Raters and movies by the numbers**
   - How many raters are there?
   - How many movies have been rated?

2. **Questions about genres**
   - What are the most common movie genres?
   - What is the average rating per genre?
   - What is the average variance per genre?

3. **Movie ratings through the years**
   - Which years had the most movies released?
   - What is the average movie rating per year?

    
4. **About the raters**
   - Which rater gives the highest rating on average? Who gives the lowest?
   - Do raters who rate movies more often give higher or lower ratings on average?


"""

""


ratings = pd.read_csv("ratings.csv")
movies = pd.read_csv("movies.csv")

movies["genres"] = movies["genres"].str.split("|")
movies["year"]=  (
    movies["title"]
    .str.extract(r"\((\d{4})\)")
    .astype("float")
    .astype("Int64")
)
ratings_with_genres = pd.merge(
    ratings, 
    movies[["movieId", "genres", "year"]], 
    on="movieId", 
    how="left"
)

number_of_unique_raters = ratings["userId"].nunique()
number_of_movies_rated = ratings["movieId"].nunique()


f"""
## Raters and movies by the numbers

There are {number_of_unique_raters} raters that have rated {number_of_movies_rated} different movies.

"""

unique_genres = movies.explode("genres")["genres"].nunique()

"" 

f"""

## Questions about genres

There are {unique_genres} genres the movies can be categorized into. Below are the average ratings, variance between ratings and number of ratings for each genre.

"""


st.dataframe(
    ratings_with_genres
    .explode("genres")
    .groupby("genres")
    .agg(
        avg_rating=("rating", "mean"),
        var_of_ratings=("rating", "var"),
        num_of_ratings=("rating", "count"),
    )
)

f"""
## Movie Ratings through the years

Considering the years after 1920, the average rating per year stays within the 3-4 range for most of the years, while variance stays between 0.5 and 1.5.
"""

st.dataframe(
    ratings_with_genres
    .groupby("year")
    .agg(
        avg_rating=("rating", "mean"),
        var_of_ratings=("rating", "var"),
        num_of_ratings=("rating", "count"),
    )
)

""

avg_rating_per_user = ratings.groupby("userId").rating.mean().sort_values(ascending=False)
highest_avg_rating = avg_rating_per_user.iloc[0]
lowest_avg_rating = avg_rating_per_user.iloc[-1]
average_rating_overall = ratings.rating.mean()

"""

# About the raters


How do the critics rate? If we get an average rating per critic, our highest and lowest average rating 
are similar distances away from the mean of average ratings.

"""

c1, c2, c3 = st.columns(3)
c1.metric(
    "Highest Average Rating",
    highest_avg_rating,
)
c2.metric(
    "Lowest Average Rating",
    lowest_avg_rating,
)
c3.metric(
    "Average Rating Overall",
    round(average_rating_overall, 3)
)

user_info = ratings.groupby("userId").agg(
    avg_rating=("rating", "mean"),
    num_of_ratings=("rating", "count")
).reset_index()
corr_between_ratings_num_of_ratings = user_info["avg_rating"].corr(user_info["num_of_ratings"], method="kendall")


f"""
The correlation between average rating and the number of ratings is {round(corr_between_ratings_num_of_ratings, 3)}. 
There is a very weak negative correlation between a user's average rating and the number of ratings they have given, 
meaning that higher average ratings are slightly associated with fewer total ratings, but the trend is very small.
"""

