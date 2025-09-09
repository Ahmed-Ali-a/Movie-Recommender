import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

#read data
movie=pd.read_csv("tmdb_5000_movies.csv")

#cleaning data
for col in ['overview','genres','keywords']:
    movie[col]=movie[col].fillna('')

#combine column
movie['combined']=movie['overview']+' '+ movie['genres']+' '+movie['keywords']

#Vectorizing Text
tfidf=TfidfVectorizer(stop_words="english")
tfidf_matrix=tfidf.fit_transform(movie['combined'])

#compute similarity
cosine_sim=cosine_similarity(tfidf_matrix,tfidf_matrix)

#look-up
movies=movie.reset_index()
indices=pd.Series(movies.index,index=movie['title']).drop_duplicates()

#Recommender_func
def recommended_movies(title,n=5):
    if title not in indices:
        return f"Movie {title} not found."

    idx=indices[title]
    sim_score=list(enumerate(cosine_sim[idx]))
    sim_score=sorted(sim_score, key=lambda x: x[1], reverse=True)
    sim_score=sim_score[1:n+1]

    movie_indices = [i[0] for i in sim_score]
    return movies[['title', 'vote_average', 'vote_count']].iloc[movie_indices]

#Example

if __name__=='__main__':
    print("Movies similar to 'The Dark Knight Rises':\n")
    print(recommended_movies("The Dark Knight Rises", n=5))
