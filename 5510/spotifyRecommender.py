#!/usr/bin/env python
# coding: utf-8

# # Spotify Song Recommendations Using Clustering Models
# 
# The idea for this project actually began when I was working on an idea for my data mining project and as I was working through that project, I thought that I could build off of it with this class. So to catch everyone up, ever since I have found Spotify, I have been in love with that platform. Especially with how much data they use for their recommendations and even their Wrapped when they provide users with all of their listening data. That level of work has always had me wanting to work on a data project using Spotify data and while I was working on the Data Mining Project, I realized that I was could build on my project that was finding what aspects of a song make it popular, to try and see if I could make my own music recommeder model through clustering. 
# 
# I had a lot of fun with this project and I can't wait to get into what I was able to accomplish.

# In[1]:


import os
import numpy as np
import pandas as pd
import seaborn as sns
import plotly.express as px 
import matplotlib.pyplot as plt
import json
import base64
import difflib
import spotipy
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import euclidean_distances
from scipy.spatial.distance import cdist
from collections import defaultdict
from dotenv import load_dotenv
from requests import post, get
from spotipy.oauth2 import SpotifyClientCredentials

import warnings
warnings.filterwarnings("ignore")


# ## Models:
# 
# I will be doing the standard process of an Exploratory Data Analysis (EDA) which involves cleaning the data and gaining some insight on trends through some visualizations. For the unsupervised learning models, I will be creating two Cluster models, one for songs and one for genres using K-Means. 

# ## Data: 
# 
# Knowing that I wanted to use Spotify data for this project, there were two options I could go to aquire data. I could use the Spotify API and do some work to make it even more usable. Or I could go scroll through Kaggle until I found a dataset that fit the criteria of my project. I decided to go the Kaggle route because I am much more comfortable downloading from there than working with APIs.
# 
# https://www.kaggle.com/datasets/zaheenhamidani/ultimate-spotify-tracks-db
# 
# Specifically I used the Spotify Tracks DB that is available at the link above. This is the data set that I used for song features when making predictions for what aspects of a song increase the chances of it being popular. I became very familiar with the dataset for the project and continued using it for this one.
# 
# Specific features that are important for the data include:
# * Name: The name of the song. This is incredibly important to actually recommend songs in the future
# * Year: When the song was released. This helps identify a song when requesting a recommendation as many songs have the same names
# * Genres: This is important when clustering because people tend to want to listen to songs from the same style of music
# * Artists: Who created the song. This isn't as important of a variable to me because the goal is to recommend music that people enjoy which shouldn't matter the name attached to the song
# * General Song Features: The rest of the features in the dataset are all related to song features such as danceability, loudness, energy, etc. I don't think it is important to list them all here as you will see them represented in the EDA

# In[2]:


# these are the datasets that I used for this project
dfTracks = pd.read_csv('data/data.csv')
dfGenres = pd.read_csv('data/data_by_genres.csv')
dfYear = pd.read_csv('data/data_by_year.csv')
dfTracks.head()


# ### Data Description
# 
# There are 19 columns of data and 170653 entries that make up this data set. 
# 
# There are also no missing or null values which helps save a step when cleaning the data. Columns have a range of data types that include float64, int64, and object. There are a lot of columns that I feel need to be worked on still to help make more sense of the data.

# In[3]:


dfTracks.info()
dfTracks.describe()


# ### Data Cleaning
# 
# After checking all three datasets if there were any null values, I thankfully wasn't able to find any. It is always a tricky step in data preprocessing to figure out how to handle the null values. There was still work to be done for the data to make more sense for my project. First, I didn't find any value with how the duration_ms was being formatted. I personally can't convert miliseconds to any meaningful time so I decided to quickly change it to seconds as a song that is 185 seconds is way easier to appreciate than one that is 227820 miliseconds long. Additionally I decided to remove mode and key as neither were too usefule in my model to create song recommendations. And I also removed the explicit column because I wasn't a fan of trying to work with a binary variable when every other song feature was measured continuously.

# In[4]:


#checking null values
print("The Null Values for the Tracks Data Set")
print(pd.isnull(dfTracks).sum())
print("The Null Values for the Genres Data Set")
print(pd.isnull(dfGenres).sum())
print("The Null Values for the Years Data Set")
print(pd.isnull(dfYear).sum())


# In[5]:


# renaming duration_ms to duration and then formatting it to seconds
dfTracks["duration"] = dfTracks["duration_ms"].apply(lambda x: round(x/1000))
dfTracks.drop(["duration_ms", "mode", "key", "explicit"], inplace = True, axis = 1)


# ## Exploratory Data Analysis:
# 
# I am reusing a model that I made for the data mining project. I was curious what features of a song made it more likely for the song the be popular and how that had changed over time. Making a linegraph was a great visualization to show exactly that. There are some factors which help explain some things over the years, acousticness has dipped way down since the 1920s where it was incredibly popular and that can probably just be explained with technology advancements with music. 
# 
# The next step was comparing the top features in songs with the top genres. I honestly tried my darndest to view this in terms of genres that are more widely popular like hip-hop or alternative rock. But I just couldn't figure it out so we are left with alberta hip hop and trap venezalano whatever that is.
# 
# Finally I feel like a machine learning project would be incomplete if it weren't to include a correlation heatmap somewhere in it to provide some quick insight on how different attributes relate to one another. This was again, more important in my project for figuring out what makes a song popular, but it is still nice to look at to just have a greater understanding of the data before we get into makeing a cluster model for the unsupervised learning part of the class.

# In[6]:


# line graph to show the song feature popularity over time.
features = ['acousticness', 'danceability', 'energy', 'instrumentalness', 'liveness', 'valence']
fig = px.line(dfYear, x='year', y=features)
fig.show()


# In[7]:


#the top 5 genres according to this dataset and then comparing the top 4 features in the previous graph with this barchart

topGenres = dfGenres.nlargest(5, 'popularity')

fig = px.bar(topGenres, x='genres', y=['danceability', 'energy', 'valence', 'acousticness'], barmode='group')
fig.show()


# In[8]:


# correlation heatmap comparing the columns in the main dataset that I was using

cm = dfTracks.corr()
plt.figure(figsize = (12,8))
sns.heatmap(cm, cmap='coolwarm', annot=True, fmt = ".2f")
plt.title('Correlation Heatmap')
plt.show()


# ## Clustering Models using K-Means
# 
# To help create a proper song recommender, it made sense to do unsupervised learning as a great way to predict what people will like, is by finding them matches with things that have similar attributes. Knowing this, making the model using clustering just felt right and that is what I decided to do with both Genres and Songs. The models were fairly efficient and I feel they passed what I was hoping for when it came to being effective too. Using the work I did for these models, I will continue working deeper for the song recommendations.

# In[9]:


#creating a cluster pipeline for the genres

cluster_pipeline = Pipeline([('scaler', StandardScaler()), ('kmeans', KMeans(n_clusters=10, n_jobs=-1))])
X = dfGenres.select_dtypes(np.number)
cluster_pipeline.fit(X)
dfGenres['cluster'] = cluster_pipeline.predict(X)

tsne_pipeline = Pipeline([('scaler', StandardScaler()), ('tsne', TSNE(n_components=2, verbose=1))])
genre_embedding = tsne_pipeline.fit_transform(X)
projection = pd.DataFrame(columns=['x', 'y'], data=genre_embedding)
projection['genres'] = dfGenres['genres']
projection['cluster'] = dfGenres['cluster']

#displaying the cluser map in an interactive way so you can see what genres are similar
fig = px.scatter(
    projection, x='x', y='y', title = 'Clustering Genres with K-Means', color='cluster', hover_data=['x', 'y', 'genres'])
fig.show()


# In[10]:


#same idea as the previous model but for songs
song_cluster_pipeline = Pipeline([('scaler', StandardScaler()), 
                                  ('kmeans', KMeans(n_clusters=20, 
                                   verbose=False, n_jobs=4))
                                 ], verbose=False)

X = dfTracks.select_dtypes(np.number)
number_cols = list(X.columns)
song_cluster_pipeline.fit(X)
song_cluster_labels = song_cluster_pipeline.predict(X)
dfTracks['cluster_label'] = song_cluster_labels

pca_pipeline = Pipeline([('scaler', StandardScaler()), ('PCA', PCA(n_components=2))])
song_embedding = pca_pipeline.fit_transform(X)
projection = pd.DataFrame(columns=['x', 'y'], data=song_embedding)
projection['title'] = dfTracks['name']
projection['cluster'] = dfTracks['cluster_label']

fig = px.scatter(
    projection, x='x', y='y', title = 'Clustering Songs with K-Means', color='cluster', hover_data=['x', 'y', 'title'])
fig.show()


# ## Getting Song Recommendations:
# 
# The next and final portion of the programming is to use the code above to have songs be recommended to the user. The first step was me just becoming familiar with working with Spotify connections. You needed to go to their developer page to create an app to aquire a client ID and a Client Secret. Once you have those two you are able to have authorization to connect with their API. Unfortunaltely for me, I was trying to do this while Spotify was down. And I spent a good amount of time debugging things from my end until I checked Reddit and saw that people were complaining that the whole site was crashed. 
# 
# Once their site was back up, I was able to start feeling comfortable making calls and confirming that I had access through the Client ID and Client Secret. I have those stored on a .env file that I will not be attaching to the github repository as I am pretty sure it is supposed to remain a secret.

# In[11]:


#I am keeping the id and secret in a file on my pc and not adding it to the repo to avoid any risk of mayhem
load_dotenv()

client_id = os.getenv("CLIENT_ID")
client_secret = os.getenv("CLIENT_SECRET")

def get_token():
    auth_string = client_id + ":" + client_secret
    auth_bytes = auth_string.encode("utf-8")
    auth_base64 = str(base64.b64encode(auth_bytes), "utf-8")
    
    url = "https://accounts.spotify.com/api/token"
    headers = {
        "Authorization": "Basic " + auth_base64,
        "Content-Type": "application/x-www-form-urlencoded"
    }
    data = {"grant_type": "client_credentials"}
    result = post(url, headers = headers, data = data)
    json_result = json.loads(result.content)
    token = json_result["access_token"]
    return token

def get_auth_header(token):
    return {"Authorization": "Bearer " + token}

def search_for_artist(token, artist_name):
    url = "https://api.spotify.com/v1/search"
    headers = get_auth_header(token)
    query = f"?q={artist_name}&type=artist&limit=1"
    
    query_url = url + query
    result = get(query_url, headers = headers)
    json_result = json.loads(result.content)["artists"]["items"]
    
    if len(json_result) == 0:
        print("No Artist")
        return None
    
    return json_result[0]

def get_songs_by_artist(token, artist_id):
    url = f"https://api.spotify.com/v1/artists/{artist_id}/top-tracks?country=US"
    headers = get_auth_header(token)
    result = get(url, headers = headers)
    json_result = json.loads(result.content)["tracks"]
    return json_result

token = get_token()
result = search_for_artist(token, "Rise Against")
artist_id = result["id"]
songs = get_songs_by_artist(token, artist_id)

for idx, song in enumerate(songs):
    print(f"{idx + 1}. {song['name']}")
    
sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id,client_secret))


# In[12]:


def find_song(name, year):
    song_data = defaultdict()
    results = sp.search(q= 'track: {} year: {}'.format(name,year), limit=1)
    if results['tracks']['items'] == []:
        return None

    results = results['tracks']['items'][0]
    track_id = results['id']
    audio_features = sp.audio_features(track_id)[0]

    song_data['name'] = [name]
    song_data['year'] = [year]
    song_data['duration'] = [results['duration']]
    song_data['popularity'] = [results['popularity']]

    for key, value in audio_features.items():
        song_data[key] = value

    return pd.DataFrame(song_data)


# In[13]:


number_cols = ['valence', 'year', 'acousticness', 'danceability', 'energy',
 'instrumentalness', 'liveness', 'loudness', 'popularity', 'speechiness', 'tempo', 'duration']


def get_song_data(song, spotify_data):
    
    try:
        song_data = spotify_data[(spotify_data['name'] == song['name']) 
                                & (spotify_data['year'] == song['year'])].iloc[0]
        return song_data
    
    except IndexError:
        return find_song(song['name'], song['year'])
        

def get_mean_vector(song_list, spotify_data):
    
    song_vectors = []
    
    for song in song_list:
        song_data = get_song_data(song, spotify_data)
        if song_data is None:
            print('Warning: {} does not exist in Spotify or in database'.format(song['name']))
            continue
        song_vector = song_data[number_cols].values
        song_vectors.append(song_vector)  
    
    song_matrix = np.array(list(song_vectors))
    return np.mean(song_matrix, axis=0)


def flatten_dict_list(dict_list):
    
    flattened_dict = defaultdict()
    for key in dict_list[0].keys():
        flattened_dict[key] = []
    
    for dictionary in dict_list:
        for key, value in dictionary.items():
            flattened_dict[key].append(value)
            
    return flattened_dict


def recommend_songs( song_list, spotify_data, n_songs=10):
    
    metadata_cols = ['name', 'year', 'artists']
    song_dict = flatten_dict_list(song_list)
    
    song_center = get_mean_vector(song_list, spotify_data)
    scaler = song_cluster_pipeline.steps[0][1]
    scaled_data = scaler.transform(spotify_data[number_cols])
    scaled_song_center = scaler.transform(song_center.reshape(1, -1))
    distances = cdist(scaled_song_center, scaled_data, 'cosine')
    index = list(np.argsort(distances)[:, :n_songs][0])
    
    rec_songs = spotify_data.iloc[index]
    rec_songs = rec_songs[~rec_songs['name'].isin(song_dict['name'])]
    return rec_songs[metadata_cols].to_dict(orient='records')


# In[14]:


#these are the songs I was most curious to see recommended
recommend_songs([{'name': 'Paper Wings', 'year':2004},
                 {'name': 'R U Mine?', 'year': 2013},
                 {'name': 'Kilby Girl', 'year': 2019},
                 {'name': 'Out of the Black', 'year': 2014},
                 {'name': 'OutRight', 'year': 2014}],  dfTracks)


# ## Results and Alalysis
# 
# I was able to spit out some songs! Though I don't know about the accuracy of the recommendation. There are also some issues with the song input. Only some songs actually are recognized. I tried getting it to work through the artist name instead of year so it would be significantly more unique, but I couldn't figure it out. The song recommendations also just feel like they are recommending the popular songs based on year instead of matching the songs based on genre. 
# 
# I don't know, I am really happy that I was able to tell a rock to do something and it actually responded with songs, but there is a lot to clean up with before I am completely satisfied. Maybe some work with the Deep Learning class can be applied here.

# ## Conclusion
# 
# In all, I am happy with the results of this project. If I am going to be honest, I still feel like I am lost on the topic of unsupervised learning, but I do feel I was able to learn a lot from the lectures, practicing with the programming assignments, and some learning on my own.
# 
# If I were to expand on this project, I would proabably like to do more work with the Spotify API. Finding a dataset on Kaggle was great, and I was able to be inspired by some other people's work. But I would really like to be more comfortable working on obtaining as much data on my own, cleaning it, and just doing the whole project without any assistance.
