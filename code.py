import pandas as p
import pandas as p


data1 = p.read_csv("Project139/csv file/shared_articles.csv")

data2 = p.read_csv("Project139/csv file/users_interactions.csv")


data1 = data1[data1['eventType'] == "CONTENT SHARED"]

print(data1.shape)

print(data2.shape)

def totalEvents(data1_row):
  total_likes = data2[(data2["contentId"] == data1_row["contentId"]) & (data2["eventType"] == "LIKE")].shape[0]
  total_views = data2[(data2["contentId"] == data1_row["contentId"]) & (data2["eventType"] == "VIEW")].shape[0]
  total_bookmark = data2[(data2["contentId"] == data1_row["contentId"]) & (data2["eventType"] == "BOOKMARK")].shape[0]
  total_comment = data2[(data2["contentId"] == data1_row["contentId"]) & (data2["eventType"] == "COMMENT CREATED")].shape[0]
  total_follow = data2[(data2["contentId"] == data1_row["contentId"]) & (data2["eventType"] == "FOLLOW")].shape[0]

  return total_likes + total_views + total_bookmark + total_comment + total_follow


data1['total_events']= data1.apply(totalEvents, axis=1)

data1 = data1.sort_values(["total_events"], ascending = False)

print(data1.head())

# --------------------------------------------------------------------------------------------------------------------------------

def convert_lowercase(x):
  if isinstance(x, str):
      return x.lower()
  else:
      return ''

data1["title"] = data1["title"].apply(convert_lowercase)

data1.head()


from sklearn.feature_extraction.text import CountVectorizer
count = CountVectorizer(stop_words='english')
count_matrix = count.fit_transform(data1['title'])


from sklearn.metrics.pairwise import cosine_similarity
cosine_sim2 = cosine_similarity(count_matrix, count_matrix)


df1 = data1.reset_index()
indices = p.Series(df1.index, index=data1['contentId'])


def get_recommendations(contentId, cosine_sim):
    idx = indices[contentId]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    movie_indices = [i[0] for i in sim_scores]
    return df1['contentId'].iloc[movie_indices]


get_recommendations(-4029704725707465084, cosine_sim2)



