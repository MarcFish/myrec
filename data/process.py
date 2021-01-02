import pandas as pd
from tqdm import tqdm
from collections import namedtuple, defaultdict
from sklearn.preprocessing import LabelEncoder
from myrec.utils import write_csv
from sklearn.model_selection import train_test_split
import numpy as np


ratings = pd.read_csv("./ratings.csv")
movies = pd.read_csv("./movies.csv")
tags = pd.read_csv("./genome-scores.csv")

movie_dict = defaultdict(list)
cat_set = set()
trans = LabelEncoder()
movie_tag = defaultdict(list)


for row in tqdm(movies.itertuples()):
    id = row[1]
    cats = row[3].split("|")
    for cat in cats:
        cat_set.add(cat)
trans.fit(list(cat_set))
for row in tqdm(movies.itertuples()):
    id = row[1]
    cats = row[3].split("|")
    if len(cats) < 2:
        cats = [cats[0], cats[0]]
    else:
        cats = cats[:2]
    cats = trans.transform(cats)
    movie_dict[id].extend(cats)

for row in tqdm(tags.itertuples()):
    movie_tag[row[1]].append((row[2], row[3]))

for movie, tags in tqdm(movie_tag.items()):
    tags.sort(key=lambda x: x[1])

for movie, tags in tqdm(movie_tag.items()):
    l = [tag for tag, _ in tags[-2:]]
    if len(l) < 2:
        l = l.append(l[0])
    movie_dict[movie].extend(l)

content_list = [["user_id", "movie_id", "cat1", "cat2", "tag1", "tag2", "rating", "time"]]
for row in tqdm(ratings.itertuples()):
    user_id = row[1]
    movie_id = row[2]
    rating = row[3] / 5.0
    time = row[4]
    if len(movie_dict[movie_id]) < 4:
        continue
    l = [user_id, movie_id, *movie_dict[movie_id], rating, time]
    content_list.append(l)

content_list[1:].sort(key=lambda x: x[-1])
for row in tqdm(content_list):
    row.pop()
train_list, test_list = train_test_split(content_list[1:])

train_list.insert(0, content_list[0])
test_list.insert(0, content_list[0])
write_csv("train.csv", train_list)
write_csv("test.csv", test_list)
