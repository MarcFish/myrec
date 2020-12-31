import pandas as pd
from tqdm import tqdm
from collections import namedtuple, defaultdict
from sklearn.preprocessing import LabelEncoder


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
    movie_dict[movie].extend([tag for tag, _ in tags[-2:]])
