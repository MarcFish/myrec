import tensorflow as tf
from tqdm import tqdm


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _int64_list_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _float_list_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _bytes_feature(value):  # TODO
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def serialize_ratings(u, m, rating, time,gender,age,occ,zip_code,title,year,cats):
    feature = {
        'user':_int64_feature(u),
        'movie':_int64_feature(m),
        'rating':_float_feature(rating),
        'time':_int64_feature(time),
        'gender':_int64_feature(gender),
        'age':_float_feature(age),
        'occupation':_int64_feature(occ),
        'zip_code':_int64_feature(zip_code),
        'title':_bytes_feature(title.encode('utf-8')),
        'year':_int64_feature(year),
        'cats':_int64_list_feature(cats),
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


zip_dict = dict()
age_set = set()
occ_set = set()
with open("users.dat",encoding='utf-8') as f:
    row_list = f.readlines()
    for row in row_list:
        row = row.strip().split("::")
        zip_code = row[4]
        age = int(row[2])
        age_set.add(age)
        occ = int(row[3])
        occ_set.add(occ)
        zip_dict.setdefault(zip_code, len(zip_dict))
print("zip_dict len:{}".format(len(zip_dict)))
print("occ len:{}".format(len(occ_set)))
max_age = max(age_set)
min_age = min(age_set)
user_dict = dict()
with open("users.dat",encoding='utf-8') as f:
    row_list = f.readlines()
    for row in tqdm(row_list):
        row=row.strip().split("::")
        user = int(row[0])
        if row[1] == 'F':
            gender = 0
        else:
            gender = 1
        age = (int(row[2])-min_age)/(max_age-min_age)
        occ = int(row[3])
        zip_code = zip_dict[row[4]]
        user_dict[user] = [gender, age, occ, zip_code]

movie_list = list()
cat_list = list()
with open("movies.dat",encoding='unicode_escape') as f:
    row_list = f.readlines()
    for row in tqdm(row_list):
        row=row.strip().split("::")
        movie = int(row[0])
        title = row[1]
        temp = title.split()
        year = int(temp[-1][1:-1])
        title = ' '.join(temp[:-1])
        cats = row[2].strip().split("|")
        cat_list.extend(cats)
        movie_list.append([movie,title,year,cats])
cat_list = list(set(cat_list))
print("cat len:{}".format(len(cat_list)))
cat_dict = dict()
for cat in cat_list:
    cat_dict.setdefault(cat, len(cat_dict))

movie_dict = dict()
for movie in tqdm(movie_list):
    m = movie[0]
    title = movie[1]
    year = movie[2]
    cats = [cat_dict.get(x) for x in movie[3]]
    movie_dict[m] = [title,year,cats]

rating_list = list()
with open("ratings.dat",encoding='utf-8') as f:
    row_list = f.readlines()
    for row in tqdm(row_list):
        row = row.strip().split("::")
        user = int(row[0])
        movie = int(row[1])
        rating = float(row[2])/5.0
        time = int(row[3])
        rating_list.append([user,movie,rating,time])

sorted(rating_list,key=lambda x:x[3])
slice_ = len(rating_list)//10*7
with tf.io.TFRecordWriter('train.tfrecord') as writer:
    for user,movie,rating,time in tqdm(rating_list[:slice_]):
        gender,age,occ,zip_code = user_dict[user]
        title,year,cats = movie_dict[movie]
        example = serialize_ratings(user, movie, rating, time,gender,age,occ,zip_code,title,year,cats)
        writer.write(example)

with tf.io.TFRecordWriter('test.tfrecord') as writer:
    for user,movie,rating,time in tqdm(rating_list[slice_:]):
        gender,age,occ,zip_code = user_dict[user]
        title,year,cats = movie_dict[movie]
        example = serialize_ratings(user, movie, rating, time,gender,age,occ,zip_code,title,year,cats)
        writer.write(example)
