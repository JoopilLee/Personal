import pandas as pd
from surprise import KNNWithMeans
from surprise import BaselineOnly
from surprise import Dataset
from surprise import accuracy
from surprise import Reader
from surprise.model_selection import cross_validate
from surprise.model_selection import train_test_split

# load dataset
ratings = pd.read_csv('./data/BX-Book-Ratings.csv', encoding='latin-1')
reader = Reader(rating_scale=(1,10))
data = Dataset.load_from_df(ratings[['User-ID','ISBN','Book-Rating']],
                            reader=reader)
trainset, testset = train_test_split(data, test_size = 0.3)

# Baseline Algorithm
algo = BaselineOnly()
result = cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=3, verbose=True)

# Save GridSearch results
result = pd.DataFrame(result)
result.to_csv('./result/baseline_cross_validation3.csv')

# Set full train data
trainset = data.build_full_trainset()
pred = algo.predict('1', '2', r_ui=3, verbose=True)  # user_id, item_id, default rating