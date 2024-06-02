import numpy as np
from surprise import KNNWithZScore, SVDpp, NMF, SlopeOne, KNNWithMeans, SVD, BaselineOnly, NormalPredictor, KNNBasic, KNNBaseline
from surprise import Dataset, accuracy, Reader
from surprise.model_selection import train_test_split
import pandas as pd

# load dataset
ratings = pd.read_csv('./data/BX-Book-Ratings.csv', encoding='latin-1')
# items = pd.read_csv('./data/BX-Books.csv', encoding='latin-1')
# users = pd.read_csv('./data/BX-Users.csv', encoding='latin-1')
reader = Reader(rating_scale=(1,10))
data = Dataset.load_from_df(ratings[['User-ID','ISBN','Book-Rating']],
                            reader=reader)
trainset, testset = train_test_split(data, test_size = 0.3)

# print('Number of users: ', train.n_users)
# print('Number of items: ', train.n_items)
# print('Number of rating: ', train.n_ratings)

# result variables
algorithms = [KNNWithMeans, KNNWithZScore, SVD, SVDpp, NMF, SlopeOne, BaselineOnly, NormalPredictor, KNNBasic, KNNBaseline]
names = []
results = []

# Loop 
for option in algorithms:
    algo = option()
    names.append(option.__name__)  
    algo.fit(trainset)
    predictions = algo.test(testset)
    results.append(accuracy.rmse(predictions))
names = np.array(names)
results = np.array(results)

# result plot
import matplotlib.pyplot as plt
plt.figure(figsize=(15,7))
index = np.argsort(results)
plt.plot(names[index], results[index])
plt.title('RMSE of Algorithms by Surprise')
for i, value in enumerate(results[index]):
    plt.text(i, value, f'{value:.4f}', ha='center', va='bottom')

# save plot
mode = 'surprise'
save_path = './result/'
file_name = f'{mode}_experiment.png'
plt.savefig(save_path + file_name)