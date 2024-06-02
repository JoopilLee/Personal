from surprise import SVD, KNNWithZScore
from surprise import Dataset, Reader
from surprise.model_selection import GridSearchCV, train_test_split
import pandas as pd
import json

# load dataset
ratings = pd.read_csv('./data/BX-Book-Ratings.csv', encoding='latin-1')
reader = Reader(rating_scale=(1,10)) # ratings range 1~10
data = Dataset.load_from_df(ratings[['User-ID','ISBN','Book-Rating']],
                            reader=reader)
# trainset, testset = train_test_split(data, test_size = 0.3)

# compare SVD parameteres
param_grid = {'n_epochs': [10, 20, 30],
              'lr_all': [0.002, 0.005, 0.01],
              'reg_all': [0.01 ,0.05, 0.07, 0.1, 0.4],
              'n_factors': [50,100,200,300]}
# K는 작을수록 good lr = 0.005 reg = 0.05 ~ 0.1
gs = GridSearchCV(SVD, param_grid, measures=['rmse'], cv=3)
gs.fit(data)

result_dict = {
    'best_rmse': gs.best_score['rmse'],
    'best_paramse': gs.best_params['rmse']    
}
with open('./result/SVD_gridsearch_result.json','w') as json_file:
    json.dump(result_dict, json_file)

# best RMSE
print(gs.best_score['rmse'])

# best parameter
print(gs.best_params['rmse'])