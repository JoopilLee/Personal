import numpy as np
import pandas as pd
import re
from sklearn.utils import shuffle

def extract_numeric(strings):   
    return re.sub(r'\D', '', strings)

# load dataset
r_cols = ['user_id', 'movie_id', 'rating']
ratings = pd.read_csv('./data/BX-Book-Ratings.csv', encoding='latin-1')
ratings.columns = r_cols
ratings['movie_id'] = ratings['movie_id'].apply(extract_numeric)
ratings = ratings[['user_id', 'movie_id', 'rating']].astype(int)
print(ratings.info())

items = pd.read_csv('./data/BX-Books.csv', encoding='latin-1')
users = pd.read_csv('./data/BX-Users.csv', encoding='latin-1') 

# train test 분리
TRAIN_SIZE = 0.7
ratings = shuffle(ratings, random_state=12)
cutoff = int(TRAIN_SIZE * len(ratings))
ratings_train = ratings.iloc[:cutoff]
ratings_test = ratings.iloc[cutoff:]

# New MF class for training & testing
class NEW_MF():
    # Initializing the object
    def __init__(self, ratings, K, alpha, beta, iterations, tolerance=0.005, verbose=True):
        self.R = np.array(ratings)
        # user_id, movie_id를 R의 index와 매칭하기 위한 dictionary 생성
        item_id_index = []
        index_item_id = []
        for i, one_id in enumerate(ratings):
            item_id_index.append([one_id, i])
            index_item_id.append([i, one_id])
        self.item_id_index = dict(item_id_index)
        self.index_item_id = dict(index_item_id)        
        user_id_index = []
        index_user_id = []
        for i, one_id in enumerate(ratings.T):
            user_id_index.append([one_id, i])
            index_user_id.append([i, one_id])
        self.user_id_index = dict(user_id_index)
        self.index_user_id = dict(index_user_id)
        # 다른 변수 초기화
        self.num_users, self.num_items = np.shape(self.R)
        self.K = K
        self.alpha = alpha
        self.beta = beta
        self.iterations = iterations
        self.tolerance = tolerance
        self.verbose = verbose

    # 테스트 셋을 선정하는 메소드 
    def set_test(self, ratings_test):                           # Setting test set
        test_set = []
        for i in range(len(ratings_test)):                      # Selected ratings
            x = self.user_id_index[ratings_test.iloc[i,0]]      # Getting R indice for the given user_id and movie_id
            y = self.item_id_index[ratings_test.iloc[i,1]]
            z = ratings_test.iloc[i,2]
            test_set.append([x, y, z])
            self.R[x, y] = 0                    # Setting test set ratings to 0
        self.test_set = test_set
        return test_set                         # Return test set

    def test(self):                             # Training 하면서 test set의 정확도를 계산하는 메소드 
        # Initializing user-feature and movie-feature matrix
        self.P = np.random.normal(scale=1./self.K, size=(self.num_users, self.K))
        self.Q = np.random.normal(scale=1./self.K, size=(self.num_items, self.K))

        # Initializing the bias terms
        self.b_u = np.zeros(self.num_users)
        self.b_d = np.zeros(self.num_items)
        self.b = np.mean(self.R[self.R.nonzero()])

        # List of training samples
        rows, columns = self.R.nonzero()
        self.samples = [(i,j, self.R[i,j]) for i, j in zip(rows, columns)]

        # Stochastic gradient descent for given number of iterations
        best_RMSE = 10000
        best_iteration = 0
        training_process = []
        for i in range(self.iterations):
            np.random.shuffle(self.samples)
            self.sgd()
            rmse1 = self.rmse()
            rmse2 = self.test_rmse()
            training_process.append((i, rmse1, rmse2))
            if self.verbose:
                if (i+1) % 10 == 0:
                    print("Iteration: %d ; Train RMSE = %.6f ; Test RMSE = %.6f" % (i+1, rmse1, rmse2))
            if best_RMSE > rmse2:                      # New best record
                best_RMSE = rmse2
                best_iteration = i
            elif (rmse2 - best_RMSE) > self.tolerance: # RMSE is increasing over tolerance
                break
        print(best_iteration, best_RMSE)
        return training_process

    # Stochastic gradient descent to get optimized P and Q matrix
    def sgd(self):
        for i, j, r in self.samples:
            prediction = self.get_prediction(i, j)
            error = (r - prediction)
            self.b_u[i] += self.alpha * (error - self.beta * self.b_u[i])
            self.b_d[j] += self.alpha * (error - self.beta * self.b_d[j])

            self.Q[j, :] += self.alpha * (error * self.P[i, :] - self.beta * self.Q[j,:])
            self.P[i, :] += self.alpha * (error * self.Q[j, :] - self.beta * self.P[i,:])

    # Computing mean squared error
    def rmse(self):
        rows, columns = self.R.nonzero()
        self.predictions = []
        self.errors = []
        for x, y in zip(rows, columns):
            prediction = self.get_prediction(x, y)
            self.predictions.append(prediction)
            self.errors.append(self.R[x, y] - prediction)
        self.predictions = np.array(self.predictions)
        self.errors = np.array(self.errors)
        return np.sqrt(np.mean(self.errors**2))

    # Test RMSE 계산하는 method 
    def test_rmse(self):
        error = 0
        for one_set in self.test_set:
            predicted = self.get_prediction(one_set[0], one_set[1])
            error += pow(one_set[2] - predicted, 2)
        return np.sqrt(error/len(self.test_set))

    # Ratings for user i and moive j
    def get_prediction(self, i, j):
        prediction = self.b + self.b_u[i] + self.b_d[j] + self.P[i, :].dot(self.Q[j, :].T)
        return prediction

    # Ratings for user_id and moive_id
    def get_one_prediction(self, user_id, movie_id):
        return self.get_prediction(self.user_id_index[user_id], self.item_id_index[movie_id])

# Testing MF RMSE ,ISBN,Book-Rating
R_temp = ratings.pivot(index='user_id', columns='movie_id', values='rating').fillna(0)
mf = NEW_MF(R_temp, K=220, alpha=0.0014, beta=0.075, iterations=350, tolerance=0.0001, verbose=True)
test_set = mf.set_test(ratings_test)
result = mf.test()
print(mf.get_one_prediction(1,2),R_temp.loc[1][2])


###################### 추천하기 ######################

# import pandas as pd
# # 추천을 위한 데이터 읽기 (추천을 위해서는 전체 데이터를 읽어야 함)
# r_cols = ['user_id', 'movie_id', 'rating', 'timestamp']
# ratings = pd.read_csv('../data/u.data', names=r_cols,  sep='\t',encoding='latin-1')
# ratings = ratings.drop('timestamp', axis=1)
# rating_matrix = ratings.pivot(values='rating', index='user_id', columns='movie_id')

# # 영화 제목 가져오기
# i_cols = ['movie_id', 'title', 'release date', 'video release date', 'IMDB URL', 
#           'unknown', 'Action', 'Adventure', 'Animation', 'Children\'s', 'Comedy', 
#           'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 
#           'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
# movies = pd.read_csv('../data/u.item', sep='|', names=i_cols, encoding='latin-1')
# movies = movies[['movie_id', 'title']]
# movies = movies.set_index('movie_id')

# # 추천하기
# def recommender(user, n_items=10):
#     # 현재 사용자의 모든 아이템에 대한 예상 평점 계산
#     predictions = []
#     rated_index = rating_matrix.loc[user][rating_matrix.loc[user] > 0].index    # 이미 평가한 영화 확인
#     items = rating_matrix.loc[user].drop(rated_index)
#     for item in items.index:
#         predictions.append(mf.get_one_prediction(user, item))                   # 예상평점 계산
#     recommendations = pd.Series(data=predictions, index=items.index, dtype=float)
#     recommendations = recommendations.sort_values(ascending=False)[:n_items]    # 예상평점이 가장 높은 영화 선택
#     recommended_items = movies.loc[recommendations.index]['title']
#     return recommended_items

# # 영화 추천 함수 부르기
# recommender(2, 10)
