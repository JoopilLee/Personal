# Predictions using MF ###########################################################################
import pandas as pd
import numpy as np
import argparse
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from sklearn.utils import shuffle


# mf class
class NEW_MF():
    # user - book matrix 행렬을 받음
    def __init__(self, ratings, K, alpha, beta, iterations, tolerance=0.005, verbose=True):
        self.R = np.array(ratings)
        # book id와 index 리스트 선언
        item_id_index = []
        # index와 book id 리스트 선언
        index_item_id = []

        for i, one_id in enumerate(ratings):
            item_id_index.append([one_id, i])
            index_item_id.append([i, one_id])

        # 딕셔너리화
        self.item_id_index = dict(item_id_index)
        self.index_item_id = dict(index_item_id)

        # user와 index 리스트 선언
        user_id_index = []
        # index와 user와 리스트 선언
        index_user_id = []

        # book user matrix 행렬을 받음
        for i, one_id in enumerate(ratings.T):
            user_id_index.append([one_id, i])
            index_user_id.append([i, one_id])

        # 딕셔너리화
        self.user_id_index = dict(user_id_index)
        self.index_user_id = dict(index_user_id)

        # 파라미터 선언
        self.num_users, self.num_items = np.shape(self.R)
        self.K = K
        self.alpha = alpha
        self.beta = beta
        self.iterations = iterations
        self.tolerance = tolerance
        self.verbose = verbose
        # print(len(self.user_id_index))

    # 테스트 데이터 세팅
    def set_test(self, ratings_test):                           # Setting test set
        test_set = []
        for i in range(len(ratings_test)):                      # Selected ratings
            # print(self.user_id_index)
            # print(self.user_id_index[4738])

            # 인덱스 확인해서 유저의 인덱스에 해당하는 것
            x = self.user_id_index[ratings_test.iloc[i,0]]

            # 인덱스 확인해서 book의 인덱스에 해당하는 것
            y = self.item_id_index[ratings_test.iloc[i,1]]

            # 점수
            z = ratings_test.iloc[i,2]

            # 테스트 데이터
            test_set.append([x, y, z])

            # 테스트 데이터에 해당되는 원본데이터 점수 0점으로 만듬
            self.R[x, y] = 0
        self.test_set = test_set
        return test_set

    def test(self):                             # Training 하면서 test set의 정확도를 계산하는 메소드
        # P,Q 선언
        self.P = np.random.normal(scale=1./self.K, size=(self.num_users, self.K))
        self.Q = np.random.normal(scale=1./self.K, size=(self.num_items, self.K))

        # bias 선언
        self.b_u = np.zeros(self.num_users)
        self.b_d = np.zeros(self.num_items)
        self.b = np.mean(self.R[self.R.nonzero()])

        # 비어있는값 삭제 -> bookratings는 이미 비어있는 값이 없으므로 사실상 필요 x
        rows, columns = self.R.nonzero()

        # 샘플값
        self.samples = [(i,j, self.R[i,j]) for i, j in zip(rows, columns)]


        # for문을 통해서 비교 후 rmse가 작은 값을 찾기 위해 큰값을 best RMSE에 선언
        best_RMSE = 10000

        # iteration확인하기 위한 선언
        best_iteration = 0
        training_process = []
        for i in range(self.iterations):
            np.random.shuffle(self.samples)#샘플 셔플
            self.sgd()# sgd
            rmse1 = self.rmse()
            rmse2 = self.test_rmse() # test rmse
            training_process.append((i, rmse1, rmse2)) # 진행과정 확인위한 리스트

            # 출력되는부분
            if self.verbose:# default값이 True설정 False하면 안보임
                if (i+1) % 10 == 0:
                    print("Iteration: %d ; Train RMSE = %.6f ; Test RMSE = %.6f" % (i+1, rmse1, rmse2))

            # rmse값이 더 좋으면 bestrmse에 반영
            if best_RMSE > rmse2:                      # New best record
                best_RMSE = rmse2
                best_iteration = i
            elif (rmse2 - best_RMSE) > self.tolerance: # 점수가 더 나아지지 않으면 break
                break
        print(best_iteration, best_RMSE) # 출력하는부분
        return training_process,best_RMSE # 평균값을 받기 위해 리턴값 하나 더 추가

    # sgd 함수
    def sgd(self):
        for i, j, r in self.samples: #샘플을 받아서
            prediction = self.get_prediction(i, j) # 예측
            error = (r - prediction) #loss 평가
            self.b_u[i] += self.alpha * (error - self.beta * self.b_u[i]) # 파라미터값과 계산하여 가중치 변화
            self.b_d[j] += self.alpha * (error - self.beta * self.b_d[j]) # ==

            self.Q[j, :] += self.alpha * (error * self.P[i, :] - self.beta * self.Q[j,:]) # ==
            self.P[i, :] += self.alpha * (error * self.Q[j, :] - self.beta * self.P[i,:]) # ==

    # rmse 측정 학습 rmse를 평가하는 함수
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

    # predict하는 함수
    def get_prediction(self, i, j):
        prediction = self.b + self.b_u[i] + self.b_d[j] + self.P[i, :].dot(self.Q[j, :].T)
        return prediction

    # predict하는 함수
    def get_one_prediction(self, user_id, isbn):
        return self.get_prediction(self.user_id_index[user_id], self.item_id_index[isbn])

# 데이터 전처리 및 데이터 준비
def dataprepare(args):
    path=args.path
    ratings = pd.read_csv(path)
    ratings['Book-Rating'] = ratings['Book-Rating'].astype(int)
    ratings.columns=['user_id','isbn','rating']

    # 0점은 다 제거
    ratings=ratings[ratings['rating']!=0]
    ratings=ratings.reset_index(drop=True)

    #  user과 isbn의 인덱스를 정해주기 위해 LabelEncoder 사용!
    user_encoder = LabelEncoder()
    isbn_encoder = LabelEncoder()

    # user_id와 isbn label인코딩
    ratings['user_id'] = user_encoder.fit_transform(ratings['user_id'])
    ratings['isbn'] = isbn_encoder.fit_transform(ratings['isbn'])
    return ratings

# 학습 및 평가하는 함수
def main(args, ratings):
    # kfold이용해서 트레인 테스트 3개로 분리
    kf = KFold(n_splits=3, shuffle=True)

    # rmse리스트
    average_rmse_values = []

    # train index test index 무작위로 1/3 나눔
    for train_index, test_index in kf.split(ratings):

        # print(train_index)
        # print(test_index)

        # Train + test index의 리스트를 각각받아 원본데이터를 나눔
        train_data = ratings.iloc[train_index]
        test_data = ratings.iloc[test_index]
        # print(test_data)

        # 피봇으로 만듬 fillna는 사실상 안쓰임 + user_id isbn 행렬을 만듬
        temp = ratings.pivot(index = 'user_id', columns ='isbn', values = 'rating').fillna(0)
        # mf = NEW_MF(temp, K=220, alpha=0.0014, beta=0.075, iterations=350, tolerance=0.0001, verbose=True)

        # 인스턴스생성 + 파라미터를 대입
        mf = NEW_MF(temp, K=args.K,alpha=args.alpha, beta=args.beta, iterations=args.iter, tolerance=args.tol, verbose=True)

        # Test 데이터 세팅
        test_set = mf.set_test(test_data)

        # 추가 리턴을 받아서 rmse또한 반환
        result,rmse = mf.test()

        # 평균을 계산하기 위한 리스트
        average_rmse_values.append(rmse)

    # 전체 폴드에 대한 RMSE 평균 계산 및 출력
    average_rmse = np.mean(average_rmse_values)

    #평균 RMSE
    print("평균 RMSE:", average_rmse)

if __name__=='__main__':
    # argparse 라이브러리를 통해서 파라미터 설정
    parser = argparse.ArgumentParser()
    parser.add_argument('--path',type=str,default='/Users/kimkirok/Documents/대학원/DA/2학기/추천/팀 과제/BX-Book-Ratings.csv')
    parser.add_argument('--K',type=int,default=30)
    parser.add_argument('--alpha',type=float,default=0.01)
    parser.add_argument('--beta',type=float,default=0.02)
    parser.add_argument('--iter',type=int,default=100)
    parser.add_argument('--tol',type=float,default=0.01)
    arg = parser.parse_args()
    main(arg,dataprepare(arg))
