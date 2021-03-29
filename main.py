import pandas as pd
import numpy as np
from numba import jit, cuda, float32
import math
import time
import sys


FEATURES = 16
n_users = 1000
n_movies = 5000

TPB = 16


def get_data():
    df = pd.read_csv('./dataset/ratings.csv')

    print(df.shape)
    df.drop(columns=['timestamp'], inplace=True)
    df = df.loc[(df['userId'] < n_users) & (df['movieId'] < n_movies)]
    print(df.shape)

    df.to_csv('./dataset/data.csv')

    # Returns reviews user ids and movie ids
    return df.to_numpy(), df['userId'].unique(), df['movieId'].unique()


@cuda.jit
def update_weights(A, B, x, y, err, alpha, beta):
    i = cuda.threadIdx.x
    tmp = A[x][i]
    A[x][i] += 2. * alpha * (err * B[i][y] - beta * A[x][i])
    B[i][y] += 2. * alpha * (err * tmp - beta * B[i][y])


@cuda.jit
def get_error(users, movies, review_act, mov_map):
    # Define an array in the shared memory
    sA = cuda.shared.array(shape=(TPB, TPB), dtype=float32)
    sB = cuda.shared.array(shape=(TPB, TPB), dtype=float32)

    x, y = cuda.grid(2)

    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    bpg = cuda.gridDim.x

    score = -1
    idx = -1
    for i, row in enumerate(review_act):
        if row[0] == y and row[1] == mov_map[x]:
            score = row[2]
            idx = i
            break

    # Each thread computes one element in the result matrix.
    tmp = 0.0
    for i in range(bpg):
        # Preload data into shared memory
        sA[ty, tx] = 0.0
        sB[ty, tx] = 0.0
        if y < users.shape[0] and (tx + i * TPB) < users.shape[1]:
          sA[ty, tx] = users[y, tx + i * TPB]
        if x < movies.shape[1] and (ty + i * TPB) < movies.shape[0]:
          sB[ty, tx] = movies[ty + i * TPB, x]

        # Wait until all threads finish preloading
        cuda.syncthreads()

        # Computes partial product on the shared memory
        for j in range(TPB):
            tmp += sA[ty, j] * sB[j, tx]

        # Wait until all threads finish computing
        cuda.syncthreads()

    if score == -1 or x >= movies.shape[1] or y >= users.shape[0]:
        return

    err = score - tmp
    review_act[idx][3] = tmp
    review_act[idx][4] = err


if __name__ == '__main__':
    """
    Dependencies: numpy, pandas, numba
    
    Dataset required: https://www.kaggle.com/rounakbanik/the-movies-dataset?select=ratings.csv
    save to address: './dataset/ratings.csv'
    
    Run for training: python main.py train
    
    Run for test: python main.py test
    """
    ratings, usr, mov = get_data()

    if sys.argv[1] == 'test':
        x = int(input('User: '))
        y = int(input('Movie: '))

        # Modify latest epoch completed
        users = np.loadtxt('./dataset/user_49.csv', delimiter=',')
        movies = np.loadtxt('./dataset/movies_49.csv', delimiter=',')

        y = np.where(mov == y)[0][0]
        score = np.matmul(users[x, :], movies[:, y])
        print('Predicted score: {}'.format(score))

    else:
        err_col = np.zeros((ratings.shape[0], 2), dtype=float)
        ratings = np.append(ratings, err_col, axis=1)
        # print(ratings)

        ratings_gmem = cuda.to_device(ratings)
        movie_map_gmem = cuda.to_device(mov)

        users = np.random.uniform(0.0, 1.0, size=(n_users, FEATURES))
        movies = np.random.uniform(0.0, 1.0, size=(FEATURES, n_movies))

        print(users.shape)
        print(movies.shape)

        threadsperblock = (TPB, TPB)
        grid_y_max = max(users.shape[0], movies.shape[0])
        grid_x_max = max(users.shape[1], movies.shape[1])
        blockspergrid_x = math.ceil(grid_x_max / threadsperblock[0])
        blockspergrid_y = math.ceil(grid_y_max / threadsperblock[1])
        blockspergrid = (blockspergrid_x, blockspergrid_y)

        print('Threads per block: {} \nBlocks per grid:{}'.format(threadsperblock, blockspergrid))

        epochs = 50
        alpha = 0.0002
        beta = 0.01
        for i in range(epochs):
            start_time = time.time()

            users_gmem = cuda.to_device(users)
            movies_gmem = cuda.to_device(movies)

            get_error[blockspergrid, threadsperblock](users_gmem, movies_gmem, ratings_gmem, movie_map_gmem)

            ratings = ratings_gmem.copy_to_host()

            rmse = math.sqrt(np.sum(np.square(ratings[:, 4])) / ratings.shape[0])

            stream = cuda.stream()
            with stream.auto_synchronize():
                for j, row in enumerate(ratings):
                    x = row[0]
                    y = np.where(mov==row[1])[0][0]

                    if row[4] != 0.0:
                        update_weights[(1), (FEATURES), stream](users_gmem, movies_gmem, int(x), int(y), row[4], alpha, beta)

            users = users_gmem.copy_to_host()
            movies = movies_gmem.copy_to_host()

            end_time = time.time()
            print("Epoch: {} Execution time = {} RMSE: {}".format(i + 1, (end_time - start_time), rmse))
            np.savetxt('./dataset/users_' + str(i) + '.csv', users, delimiter=',')
            np.savetxt('./dataset/movies_' + str(i) + '.csv', movies, delimiter=',')

        print('\nAfter:\n')
        print(users[0:5][:])
        print(movies[:][0:5])

        df = pd.DataFrame(ratings, columns=['userId', 'movieId', 'Rating', 'Pred', 'Err_Sqr'])
        df.to_csv('./dataset/result.csv')
