import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import keras
from keras.layers import Activation
from keras.layers import Conv2D, BatchNormalization, Dense, Flatten, Reshape


def get_data(file):
    data = pd.read_csv(file)

    puzzles = data['quizzes']
    solutions = data['solutions']

    puzzle_inputs = []
    labels = []

    for i in puzzles:
        x = np.array([int(j) for j in i]).reshape((9, 9, 1))
        puzzle_inputs.append(x)

    puzzle_inputs = np.asarray(puzzle_inputs)
    puzzle_inputs = puzzle_inputs / 9
    puzzle_inputs -= .5

    for i in solutions:
        x = np.asarray([int(j) for j in i]).reshape((81, 1)) - 1
        labels.append(x)

    labels = np.asarray(labels)

    del puzzles
    del solutions

    x_train, x_test, y_train, y_test = train_test_split(puzzle_inputs, labels, test_size=0.2, random_state=42)

    return x_train, x_test, y_train, y_test


def norm(a): return (a / 9) - .5

def denorm(a): return (a + .5) * 9

def train_model(x_train, y_train, batch_size=64, epochs=2):
    model = keras.models.Sequential()

    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same', input_shape=(9, 9, 1)))
    model.add(BatchNormalization())
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, kernel_size=(1, 1), activation='relu', padding='same'))

    model.add(Flatten())
    model.add(Dense(81 * 9))
    model.add(Reshape((-1, 9)))
    model.add(Activation('softmax'))

    adam = keras.optimizers.adam(lr=.001)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=adam)
    model.summary()

    print(model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs))

    model.save('sudoku.model')


def inference_sudoku(sample):
    '''
        This function solve the sudoku by filling blank positions one by one.
    '''

    feat = sample

    while 1:
        out = model.predict(feat.reshape((1, 9, 9, 1)))
        out = out.squeeze()

        pred = np.argmax(out, axis=1).reshape((9, 9)) + 1
        prob = np.around(np.max(out, axis=1).reshape((9, 9)), 2)

        feat = denorm(feat).reshape((9, 9))
        mask = (feat == 0)

        if mask.sum() == 0:
            break

        prob_new = prob * mask

        ind = np.argmax(prob_new)
        x, y = (ind // 9), (ind % 9)

        val = pred[x][y]
        feat[x][y] = val
        feat = norm(feat)

    return pred


def solve_sudoku(game):
    game = game.replace('\n', '')
    game = game.replace(' ', '')
    game = np.array([int(j) for j in game]).reshape((9, 9, 1))
    game = norm(game)
    game = inference_sudoku(game)
    return game


def evaluate_test_accuracy(puzzles, labels):
    correct = 0
    for i, puzzle in enumerate(puzzles):
        prediction = inference_sudoku(puzzle)
        label = labels[i].reshape((9, 9)) + 1
        if abs(label - prediction).sum() == 0:
            correct += 1
    print("test accuracy: {}/{}={}".format(correct, puzzles.shape[0], correct / puzzles.shape[0]))


if __name__ == '__main__':
    x_train, x_test, y_train, y_test = get_data('sudoku.csv')

    train_model(x_train, y_train)
    model = keras.models.load_model('sudoku.model')

    evaluate_test_accuracy(x_test[:100], y_test[:100])

    game = '''
              0 8 0 0 3 2 0 0 1
              7 0 3 0 8 0 0 0 2
              5 0 0 0 0 7 0 3 0
              0 5 0 0 0 1 9 7 0
              6 0 0 7 0 9 0 0 8
              0 4 7 2 0 0 0 5 0
              0 2 0 6 0 0 0 0 9
              8 0 0 0 9 0 3 0 5
              3 0 0 8 2 0 0 1 0
          '''
    print("raw puzzle:", game, sep="\n")
    game = solve_sudoku(game)
    print('solved puzzle:', game, sep="\n")
    print(np.sum(game, axis=1))
