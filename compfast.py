import sklearn
from sklearn import model_selection, linear_model, svm, ensemble
import numpy as np
np.set_printoptions(suppress=True)

def yes_no_binary(value):
    return 1 if value == 'yes' else 0 #converters={6: yes_no_binary, 7: yes_no_binary, 8: yes_no_binary}


test = np.array([[1,2,3],[4,5,6],[7,8,9]])

print(test[1,:2])

data = np.genfromtxt('Computers.csv', delimiter=',', dtype=int,
                    converters={6: yes_no_binary, 7: yes_no_binary, 8: yes_no_binary})[1:,1:]
print(data[0])

# print(test2)

print(data.shape)

y = data[:, 0]
X = data[:,1:len(data)]


X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y)

reg = linear_model.LinearRegression()
reg.fit (X_train, y_train)

predict = reg.predict(X_test)
actual = y_test

# print('reg.coef_', reg.coef_)`
print('predict', predict)
print('actual', actual)

total_correct = 0
total_wrong = 0
for guess_num in range(len(predict)):
    print('--\n prediction:', np.round_(predict[guess_num]), '\n actual:', np.round_(actual[guess_num]), '\n diff: ', np.round_(predict[guess_num] - actual[guess_num]), '\n')

print('accuracy_rate:', total_correct/(total_correct+total_wrong))