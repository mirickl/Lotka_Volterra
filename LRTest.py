import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

from LinearRegression import LinearRegression

import matplotlib as mpl
#add capacity
mpl.rcParams['agg.path.chunksize'] = 10000

data = pd.read_csv(r'res_data_log.csv')
# print(data.info())
#得到训练数据
train_data = data.sample(frac = 0.8)
test_data = data.drop(train_data.index)
# print(train_data.shape)
#
input_param_name = 'time'
output_param_name1 = 'prey'
output_param_name2 = 'predator'
#
x_train = train_data[[input_param_name]].values
prey_train = train_data[[output_param_name1]].values
predator_train = train_data[[output_param_name2]].values
#
x_test = test_data[input_param_name].values
prey_test = test_data[output_param_name1].values
predator_test = test_data[output_param_name2].values

#Graphic drawing
#训练数据画图先注释
# plt.plot(x_train, prey_train, label='Predator', color='r', linestyle='dashed')
# # plt.scatter(x_train,prey_train,label = 'Train data')
# # plt.scatter(x_test,pre_test,label = 'test data')
# plt.xlabel(input_param_name)
# plt.ylabel(output_param_name1)
# plt.title('prey_num_staring')
# plt.legend()
# plt.show()

#
#
num_iterations = 500
learning_rate = 0.01
#
linear_regression = LinearRegression(x_train,prey_train)
theta, cost_history=linear_regression.train(learning_rate,num_iterations)
#
print('the loss before trianing：',cost_history[0])
print('the loss after training:',cost_history[-1])

plt.plot(range(num_iterations),cost_history)
plt.xlabel('Iter')
plt.ylabel('cost')
plt.title('GD')
plt.show()
#
predictions_num = len(x_train)

x_predictions = np.linspace(x_train.min(),x_train.max(),predictions_num).reshape(predictions_num,1)
y_predictions = linear_regression.predict(x_predictions)

# plt.scatter(x_train,prey_train,label = 'Train data')
# plt.scatter(x_test,prey_test,label = 'test data')
plt.plot(x_predictions, y_predictions, label='Predator', color='r', linestyle='dashed')
plt.xlabel(input_param_name)
plt.ylabel(output_param_name1)
plt.title('prediction')
plt.legend()
plt.show()

#MSE
predictions_num = len(x_test)
x_test_pre = np.linspace(x_test.min(),x_test.max(),predictions_num).reshape(predictions_num,1)
prey_pre = linear_regression.predict(x_test_pre)
LR_MSE = mean_squared_error(prey_pre,prey_test)
print('the MSE IS:',LR_MSE)


