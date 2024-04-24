import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np

def print_smoothing(data, x="time"):
  fig = plt.figure(figsize=(20, 30), dpi=80)
  plt.axis('off')
  plt.title("Sequential exponential smoothing results (a=0.25)")
  fig.tight_layout(pad=6.0,  h_pad=3, w_pad=3)
  fig.patch.set_visible(False)
  columns_count = 2
  axes = fig.subplots(nrows=12//columns_count, ncols=columns_count)

  axes[0, 0].plot(data["population"][x], data["population"]['pop'], "r", label="population")
  axes[0, 0].legend()
  axes[0, 0].set_xlabel(x)
  axes[0, 0].set_ylabel('pop')

  for count in range(1, 11):
    name = "exp_pop"+(str(count) if count!=1 else "")
    i, j = count//columns_count , count %columns_count
    axes[i, j].plot(data[name][x], data[name]["exp_pop" if count==1 else "exp_pop2"],"g" if count==10 else "b", label=name)
    axes[i, j].legend(loc="lower right")
    axes[i, j].set_xlabel(x)
    axes[i, j].set_ylabel('pop')
  fig.delaxes(axes[5, 1])

def print_smoothing_in_one_window(print_data1, print_data2, x, x_name="epochs", start_end = None, title = None, legend1 = "population", legend2 = "smoothing"):
  fig = plt.figure(figsize=(10, 10), dpi=80)
  plt.axis('off')
  plt.title(title or "Smoothing")
  fig.tight_layout(pad=6.0,  h_pad=3, w_pad=3)
  fig.patch.set_visible(False)
  columns_count = 2
  axes = fig.subplots(1)

  axes.plot(x, print_data1, "b", label=legend1)
  axes.plot(x, print_data2, "r", label=legend2)
  axes.legend(loc="lower right")
  axes.set_xlabel(x_name)
  axes.set_ylabel('pop')

#移动平均举例
def moving_average(series, k):
  print("return",np.average(series[-k:]) )
  return np.average(series[-k:])

if __name__ == "__main__":
  df1 = pd.read_csv('1.csv').iloc[:, 1:]

  time = df1["population"]["epochs"]
  population = df1["population"]["pop"]


  for k in [10, 50, 100]:
    smoothing_data = population[:k - 1]
    for i in range(k - 1, len(population)):
      y_i = np.average(population[i - (k - 1): i + 1])
      smoothing_data = pd.concat([smoothing_data, pd.Series(y_i)], ignore_index=True)
    print_smoothing_in_one_window(print_data1=population, print_data2=smoothing_data, x=time, legend2="moving average",
                                  title="Moving average with k = {}".format(k))


    for a in [0.1, 0.25, 0.5, 0.7]:
      smoothing_data = population[:1]
      for i in range(1, len(population)):
        y_i = a * population[i] + smoothing_data[i - 1] * (1 - a)
        smoothing_data = pd.concat([smoothing_data, pd.Series(y_i)], ignore_index=True)
      print_smoothing_in_one_window(print_data1=population, print_data2=smoothing_data, x=time, legend2="SES",
                                    title="simple exponential smoothing with a = {}".format(a))