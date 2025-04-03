import pandas as pd
import matplotlib.pyplot as plt

def finding_tops(df, window_size):
    df_x = df.iloc[:, 0]
    df_y = df.iloc[:, 1]

    tops_x, tops_y = [], []
    downs_x, downs_y = [], []

    for i in range(window_size, len(df_x) - window_size):
        window = df_y[i - window_size: i + window_size+1]
        middle = df_y[i]

        if middle == max(window):
            tops_x.append(df_x[i])
            tops_y.append(df_y[i])

        if middle == min(window):
            downs_x.append(df_x[i])
            downs_y.append(df_y[i])

    return tops_x, tops_y, downs_x, downs_y

data_set = pd.read_csv('kgh_d-2.csv')
data_for_chart = data_set[['Data', 'Zamkniecie']]

tops_x, tops_y, downs_x, downs_y = finding_tops(data_for_chart, 1)

plt.plot(data_for_chart['Data'], data_for_chart['Zamkniecie'])
plt.plot(tops_x, tops_y, c='red')
plt.plot(downs_x, downs_y, c='blue')
plt.show()