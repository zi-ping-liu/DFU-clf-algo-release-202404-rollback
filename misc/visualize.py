# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 300


def gen_line_plot(df, title, xlabel, ylabel, labels = ['acc', 'sen', 'spe'], markers = ['o', 'd', '*']):

    for index, row in df.iterrows():
        plt.plot(row.index, row, label = labels[index], marker = markers[index])

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.show()
    
df = pd.DataFrame({
    'Original': [0.653, 0.577, 0.739], # US - N = 49 || 26 pos
    'change_MSI': [0.633, 0.577, 0.696], # US - N = 49 || 26 pos
    'change_3D': [0.633, 0.577, 0.696], # US - N = 49 || 26 pos
    'change_GT': [0.633, 0.571, 0.714],  # US - N = 49 || 28 pos
    'change_test_set': [0.671, 0.610, 0.737] # US - N = 79 || 41 pos
})
labels = ['acc', 'sen', 'spe']
gen_line_plot(df,
              title = 'ShiftWin Model Performance',
              xlabel = 'Experiment',
              ylabel = 'Metric Value')