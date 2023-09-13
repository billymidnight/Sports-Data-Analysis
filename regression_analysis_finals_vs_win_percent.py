
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the 'participants.csv' file into a DataFrame
participants_df = pd.read_csv('participants.csv')

trofre = df['finals_percent']
trophies = df['win_percent']
participant_names = df['name']

regression_coef = np.polyfit(trophies, trofre, 1)
regression_line = np.poly1d(regression_coef)


plt.figure(figsize=(10, 6))
plt.scatter(trophies, trofre, marker='o', color='red', alpha=0.75)

plt.plot(trophies, regression_line(trophies), color='red', label='Regression Line')



correlation_coefficient = np.corrcoef(trophies, trofre)[0, 1]


for i, name in enumerate(participant_names):
    plt.annotate(name, (trophies[i], trofre[i]), textcoords='offset points', xytext=(15, 0), ha='center', alpha = 0.4)

plt.xlabel('Win Percent')
plt.ylabel('Finals Percent')
plt.title('Scatter Plot: ')

plt.grid(True)

plt.text(0.5, 0.9, f'Correlation Coefficient: {correlation_coefficient:.2f}',
         transform=plt.gca().transAxes, fontsize=12)
plt.legend()
plt.show()


