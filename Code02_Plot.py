import numpy as np
import matplotlib.pyplot as plt

results_folder = 'results_lr_0_02/'

# load the results
loss_test_pre_array = np.load(results_folder + 'loss_test_pre_array.npy')
loss_array_blocked_array = np.load(results_folder + 'loss_array_blocked_array.npy')
loss_retention_blocked_array = np.load(results_folder + 'loss_retention_blocked_array.npy')
loss_test_blocked_array = np.load(results_folder + 'loss_test_blocked_array.npy')
loss_array_random_array = np.load(results_folder + 'loss_array_random_array.npy')
loss_retention_random_array = np.load(results_folder + 'loss_retention_random_array.npy')
loss_test_random_array = np.load(results_folder + 'loss_test_random_array.npy')
loss_retention_array_blocked_array = np.load(results_folder + 'loss_retention_array_blocked_array.npy')
loss_test_array_blocked_array = np.load(results_folder + 'loss_test_array_blocked_array.npy')
loss_retention_array_random_array = np.load(results_folder + 'loss_retention_array_random_array.npy')
loss_test_array_random_array = np.load(results_folder + 'loss_test_array_random_array.npy')
loss_test_array_pre_array = np.load(results_folder + 'loss_test_array_pre_array.npy')
loss_retention_noisy_array_blocked = np.load(results_folder + 'loss_retention_noisy_array_blocked.npy')
loss_test_noisy_array_blocked = np.load(results_folder + 'loss_test_noisy_array_blocked.npy')
loss_retention_noisy_array_random = np.load(results_folder + 'loss_retention_noisy_array_random.npy')
loss_test_noisy_array_random = np.load(results_folder + 'loss_test_noisy_array_random.npy')
loss_retention_pruned_array_blocked = np.load(results_folder + 'loss_retention_pruned_array_blocked.npy')
loss_test_pruned_array_blocked = np.load(results_folder + 'loss_test_pruned_array_blocked.npy')
loss_retention_pruned_array_random = np.load(results_folder + 'loss_retention_pruned_array_random.npy')
loss_test_pruned_array_random = np.load(results_folder + 'loss_test_pruned_array_random.npy')
loss_retention_interf_array_blocked = np.load(results_folder + 'loss_retention_interf_array_blocked.npy')
loss_test_interf_array_blocked = np.load(results_folder + 'loss_test_interf_array_blocked.npy')
loss_retention_interf_array_random = np.load(results_folder + 'loss_retention_interf_array_random.npy')
loss_test_interf_array_random = np.load(results_folder + 'loss_test_interf_array_random.npy')
print('All data loaded.')


# calculate correlation coefficient for the mean of the loss array
print('Significance of learnings')
from scipy.stats import pearsonr
corr_coefficient, p_value = pearsonr(np.array(range(1, loss_array_blocked_array.shape[1]+1)), loss_array_blocked_array.mean(axis=0))
print(f'Corr : {corr_coefficient}, p-value: {p_value}')
corr_coefficient, p_value = pearsonr(np.array(range(1, loss_array_blocked_array.shape[1]+1)), loss_array_random_array.mean(axis=0))
print(f'Corr : {corr_coefficient}, p-value: {p_value}')

print('\n\nDifference of test score between pre and retention')
# paired t-test for loss_test_pre_array and loss_retention_blocked_array
from scipy.stats import ttest_rel
print(f'Mean Loss Test Pre: {loss_test_pre_array.mean()}, Std: {loss_test_pre_array.std()}')
print(f'Mean Loss Retention Blocked: {loss_retention_blocked_array.mean()}, Std: {loss_retention_blocked_array.std()}')
t_stat, p_value = ttest_rel(loss_test_pre_array, loss_retention_blocked_array)
print(f't-statistic: {t_stat}, p-value: {p_value}')
# paired t-test for loss_test_pre_array and loss_retention_random_array
print(f'Mean Loss Test Pre: {loss_test_pre_array.mean()}, Std: {loss_test_pre_array.std()}')
print(f'Mean Loss Retention Random: {loss_retention_random_array.mean()}, Std: {loss_retention_random_array.std()}')
t_stat, p_value = ttest_rel(loss_test_pre_array, loss_retention_random_array)
print(f't-statistic: {t_stat}, p-value: {p_value}')


print('\n\nDifference of test score between pre and transfer-test')
# paired t-test for loss_test_pre_array and loss_test_blocked_array
print(f'Mean Loss Test Pre: {loss_test_pre_array.mean()}, Std: {loss_test_pre_array.std()}')
print(f'Mean Loss Test Blocked: {loss_test_blocked_array.mean()}, Std: {loss_test_blocked_array.std()}')
t_stat, p_value = ttest_rel(loss_test_pre_array, loss_test_blocked_array)
print(f't-statistic: {t_stat}, p-value: {p_value}')
# paired t-test for loss_test_pre_array and loss_test_random_array
print(f'Mean Loss Test Pre: {loss_test_pre_array.mean()}, Std: {loss_test_pre_array.std()}')
print(f'Mean Loss Test Random: {loss_test_random_array.mean()}, Std: {loss_test_random_array.std()}')
t_stat, p_value = ttest_rel(loss_test_pre_array, loss_test_random_array)
print(f't-statistic: {t_stat}, p-value: {p_value}')

print('\n\nDifference between blocked and random practice')
# paired t-test for loss_retention_blocked_array and loss_retention_random_array
from scipy.stats import ttest_rel
t_stat, p_value = ttest_rel(loss_retention_blocked_array, loss_retention_random_array)
print(f'Retention -- t-statistic: {t_stat}, p-value: {p_value}')
# paired t-test for loss_test_blocked_array and loss_test_random_array
t_stat, p_value = ttest_rel(loss_test_blocked_array, loss_test_random_array)
print(f'Transfer -- t-statistic: {t_stat}, p-value: {p_value}')

## diplay the mean and std of six divisions of one depicted retention performance
print('\nOne depicted Retention Performance (mean ± std)')
iter_idx = 2
mean_loss_array_blocked_6_first = np.mean(np.split(loss_retention_array_blocked_array[iter_idx, :], 6), axis=1)
std_loss_array_blocked_6_first = np.std(np.split(loss_retention_array_blocked_array[iter_idx, :], 6), axis=1)

print('Blocked:', end=' ')
for mean, std in zip(mean_loss_array_blocked_6_first, std_loss_array_blocked_6_first):
    print(f'{mean:.2f} ± {std:.2f},', end=' ')
print()

# Plot the results of the mean for each six parts
plt.figure()
mean_loss_array_blocked = loss_array_blocked_array.mean(axis=0)
mean_loss_array_random = loss_array_random_array.mean(axis=0)
# divide the mean loss array into six parts and average them
mean_loss_array_blocked_6 = np.mean(np.split(mean_loss_array_blocked, 3), axis=1)
mean_loss_array_random_6 = np.mean(np.split(mean_loss_array_random, 3), axis=1)
plt.plot(mean_loss_array_blocked_6, label='RP', marker='o', markersize=8)
plt.plot(mean_loss_array_random_6, label='IP', marker='o', markersize=8)
plt.xlabel('Phase')
plt.ylabel('Loss')
plt.ylim(0, 2.5)
plt.legend()
plt.title('Learning graph of each six epochs')
plt.show()


# Plot the results of the mean and error bar of the loss
plt.figure()
plt.plot(range(1, loss_array_blocked_array.shape[1]+1), loss_array_blocked_array.mean(axis=0), label='RP')
plt.fill_between(range(1, loss_array_blocked_array.shape[1]+1), loss_array_blocked_array.mean(axis=0) - 1.96*loss_array_blocked_array.std(axis=0)/np.sqrt(loss_array_blocked_array.shape[1]), loss_array_blocked_array.mean(axis=0) + 1.96*loss_array_blocked_array.std(axis=0)/np.sqrt(loss_array_blocked_array.shape[1]),
                 alpha=0.5)
plt.plot(range(1, loss_array_random_array.shape[1]+1), loss_array_random_array.mean(axis=0), label='IP')
plt.fill_between(range(1, loss_array_random_array.shape[1]+1), loss_array_random_array.mean(axis=0) - 1.96*loss_array_random_array.std(axis=0)/np.sqrt(loss_array_blocked_array.shape[1]), loss_array_random_array.mean(axis=0) + 1.96*loss_array_random_array.std(axis=0)/np.sqrt(loss_array_blocked_array.shape[1]),
                 alpha=0.5)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.ylim(0, 2.5)
plt.legend()
plt.title('Learning graph')
plt.show()


# Grouped bar chart the results of the mean and error bar of the loss for pre, retention, and test
plt.figure()
barWidth = 0.25
r1 = np.arange(3)
r2 = [x + barWidth for x in r1]
plt.bar(r1, [0, loss_retention_blocked_array.mean(), loss_test_blocked_array.mean()], yerr=[0, loss_retention_blocked_array.std()/np.sqrt(np.size(loss_test_pre_array)), loss_test_blocked_array.std()/np.sqrt(np.size(loss_test_pre_array))], width=barWidth, label='RP')
plt.bar(r2, [loss_test_pre_array.mean(), loss_retention_random_array.mean(), loss_test_random_array.mean()], yerr=[loss_test_pre_array.std()/np.sqrt(np.size(loss_test_pre_array)), loss_retention_random_array.std()/np.sqrt(np.size(loss_test_pre_array)), loss_test_random_array.std()/np.sqrt(np.size(loss_test_pre_array))], width=barWidth, label='IP')
plt.xticks([r + barWidth/2 for r in range(3)], ['Pre', 'Retention', 'Transfer'])
plt.ylabel('Loss')
# plt.ylim(2, 8.5)
plt.legend(loc='upper left')
plt.title('Comparison of Blocked and Random Practice')
plt.show()


def plot_with_ci(data_blocked, data_random, xlabel, ylabel, title):
    plt.figure()
    n_blocked = data_blocked.shape[0]
    n_random = data_random.shape[0]

    # mean & CI
    mean_blocked = data_blocked.mean(axis=0)
    ci_blocked = 1.96 * data_blocked.std(axis=0) / np.sqrt(n_blocked)

    mean_random = data_random.mean(axis=0)
    ci_random = 1.96 * data_random.std(axis=0) / np.sqrt(n_random)

    x = np.arange(mean_blocked.shape[0])

    # plot mean + CI
    plt.plot(mean_blocked, label='RP')
    plt.fill_between(x, mean_blocked - ci_blocked, mean_blocked + ci_blocked, alpha=0.2)

    plt.plot(mean_random, label='IP')
    plt.fill_between(x, mean_random - ci_random, mean_random + ci_random, alpha=0.2)

    # paired t-test at each point
    for i in range(mean_blocked.shape[0]):
        _, p = ttest_rel(data_blocked[:, i], data_random[:, i])

        if p < 0.05:
            star = '*'
        else:
            star = ''

        if star:  # 유의미하면 annotation
            y_max = max(mean_blocked[i] + ci_blocked[i], mean_random[i] + ci_random[i])
            plt.text(i, y_max + 0.05, star, ha='center', va='bottom', fontsize=12, color='red')

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(loc='lower right')
    plt.title(title)
    plt.show()


# Noise vulnerability test
plot_with_ci(loss_retention_noisy_array_blocked, loss_retention_noisy_array_random,
             'Repetition of Noisy Injection', 'Loss',
             'Noise Vulnerability Test - Trained Sequence')

plot_with_ci(loss_test_noisy_array_blocked, loss_test_noisy_array_random,
             'Repetition of Noisy Injection', 'Loss',
             'Noise Vulnerability Test - Generalization Set')

# Pruning vulnerability test
plot_with_ci(loss_retention_pruned_array_blocked, loss_retention_pruned_array_random,
             'Repetition of Weight Pruning', 'Loss',
             'Adversarial Pruning Test - Trained Sequence')

plot_with_ci(loss_test_pruned_array_blocked, loss_test_pruned_array_random,
             'Repetition of Weight Pruning', 'Loss',
             'Adversarial Pruning Test - Generalization Set')

# Interference vulnerability test
plot_with_ci(loss_retention_interf_array_blocked, loss_retention_interf_array_random,
             'Retraining Epoch', 'Loss',
             'Interference Vulnerability Test - Trained Sequence')

plot_with_ci(loss_test_interf_array_blocked, loss_test_interf_array_random,
             'Retraining Epoch', 'Loss',
             'Interference Vulnerability Test - Generalization Set')


## violin plot for six different retention performances in blocked vs random practice
seq_num = 3
loss_retention_array_blocked_each = np.split(loss_retention_array_blocked_array, seq_num, axis=1)
loss_retention_array_random_each = np.split(loss_retention_array_random_array, seq_num, axis=1)

# flatten each array
loss_retention_array_blocked_each_vec = [arr.flatten() for arr in loss_retention_array_blocked_each]
loss_retention_array_random_each_vec = [arr.flatten() for arr in loss_retention_array_random_each]

# import seaborn and pandas
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# dataframe for blocked
df_blocked = pd.DataFrame({
    'Loss': np.concatenate(loss_retention_array_blocked_each_vec),
    'Phase': np.concatenate([['Seq-' + str(i+1)] * len(loss_retention_array_blocked_each_vec[i]) for i in range(seq_num)]),
    'Condition': 'RP'
})

# dataframe for random
df_random = pd.DataFrame({
    'Loss': np.concatenate(loss_retention_array_random_each_vec),
    'Phase': np.concatenate([['Seq-' + str(i+1)] * len(loss_retention_array_random_each_vec[i]) for i in range(seq_num)]),
    'Condition': 'IP'
})

# 합치기
df_all = pd.concat([df_blocked, df_random], ignore_index=True)

# Create the violin plot
plt.figure(figsize=(10, seq_num))
sns.violinplot(
    x='Phase',
    y='Loss',
    hue='Condition',       # Blocked vs Random 구분
    data=df_all,
    palette={'RP': '#1f77b4', 'IP': '#ff7f0e'},
)

plt.show()

## violin plot for loss_retention_array_blocked_array and loss_retention_array_random_array and loss_test_array_blocked_array and loss_test_array_random_array
# first vectorize the those four arrays
loss_retention_array_blocked_vec = loss_retention_array_blocked_array.flatten()
loss_retention_array_random_vec = loss_retention_array_random_array.flatten()
loss_test_array_blocked_vec = loss_test_array_blocked_array.flatten()
loss_test_array_random_vec = loss_test_array_random_array.flatten()
loss_test_array_pre_vec = loss_test_array_pre_array.flatten()

# import seaborn and pandas
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# dataframe for main
df_main = pd.DataFrame({
    'Loss': np.concatenate([
        loss_test_array_blocked_vec,
        loss_test_array_random_vec
    ]),
    'Phase': (
            ['Generalization'] * (len(loss_test_array_blocked_vec) + len(loss_test_array_random_vec))
    ),
    'Type': (['RP'] * len(loss_test_array_blocked_vec)
        + ['IP'] * len(loss_test_array_random_vec)
    )
})

# dataframe for pre
df_pre = pd.DataFrame({
    'Loss': loss_test_array_pre_vec,
    'Phase': ['Pre'] * len(loss_test_array_pre_vec),
})
# Create the violin plot
plt.figure(figsize=(10, 6))
# Pre violin plot (grey)
sns.violinplot(
    x='Phase',
    y='Loss',
    data=df_pre,
    color='grey'
)

# Blocked/Random violin plot
sns.violinplot(
    x='Phase',
    y='Loss',
    hue='Type',
    data=df_main,
    palette={'RP': '#1f77b4', 'IP': '#ff7f0e'},
)

plt.legend(title='Type', loc='upper left')
plt.show()

# first vectorize the those four arrays

## violin plot for one depicted retention performance
# first vectorize the those four arrays
loss_retention_array_blocked_first = loss_retention_array_blocked_array[iter_idx, :]
loss_retention_array_random_first = loss_retention_array_random_array[iter_idx, :]
loss_test_array_blocked_first = loss_test_array_blocked_array[iter_idx, :]
loss_test_array_random_first = loss_test_array_random_array[iter_idx, :]
loss_test_array_pre_first = loss_test_array_pre_array[iter_idx, :]

# import seaborn and pandas
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# dataframe for main
df_main_first = pd.DataFrame({
    'Loss': np.concatenate([
        loss_retention_array_blocked_first,
        loss_retention_array_random_first,
        loss_test_array_blocked_first,
        loss_test_array_random_first
    ]),
    'Phase': (
        ['Retention'] * (len(loss_retention_array_blocked_first) + len(loss_retention_array_random_first))
        + ['Transfer'] * (len(loss_test_array_blocked_first) + len(loss_test_array_random_first))
    ),
    'Type': (
        ['RP'] * len(loss_retention_array_blocked_first)
        + ['IP'] * len(loss_retention_array_random_first)
        + ['RP'] * len(loss_test_array_blocked_first)
        + ['IP'] * len(loss_test_array_random_first)
    )
})

# dataframe for pre
df_pre_first = pd.DataFrame({
    'Loss': loss_test_array_pre_first,
    'Phase': ['Pre'] * len(loss_test_array_pre_first),
})

# Create the violin plot
plt.figure(figsize=(10, 6))
# Pre violin plot (grey)
sns.violinplot(
    x='Phase',
    y='Loss',
    data=df_pre_first,
    color='grey'
)

# Blocked/Random violin plot
sns.violinplot(
    x='Phase',
    y='Loss',
    hue='Type',
    data=df_main_first,
    order=['Retention', 'Transfer'],  # 먼저 retention/test
    palette={'RP': '#1f77b4', 'IP': '#ff7f0e'},
)

plt.legend(title='Type', loc='upper left')
plt.show()

## save all loss_* data into separate csv files
pd.DataFrame(loss_test_pre_array).to_csv(results_folder + 'loss_test_pre_array.csv', index=False)
pd.DataFrame(loss_array_blocked_array).to_csv(results_folder + 'loss_array_blocked_array.csv', index=False)
pd.DataFrame(loss_retention_blocked_array).to_csv(results_folder + 'loss_retention_blocked_array.csv', index=False)
pd.DataFrame(loss_test_blocked_array).to_csv(results_folder + 'loss_test_blocked_array.csv', index=False)
pd.DataFrame(loss_array_random_array).to_csv(results_folder + 'loss_array_random_array.csv', index=False)
pd.DataFrame(loss_retention_random_array).to_csv(results_folder + 'loss_retention_random_array.csv', index=False)
pd.DataFrame(loss_test_random_array).to_csv(results_folder + 'loss_test_random_array.csv', index=False)
pd.DataFrame(loss_retention_array_blocked_array).to_csv(results_folder + 'loss_retention_array_blocked_array.csv', index=False)
pd.DataFrame(loss_test_array_blocked_array).to_csv(results_folder + 'loss_test_array_blocked_array.csv', index=False)
pd.DataFrame(loss_retention_array_random_array).to_csv(results_folder + 'loss_retention_array_random_array.csv', index=False)
pd.DataFrame(loss_test_array_random_array).to_csv(results_folder + 'loss_test_array_random_array.csv', index=False)
pd.DataFrame(loss_test_array_pre_array).to_csv(results_folder + 'loss_test_array_pre_array.csv', index=False)
pd.DataFrame(loss_retention_noisy_array_blocked).to_csv(results_folder + 'loss_retention_noisy_array_blocked.csv', index=False)
pd.DataFrame(loss_test_noisy_array_blocked).to_csv(results_folder + 'loss_test_noisy_array_blocked.csv', index=False)
pd.DataFrame(loss_retention_noisy_array_random).to_csv(results_folder + 'loss_retention_noisy_array_random.csv', index=False)
pd.DataFrame(loss_test_noisy_array_random).to_csv(results_folder + 'loss_test_noisy_array_random.csv', index=False)
pd.DataFrame(loss_retention_pruned_array_blocked).to_csv(results_folder + 'loss_retention_pruned_array_blocked.csv', index=False)
pd.DataFrame(loss_test_pruned_array_blocked).to_csv(results_folder + 'loss_test_pruned_array_blocked.csv', index=False)
pd.DataFrame(loss_retention_pruned_array_random).to_csv(results_folder + 'loss_retention_pruned_array_random.csv', index=False)
pd.DataFrame(loss_test_pruned_array_random).to_csv(results_folder + 'loss_test_pruned_array_random.csv', index=False)
pd.DataFrame(loss_retention_interf_array_blocked).to_csv(results_folder + 'loss_retention_interf_array_blocked.csv', index=False)
pd.DataFrame(loss_test_interf_array_blocked).to_csv(results_folder + 'loss_test_interf_array_blocked.csv', index=False)
pd.DataFrame(loss_retention_interf_array_random).to_csv(results_folder + 'loss_retention_interf_array_random.csv', index=False)
pd.DataFrame(loss_test_interf_array_random).to_csv(results_folder + 'loss_test_interf_array_random.csv', index=False)
print('All data saved into csv files.')