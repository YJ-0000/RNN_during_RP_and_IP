import numpy as np
import matplotlib.pyplot as plt

results_folder = 'results/'

# load the results
loss_test_pre_array = np.load(results_folder + 'loss_test_pre_array.npy')
loss_array_blocked_array = np.load(results_folder + 'loss_array_blocked_array.npy')
loss_retention_blocked_array = np.load(results_folder + 'loss_retention_blocked_array.npy')
loss_test_blocked_array = np.load(results_folder + 'loss_test_blocked_array.npy')
loss_array_random_array = np.load(results_folder + 'loss_array_random_array.npy')
loss_retention_random_array = np.load(results_folder + 'loss_retention_random_array.npy')
loss_test_random_array = np.load(results_folder + 'loss_test_random_array.npy')

# calculate correlation coefficient for the mean of the loss array
print('Significance of learnings')
from scipy.stats import pearsonr
corr_coefficient, p_value = pearsonr(np.array(range(1, loss_array_blocked_array.shape[1]+1)), loss_array_blocked_array.mean(axis=0))
print(f'Corr : {corr_coefficient}, p-value: {p_value}')
corr_coefficient, p_value = pearsonr(np.array(range(1, loss_array_blocked_array.shape[1]+1)), loss_array_random_array.mean(axis=0))
print(f'Corr : {corr_coefficient}, p-value: {p_value}')

print('\n\nDifference of test score between pre and post-practice')
# paired t-test for loss_test_pre_array and loss_test_blocked_array
from scipy.stats import ttest_rel
t_stat, p_value = ttest_rel(loss_test_pre_array, loss_test_blocked_array)
print(f't-statistic: {t_stat}, p-value: {p_value}')
# paired t-test for loss_test_pre_array and loss_test_random_array
t_stat, p_value = ttest_rel(loss_test_pre_array, loss_test_random_array)
print(f't-statistic: {t_stat}, p-value: {p_value}')

print('\n\nDifference between blocked and random practice')
# paired t-test for loss_retention_blocked_array and loss_retention_random_array
from scipy.stats import ttest_rel
t_stat, p_value = ttest_rel(loss_retention_blocked_array, loss_retention_random_array)
print(f't-statistic: {t_stat}, p-value: {p_value}')
# paired t-test for loss_test_blocked_array and loss_test_random_array
t_stat, p_value = ttest_rel(loss_test_blocked_array, loss_test_random_array)
print(f't-statistic: {t_stat}, p-value: {p_value}')


# Plot the results of the mean for each six parts
plt.figure()
mean_loss_array_blocked = loss_array_blocked_array.mean(axis=0)
mean_loss_array_random = loss_array_random_array.mean(axis=0)
# divide the mean loss array into six parts and average them
mean_loss_array_blocked_6 = np.mean(np.split(mean_loss_array_blocked, 6), axis=1)
mean_loss_array_random_6 = np.mean(np.split(mean_loss_array_random, 6), axis=1)
plt.plot(mean_loss_array_blocked_6, label='Blocked', marker='o', markersize=8)
plt.plot(mean_loss_array_random_6, label='Random', marker='o', markersize=8)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.ylim(2, 8.5)
plt.legend()
plt.title('Learning graph of each six epochs')
plt.show()


# Plot the results of the mean and error bar of the loss
plt.figure()
plt.plot(range(1, loss_array_blocked_array.shape[1]+1), loss_array_blocked_array.mean(axis=0), label='Blocked')
plt.fill_between(range(1, loss_array_blocked_array.shape[1]+1), loss_array_blocked_array.mean(axis=0) - 1.96*loss_array_blocked_array.std(axis=0)/np.sqrt(loss_array_blocked_array.shape[1]), loss_array_blocked_array.mean(axis=0) + 1.96*loss_array_blocked_array.std(axis=0)/np.sqrt(loss_array_blocked_array.shape[1]),
                 alpha=0.5)
plt.plot(range(1, loss_array_random_array.shape[1]+1), loss_array_random_array.mean(axis=0), label='Random')
plt.fill_between(range(1, loss_array_random_array.shape[1]+1), loss_array_random_array.mean(axis=0) - 1.96*loss_array_random_array.std(axis=0)/np.sqrt(loss_array_blocked_array.shape[1]), loss_array_random_array.mean(axis=0) + 1.96*loss_array_random_array.std(axis=0)/np.sqrt(loss_array_blocked_array.shape[1]),
                 alpha=0.5)
plt.xlabel('Trials')
plt.ylabel('Loss')
plt.ylim(2, 8.5)
plt.legend()
plt.title('Learning graph')
plt.show()


# Grouped bar chart the results of the mean and error bar of the loss for retention and test
plt.figure()
barWidth = 0.25
r1 = np.arange(2)
r2 = [x + barWidth for x in r1]
plt.bar(r1, [loss_retention_blocked_array.mean(), loss_test_blocked_array.mean()], yerr=[loss_retention_blocked_array.std()/np.sqrt(np.size(loss_test_pre_array)), loss_test_blocked_array.std()/np.sqrt(np.size(loss_test_pre_array))], width=barWidth, label='Blocked')
plt.bar(r2, [loss_retention_random_array.mean(), loss_test_random_array.mean()], yerr=[loss_retention_random_array.std()/np.sqrt(np.size(loss_test_pre_array)), loss_test_random_array.std()/np.sqrt(np.size(loss_test_pre_array))], width=barWidth, label='Random')
plt.xticks([r + barWidth/2 for r in range(2)], ['Retention', 'Transfer'])
plt.ylabel('Loss')
plt.ylim(2, 8.5)
plt.legend(loc='upper left')
plt.title('Comparison of Blocked and Random Practice')
plt.show()

# Grouped bar chart the results of the mean and error bar of the loss for pre, retention, and test
plt.figure()
barWidth = 0.25
r1 = np.arange(3)
r2 = [x + barWidth for x in r1]
plt.bar(r1, [0, loss_retention_blocked_array.mean(), loss_test_blocked_array.mean()], yerr=[0, loss_retention_blocked_array.std()/np.sqrt(np.size(loss_test_pre_array)), loss_test_blocked_array.std()/np.sqrt(np.size(loss_test_pre_array))], width=barWidth, label='Blocked')
plt.bar(r2, [loss_test_pre_array.mean(), loss_retention_random_array.mean(), loss_test_random_array.mean()], yerr=[loss_test_pre_array.std()/np.sqrt(np.size(loss_test_pre_array)), loss_retention_random_array.std()/np.sqrt(np.size(loss_test_pre_array)), loss_test_random_array.std()/np.sqrt(np.size(loss_test_pre_array))], width=barWidth, label='Random')
plt.xticks([r + barWidth/2 for r in range(3)], ['Pre', 'Retention', 'Transfer'])
plt.ylabel('Loss')
plt.ylim(2, 8.5)
plt.legend(loc='upper left')
plt.title('Comparison of Blocked and Random Practice')
plt.show()
