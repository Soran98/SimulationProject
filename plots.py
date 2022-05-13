import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


de_file = "N1024-rho1.2-T1.0-sigMin0.8-sigMax1.2-bulk0-Lz12.0energy.txt"
de_file2 = "N1024-rho1.2-T1.0-sigMin0.8-sigMax1.2-bulk0-Lz12.0energy(1).txt"
data = pd.read_csv(de_file, sep='\s+',header=None, skiprows=1)
data2 = pd.read_csv(de_file2, sep='\s+',header=None, skiprows=1)
x1 = data[0]
y1 = data[1]
x2 = data2[0]
y2 = data2[1]
plt.plot (x1, y1, 'o', label='yesterday')
plt.plot(x2, y2, '-', label='today')
plt.xlabel("Distance")
plt.ylabel("e")
leg = plt.legend();
plt.show()

# sigma = 2
# sigma1 = 1
# # 100 linearly spaced numbers
# x = np.linspace(2,7,100)

# # the function, which is y = x^2 here
# y = 4 * (((sigma / x) **12) - ((sigma / x) **6))
# y1 = 4 * (((sigma1 / x) **12) - ((sigma1 / x) **6))

# # setting the axes at the centre
# fig = plt.figure()
# ax = fig.add_subplot(1, 1, 1)
# ax.spines['left'].set_position('center')
# ax.spines['bottom'].set_position('zero')
# ax.spines['right'].set_color('none')
# ax.spines['top'].set_color('none')
# ax.xaxis.set_ticks_position('bottom')
# ax.yaxis.set_ticks_position('left')

# # plot the function
# plt.plot(x,y, 'r')
# plt.plot(x,y1, 'r')

# # show the plot
# plt.show()


# z = [1, 3, 5, 6, 9]

# for i in range(z):
#     for j in range(z):
#         dz = z[i] - z[j]
# print(dz) 
