import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob


# list = []
# list = glob.glob("*.dat")
# print(list)
# exit()


"""
#de_file =  "RUMD-Software-LJ-N1024-rho1.0-T1.0-energy.dat"
#data = pd.read_csv(de_file, sep='\s+',header=None, skiprows=1)
x1, y1 = np.loadtxt('RUMD-Software-LJ-N1024-rho1.0-T1.0-energy.dat',unpack=True, usecols=[0,2], skiprows=2)
# x1 = data[1]
# y1 = data[3]
plt.plot (x1, y1, '-', label='N=1204')
plt.xlabel("Time")
plt.ylabel("Potential Energy")
plt.title("N = 1024, T = 1.5, ρ = 0.63, Wall Neutrality = 0")
leg = plt.legend();
plt.show()
"""


"""
de_file =  "N2048-rho0.85-T1.5-sigMin1.0-sigMax1.0-bulk0-Lz12.0-binwidth0.1-wNeutrality0-energy.txt"
de_file2 = "N2048-rho0.85-T1.5-sigMin0.9-sigMax1.1-bulk0-Lz12.0-binwidth0.1-wNeutrality0-energy.txt"
de_file3 = "N2048-rho0.85-T1.5-sigMin0.8-sigMax1.2-bulk0-Lz12.0-binwidth0.1-wNeutrality0-energy.txt"
de_file4 = "N2048-rho0.85-T1.5-sigMin0.7-sigMax1.3-bulk0-Lz12.0-binwidth0.1-wNeutrality0-energy.txt"
de_file5 = "N2048-rho0.85-T1.5-sigMin0.6-sigMax1.4-bulk0-Lz12.0-binwidth0.1-wNeutrality0-energy.txt"
de_file6 = "N2048-rho0.85-T1.5-sigMin0.5-sigMax1.5-bulk0-Lz12.0-binwidth0.1-wNeutrality0-energy.txt"
de_file7 = "N2048-rho0.85-T1.5-sigMin0.4-sigMax1.6-bulk0-Lz12.0-binwidth0.1-wNeutrality0-energy.txt"
data = pd.read_csv(de_file, sep='\s+',header=None, skiprows=1)
data2 = pd.read_csv(de_file2, sep='\s+',header=None, skiprows=1)
data3 = pd.read_csv(de_file3, sep='\s+',header=None, skiprows=1)
data4 = pd.read_csv(de_file4, sep='\s+',header=None, skiprows=1)
data5 = pd.read_csv(de_file5, sep='\s+',header=None, skiprows=1)
data6 = pd.read_csv(de_file6, sep='\s+',header=None, skiprows=1)
data7 = pd.read_csv(de_file7, sep='\s+',header=None, skiprows=1)
x1 = data[0]
y1 = data[1]
x2 = data2[0]
y2 = data2[1]
x3 = data3[0]
y3 = data3[1]
x4 = data4[0]
y4 = data4[1]
x5 = data5[0]
y5 = data5[1]
x6 = data6[0]
y6 = data6[1]
x7 = data7[0]
y7 = data7[1]
plt.plot(x1, y1, '-', label='0%')
plt.plot(x2, y2, '-', label='5.77%')
plt.plot(x3, y3, '-', label='11.54%')
plt.plot(x4, y4, '-', label='17.32%')
plt.plot(x5, y5, '-', label='23.09%')
plt.plot(x6, y6, '-', label='28.87%')
plt.plot(x7, y7, '-', label='34.64%')
plt.xlim(150000)
plt.xlabel("Time Steps")
plt.ylabel("Potential Energy")
plt.title("N = 2048, T = 1.5, ρ = 0.85, Wall Neutrality = 0")
leg = plt.legend();
plt.show()
"""

"""
de_file =  "N1024-rho0.85-T1.5-sigMin0.9-sigMax1.1-bulk0-Lz12.0-binwidth0.1-wNeutrality0-density.txt"
de_file2 = "N1024-rho0.85-T1.5-sigMin0.8-sigMax1.2-bulk0-Lz12.0-binwidth0.1-wNeutrality0-density.txt"
de_file3 = "N1024-rho0.85-T1.5-sigMin0.7-sigMax1.3-bulk0-Lz12.0-binwidth0.1-wNeutrality0-density.txt"
de_file4 = "N1024-rho0.85-T1.5-sigMin0.6-sigMax1.4-bulk0-Lz12.0-binwidth0.1-wNeutrality0-density.txt"
de_file5 = "N1024-rho0.85-T1.5-sigMin0.5-sigMax1.5-bulk0-Lz12.0-binwidth0.1-wNeutrality0-density.txt"
de_file6 = "N1024-rho0.85-T1.5-sigMin0.4-sigMax1.6-bulk0-Lz12.0-binwidth0.1-wNeutrality0-density.txt"
data = pd.read_csv(de_file, sep='\s+',header=None, skiprows=1)
data2 = pd.read_csv(de_file2, sep='\s+',header=None, skiprows=1)
data3 = pd.read_csv(de_file3, sep='\s+',header=None, skiprows=1)
data4 = pd.read_csv(de_file4, sep='\s+',header=None, skiprows=1)
data5 = pd.read_csv(de_file5, sep='\s+',header=None, skiprows=1)
data6 = pd.read_csv(de_file6, sep='\s+',header=None, skiprows=1)
x1 = data[0]
y1 = data[1]
x2 = data2[0]
y2 = data2[1]
x3 = data3[0]
y3 = data3[1]
x4 = data4[0]
y4 = data4[1]
x5 = data5[0]
y5 = data5[1]
x6 = data6[0]
y6 = data6[1]
plt.plot (x1, y1, '-', label='5.77%')
plt.plot(x2, y2, '-', label='11.54%')
plt.plot(x3, y3, '-', label='17.42%')
plt.plot(x4, y4, '-', label='23.09%')
plt.plot(x5, y5, '-', label='28.87%')
plt.plot(x6, y6, '-', label='34.64%')
plt.xlabel("Distance")
plt.ylabel("Density")
plt.title("N = 1024, T = 1.5, ρ = 0.85, Wall Neutrality = 0")
leg = plt.legend();
plt.show()
#"""



de_file =   "N1024-rho0.85-T1.5-sigMin1.0-sigMax1.0-bulk0-Lz12.0-binwidth0.1-wNeutrality0-energy.txt"
de_file2 =  "N1024-rho0.85-T1.5-sigMin0.9-sigMax1.1-bulk0-Lz12.0-binwidth0.1-wNeutrality0-energy.txt"
de_file3 =  "N1024-rho0.85-T1.5-sigMin0.8-sigMax1.2-bulk0-Lz12.0-binwidth0.1-wNeutrality0-energy.txt"
de_file4 =  "N1024-rho0.85-T1.5-sigMin0.7-sigMax1.3-bulk0-Lz12.0-binwidth0.1-wNeutrality0-energy.txt"
de_file5 =  "N1024-rho0.85-T1.5-sigMin0.6-sigMax1.4-bulk0-Lz12.0-binwidth0.1-wNeutrality0-energy.txt"
de_file6 =  "N1024-rho0.85-T1.5-sigMin0.5-sigMax1.5-bulk0-Lz12.0-binwidth0.1-wNeutrality0-energy.txt"
de_file7 =  "N1024-rho0.85-T1.5-sigMin0.4-sigMax1.6-bulk0-Lz12.0-binwidth0.1-wNeutrality0-energy.txt"
de_file8 =  "N2048-rho0.85-T1.5-sigMin1.0-sigMax1.0-bulk0-Lz12.0-binwidth0.1-wNeutrality0-energy.txt"
de_file9 =  "N2048-rho0.85-T1.5-sigMin0.9-sigMax1.1-bulk0-Lz12.0-binwidth0.1-wNeutrality0-energy.txt"
de_file10 = "N2048-rho0.85-T1.5-sigMin0.8-sigMax1.2-bulk0-Lz12.0-binwidth0.1-wNeutrality0-energy.txt"
de_file11 = "N2048-rho0.85-T1.5-sigMin0.7-sigMax1.3-bulk0-Lz12.0-binwidth0.1-wNeutrality0-energy.txt"
de_file12 = "N2048-rho0.85-T1.5-sigMin0.6-sigMax1.4-bulk0-Lz12.0-binwidth0.1-wNeutrality0-energy.txt"
de_file13 = "N2048-rho0.85-T1.5-sigMin0.5-sigMax1.5-bulk0-Lz12.0-binwidth0.1-wNeutrality0-energy.txt"
de_file14 = "N2048-rho0.85-T1.5-sigMin0.4-sigMax1.6-bulk0-Lz12.0-binwidth0.1-wNeutrality0-energy.txt"
data = pd.read_csv(de_file, sep='\s+',header=None, skiprows=1)
data2 = pd.read_csv(de_file2, sep='\s+',header=None, skiprows=1)
data3 = pd.read_csv(de_file3, sep='\s+',header=None, skiprows=1)
data4 = pd.read_csv(de_file4, sep='\s+',header=None, skiprows=1)
data5 = pd.read_csv(de_file5, sep='\s+',header=None, skiprows=1)
data6 = pd.read_csv(de_file6, sep='\s+',header=None, skiprows=1)
data7 = pd.read_csv(de_file7, sep='\s+',header=None, skiprows=1)
data8 = pd.read_csv(de_file8, sep='\s+',header=None, skiprows=1)
data9 = pd.read_csv(de_file9, sep='\s+',header=None, skiprows=1)
data10 = pd.read_csv(de_file10, sep='\s+',header=None, skiprows=1)
data11 = pd.read_csv(de_file11, sep='\s+',header=None, skiprows=1)
data12 = pd.read_csv(de_file12, sep='\s+',header=None, skiprows=1)
data13 = pd.read_csv(de_file13, sep='\s+',header=None, skiprows=1)
data14 = pd.read_csv(de_file14, sep='\s+',header=None, skiprows=1)
x1 = data[0]
y1 = data[1]
x2 = data2[0]
y2 = data2[1]
x3 = data3[0]
y3 = data3[1]
x4 = data4[0]
y4 = data4[1]
x5 = data5[0]
y5 = data5[1]
x6 = data6[0]
y6 = data6[1]
x7 = data7[0]
y7 = data7[1]
x8 = data[0]
y8 = data[1]
x9 = data2[0]
y9 = data2[1]
x10 = data3[0]
y10 = data3[1]
x11 = data4[0]
y11 = data4[1]
x12 = data5[0]
y12 = data5[1]
x13 = data6[0]
y13 = data6[1]
x14 = data7[0]
y14 = data7[1]
plt.plot(x1, y1, '-', color = 'red', label='1024 0%')
plt.plot(x2, y2, '-', color = 'dodgerblue', label='1024 5.77%')
plt.plot(x3, y3, '-', color = 'pink', label='1024 11.54%')
plt.plot(x4, y4, '-', color = 'yellow', label='1024 17.32%')
plt.plot(x5, y5, '-', color = 'blue', label='1024 23.09%')
plt.plot(x6, y6, '-', color = 'purple', label='1024 28.87%')
plt.plot(x7, y7, '-', color = 'orange', label='1024 34.64%')
plt.plot(x8, y8, '--', color = 'black', label='2048 0%')
plt.plot(x9, y9, '--', color = 'brown', label='2048 5.77%')
plt.plot(x10, y10, '--', color = 'lime', label='2048 11.54%')
plt.plot(x11, y11, '--', color = 'midnightblue', label='2048 17.32%')
plt.plot(x12, y12, '--', color = 'salmon', label='2048 23.09%')
plt.plot(x13, y13, '--', color = 'gold', label='2048 28.87%')
plt.plot(x14, y14, '--', color = 'cyan', label='2048 34.64%')
plt.xlim(150000)
plt.xlabel("Time Steps")
plt.ylabel("Potential Energy")
plt.title("N = 1024 & 2048, T = 1.5, ρ = 0.85, Wall Neutrality = 0")
leg = plt.legend();
plt.show()