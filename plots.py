import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob


# list = []
# list = glob.glob("*.txt")
# print(list)

"""
de_file = "N1024-rho0.85-T1.5-sigMin0.9-sigMax1.1-bulk0-Lz12.0-binwidth0.1-wNeutrality1-energy.txt"
de_file2 = "N1024-rho0.85-T1.5-sigMin0.8-sigMax1.2-bulk0-Lz12.0-binwidth0.1-wNeutrality1-energy.txt"
de_file3 = "N1024-rho0.85-T1.5-sigMin0.7-sigMax1.3-bulk0-Lz12.0-binwidth0.1-wNeutrality1-energy.txt"
de_file4 = "N1024-rho0.85-T1.5-sigMin0.6-sigMax1.4-bulk0-Lz12.0-binwidth0.1-wNeutrality1-energy.txt"
de_file5 = "N1024-rho0.85-T1.5-sigMin0.5-sigMax1.5-bulk0-Lz12.0-binwidth0.1-wNeutrality1-energy.txt"
de_file6 = "N1024-rho0.85-T1.5-sigMin0.4-sigMax1.6-bulk0-Lz12.0-binwidth0.1-wNeutrality1-energy.txt"
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
plt.xlim(150000)
plt.xlabel("Time")
plt.ylabel("Potential Energy")
plt.title("N = 1024 T = 1.5 ρ = 0.85 Wall Neutrality = 1")
leg = plt.legend();
plt.show()
"""

#"""
de_file = "N1024-rho0.85-T1.5-sigMin0.9-sigMax1.1-bulk0-Lz12.0-binwidth0.1-wNeutrality1-density.txt"
de_file2 = "N1024-rho0.85-T1.5-sigMin0.8-sigMax1.2-bulk0-Lz12.0-binwidth0.1-wNeutrality1-density.txt"
de_file3 = "N1024-rho0.85-T1.5-sigMin0.7-sigMax1.3-bulk0-Lz12.0-binwidth0.1-wNeutrality1-density.txt"
de_file4 = "N1024-rho0.85-T1.5-sigMin0.6-sigMax1.4-bulk0-Lz12.0-binwidth0.1-wNeutrality1-density.txt"
de_file5 = "N1024-rho0.85-T1.5-sigMin0.5-sigMax1.5-bulk0-Lz12.0-binwidth0.1-wNeutrality1-density.txt"
de_file6 = "N1024-rho0.85-T1.5-sigMin0.4-sigMax1.6-bulk0-Lz12.0-binwidth0.1-wNeutrality1-density.txt"
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
plt.title("N = 1024 T = 1.5 ρ = 0.85 Wall Neutrality = 1")
leg = plt.legend();
plt.show()
#"""