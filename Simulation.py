from ast import Or
from cmath import log, sqrt
import random
from re import T
from statistics import variance
from turtle import distance
import numpy as np                   
import sys, time
from numba import jit
import matplotlib.pyplot as plt


# Simulation parameters
N = 100  # number of cells
rho = 0.8  # density
T = 1  # tempurature 
dt = 0.005
kb = 1      #this is 1.381 * 10**-23 refer to power check
m = 1  #temporary 
minDist = 0.9 # the minimum distance each 
iseed = 10
nsteps = 20000 #how many steps 



# Box length
vol = N/rho # volume
L = np.power(vol, 1 / 3)  # length of the simulation
Lx = L
Ly = L
Lz = L
print("L = ", L)

#--------------------------------------------------------------------------
#Lennard-Jones potential parameters
epsilon = 1
sigma = 1
rcut = 2.5
sigma12 = sigma ** 12
sigma6 = sigma ** 6
rcut2 = rcut * rcut
rcutsq = rcut2
offset = 4.0 * epsilon* (sigma12 / rcut**12 - sigma6 / rcut**6)
print("LJ parameters: sigma=%s epsilon=%s rcut=%s offset=%s "%(sigma, epsilon, rcut, offset))
#--------------------------------------------------------------------------


#--------------------------------------------------------------------------
#  creating of Force
#--------------------------------------------------------------------------
@jit(nopython=True)
def Force(x,y,z,fx,fy,fz):
    u = 0
    for i in range(N):
        fx[i] = 0.0
        fy[i] = 0.0
        fz[i] = 0.0

    for i in range(N):
        for j in range(i+1, N):
            dx = x[i] - x[j]
            dx = dx - Lx * np.round(dx/Lx) # minimum image distance

            dy = y[i] - y[j]
            dy = dy - Ly * np.round(dy/Ly)

            dz = z[i] - z[j]
            dz = dz - Lz * np.round(dz/Lz)

            dr2 = dx**2 + dy**2 + dz**2
            #inv = inverse
            if dr2 < rcutsq:
                dr2inv = 1/dr2
                dr6inv = dr2inv * dr2inv * dr2inv
                dr12inv = dr6inv * dr6inv

                du = 4 * epsilon * ((sigma12 * dr12inv) - (sigma6 * dr6inv)) - offset
                u = u + du 

                wij = 4 * epsilon * (12 * sigma12 * dr12inv - 6 * sigma6 * dr6inv)
                wij = wij * dr2inv
                fx[i] = fx[i] + wij * dx
                fy[i] = fy[i] + wij * dy
                fz[i] = fz[i] + wij * dz

                fx[j] = fx[j] - wij * dx
                fy[j] = fy[j] - wij * dy
                fz[j] = fz[j] - wij * dz
    return u/N
#------------------------------------------------------------------------
#  generate 3 random numbers between 0 and L
#--------------------------------------------------------------------------
def x_rand(L):
    x = random.uniform(0, L)
    y = random.uniform(0, L)
    z = random.uniform(0, L)
    return x, y, z
#--------------------------------------------------------------------------
#  function to generate randomc ocnfiguration s.t. distance between 2 particles > minDist
#--------------------------------------------------------------------------
def InitConf(minDist):
    x[0], y[0], z[0] = x_rand(L) # choose an arbitrary position for the very 1st particle
    i = 0
    while i < N:
            x_t, y_t, z_t = x_rand(L) # trial position
            iflag = 1 # flag for accepting trial position in x, y, z list if dist > minDist
            for j in range(i): # look for all possible pairs
                dx = x[j] - x_t
                dx = dx - L * np.round(dx/L) # minimun image distance

                dy = y[j] - y_t
                dy = dy - L * np.round(dy/L)

                dz = z[j] - z_t
                dz = dz - L * np.round(dz/L)

                dr2 = dx**2 + dy**2 + dz**2
                if(dr2 < minDist*minDist):
                    iflag = 0 # iflag=0 means don't accept the trial position: see later lines
                    break
            if(iflag==1): # this line will reach (i) by above break statement or (ii) after finishing above for loop
                x[i] = x_t; y[i] = y_t; z[i] = z_t; i = i + 1
    #print(x[0],x[2],x[70])
#--------------------------------------------------------------------------
#  function to calculate distance of 2 particles (x1,y1,z1) and (x2,y2,z2)
#--------------------------------------------------------------------------
def dist(x1, y1, z1, x2, y2, z2, Lx, Ly, Lz):   #Use these variables in gaussian def
    dx = x1 - x2
    dx = dx - Lx * np.round(dx/Lx) # minimum image distance

    dy = y1 - y2
    dy = dy - Ly * np.round(dy/Ly)

    dz = z1 - z2
    dz = dz - Lz * np.round(dz/Lz)

    dr2 = dx**2 + dy**2 + dz**2
    dr = np.sqrt(dr2)

    return dr
#--------------------------------------------------------------------------
#  Integrator-velocity Verlet
#--------------------------------------------------------------------------
@jit(nopython=True)
def Integration(x,y,z,vx,vy,vz,fx,fy,fz,fxold,fyold,fzold):
    dt2=dt*dt

    # position update
    for i in range(N):
        # the positions are being overridden in each component
        x[i]= x[i] + vx[i]*dt + 0.5*(fxold[i]/m)*(dt2)    
        y[i]= y[i] + vy[i]*dt + 0.5*(fyold[i]/m)*(dt2)   
        z[i]= z[i] + vz[i]*dt + 0.5*(fzold[i]/m)*(dt2)

    pe = Force(x,y,z,fx,fy,fz);    

    # velocity update
    for i in range(N):
        vx[i] = vx[i] + 0.5*((fxold[i]+fx[i])/(m))*dt       # the velocities are being overridden in each component
        vy[i] = vy[i] + 0.5*((fyold[i]+fy[i])/(m))*dt     
        vz[i] = vz[i] + 0.5*((fzold[i]+fz[i])/(m))*dt     
    return pe

#--------------------------------------------------------------------------
#  KE
#--------------------------------------------------------------------------
#This will be used in Temperature control def
#Kinetic energy = (1/2)mv^2
@jit(nopython=True)
def KE(vx,vy,vz):

    cnst = (1/2)*m
    sum = 0.0
    for i in range(N):
        sum = sum + cnst * vx[i] * vx[i]
        sum = sum + cnst * vy[i] * vy[i]
        sum = sum + cnst * vz[i] * vz[i]
    return sum
#--------------------------------------------------------------------------
#Temperature Control -- since the particels are moving the velocities are scaling (T is directly proportional to v^2) if we dont do this we wont know the temperature and cant keep it fixed. It will be an E, V, N simulation
#--------------------------------------------------------------------------
@jit(nopython=True)
def velscaling(vx,vy,vz):
    K=KE(vx,vy,vz) 
#K=(3N-4)(1/2)kbT   (3N-4) comes from degrees of freedom and maikng sure that the plots dont shift
    Tk = (2*K)/(kb*(3*N-4))     #kb is the boltzmann constant

    #simulation for temperature=T
    fact = np.sqrt(T/Tk)      
    for i in range(N):
        vx[i]=vx[i]*fact
        vy[i]=vy[i]*fact
        vz[i]=vz[i]*fact

#--------------------------------------------------------------------------
#Initial Velocity 
#--------------------------------------------------------------------------
def InitVel(): 
    mu = 0
    sigma = np.sqrt(T)
    for i in range(N): 
        temp = random.gauss(mu, sigma)
        temp2 = random.gauss(mu, sigma)
        temp3 = random.gauss(mu, sigma)
        vx[i]=temp
        vy[i]=temp2
        vz[i]=temp3 
        
    sum1 = 0
    sum2 = 0
    sum3 = 0

    for i in range(N):
        sum1 = sum1 + vx[i]
        sum2 = sum2 + vy[i]
        sum3 = sum3 + vz[i]

    vxCoM = sum1/N
    vyCoM = sum2/N
    vzCoM = sum3/N

#subtracting center of Mass"
    for i in range(N):
        vx[i] = vx[i] - vxCoM
        vy[i] = vy[i] - vyCoM
        vz[i] = vz[i] - vzCoM
    
#Calc. Center of Mass"
    sum1 = 0
    sum2 = 0
    sum3 = 0
    for i in range(N):
        sum1 = sum1+ vx[i]
        sum2 = sum2 + vy[i]
        sum3 = sum3 + vz[i]

    vxCoM = sum1/N
    vyCoM = sum2/N
    vzCoM = sum3/N
    print("Velocity of Center of Mass after substraction", vxCoM, vyCoM, vzCoM)
#--------------------------------------------------------------------------
# Copy fx ---> fxold 
#--------------------------------------------------------------------------
@jit(nopython=True)
def copy_fx_to_fxold(fx,fy,fz,fxold,fyold,fzold):
    for i in range(N):
        fxold[i] = fx[i]
        fyold[i] = fy[i]
        fzold[i] = fz[i]
#--------------------------------------------------------------------------
# reading inpiut file 
#--------------------------------------------------------------------------
def read_input():

    global N, rho, L, Lx, Ly, Lz
    global T, rcut, rcutsq, offset, dt
    global iseed, nsteps, minDist

    #infile=sys.argv[1]
    infile = "in.input";
    fp = open(infile, mode = 'r')
    nline = len(fp.readlines())
    fp.close()


    fp = open(infile, mode = 'r')
    for i in range(nline):
        line = fp.readline()
        if (line.strip()):
            a = line.split()[0]
            value = line.split()[1]
            if(a == "NPART"): 
                N = int(value)
                print("N = ", N)
            if(a == "RHO"): 
                rho = float(value)
                print("RHO = ", rho)
                L = np.power(N/rho, 1.0/3.0)
                Lx = L; Ly=L; Lz=L
            if(a == "TEMP"): 
                T = float(value)
                print("TEMP = ", T)
            if(a == "RCUT"): 
                rcut = float(value)
                print("RCUT = ", rcut)
                rcutsq = rcut * rcut
                print("RCUTSQ = ", rcutsq)
                offset = 4.0*epsilon*(sigma12/rcut**12 - sigma6/rcut**6)
                print("offset = ", offset)
            if(a == "DT"): 
                dt = float(value)
                print("DT = ", dt)
            if(a == "SEED"): 
                iseed = int(value)
                print("SEED = ", iseed)
            if(a == "NSTEPS"): 
                nsteps = int(value)
                print("NSTEPS = ", nsteps)
            if(a == "MIN-DIST"): 
                minDist = float(value)
                print("MIN-DIST = ", minDist)
#--------------------------------------------------------------------------
#Tempurature Control Brown-Clarke
#--------------------------------------------------------------------------
@jit(nopython=True)
def TempBC(vx, vy, vz, fx, fy, fz): #TempBC stands for Tempurature Brown-Clarke
    free = (N-1) * 3
    dt_2 = dt / 2.0
    K = 0.0

    for i in range(N):
        vxi = (vx[i] + dt_2 * fx[i]) / m
        vyi = (vy[i] + dt_2 * fy[i]) / m
        vzi = (vz[i] + dt_2 * fz[i]) / m 
        K = K + vxi * vxi + vyi * vyi + vzi * vzi
    
    sysTemp = m * K / free #The system tempurature
    chi = np.sqrt(sysTemp / T)

    for i in range(N):
        vx[i] = (vx[i] * ( 2.0 * chi - 1.0) + chi * dt * fx[i]) / m 
        vy[i] = (vy[i] * ( 2.0 * chi - 1.0) + chi * dt * fy[i]) / m 
        vz[i] = (vz[i] * ( 2.0 * chi - 1.0) + chi * dt * fz[i]) / m 




#=========================================================================
#=========================================================================
#  MAIN function to call all functiona as required
#=========================================================================
#=========================================================================

read_input() # reading input file. 



# Initialization of arrays
fxold = np.zeros(N)
fyold = np.zeros(N)
fzold = np.zeros(N)
vx = np.zeros(N)
vy = np.zeros(N)
vz = np.zeros(N)
x = np.zeros(N)
y = np.zeros(N)
z = np.zeros(N)
fx = np.zeros(N)
fy = np.zeros(N)
fz = np.zeros(N)

# opening energy file
pe_file = "energy.txt"
fp = open(pe_file, mode="w")
fp.write("# istep   pe  t_kin   ke\n")


random.seed(iseed) # to accept seed for random number generator: Must be at the top of MaIn function

InitConf(minDist) #initial position  
InitVel()      #initial velocity

Force(x,y,z,fx,fy,fz) # calling Force first time
copy_fx_to_fxold(fx,fy,fz,fxold,fyold,fzold) # initialization of fxold=fx first time


start = time.time()

# SIMULATION ITERATION STATRS HERE
for istep in range(nsteps):      #We just decide how many steps we want --> made a variable so we can change it in one place
    pe = Integration(x,y,z,vx,vy,vz,fx,fy,fz,fxold,fyold,fzold); 
    copy_fx_to_fxold(fx,fy,fz,fxold,fyold,fzold);
    velscaling(vx,vy,vz);
    TempBC(vx,vy,vz,fx,fy,fz); 

    if(istep%100==0):
        K = KE(vx,vy,vz)
        Tk = (2*K)/(kb*(3*N-4)) 
        fp.write("%s %s %s %s \n"%(istep, pe, Tk, K));
        fp.flush();
        print("istep, pe ", istep, pe )
fp.close()
# SIMULATION ITERATION ENDS HERE

end = time.time()
print("------------------------------")
print("Elapsed time (seconds) = ", end- start)
print("------------------------------")


#--------------------------------------------------------------------------
#   Plotting potential energy vs time
#--------------------------------------------------------------------------
import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv(pe_file, sep='\s+',header=None, skiprows=1)
data = pd.DataFrame(data)

xdata = data[0]
ydata = data[1]
xmax = np.max(xdata)

plt.figure(figsize=(16,12))
plt.plot(xdata, ydata, '-')
plt.xlim([100, xmax])
plt.xlabel('time')
plt.ylabel('potential energy')
plt.show()
#--------------------------------------------------------------------------
