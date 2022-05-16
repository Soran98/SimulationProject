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

epsilon = 1
sigma = 1

# Simulation parameters
# N = 10000  # number of cells
# rho = 0.8  # density
# T = 1  # tempurature 
# dt = 0.005
# kb = 1      #this is 1.381 * 10**-23 refer to power check
# m = 1  #temporary 
# minDist = 0.9 # the minimum distance each 
# iseed = 10
# nsteps = 20000 #how many steps 



# Box length
# vol = N/rho # volume
# L = np.power(vol, 1 / 3)  # length of the simulation NEED TO CHANGE FOR RECTANGLE
# print(L)
# Lz = L
# Lx = np.sqrt(N/(rho*Lz))
# Ly = Lx
# print("L = ", L, "Lx = ", Lx, "Ly = ", Ly, "Lz = ", Lz)

# #neighbors stuff
# skin = 0.3

# nbins = 20 
# binwidth = Lz / nbins
# Volbin = Lx * Ly * binwidth

# density_sample = 1000
#--------------------------------------------------------------------------
#Tempurate and wall control method
velScale = 0
# bulk = 0

#--------------------------------------------------------------------------
#Lennard-Jones potential parameters
# if bulk == 1: 
#     epsilon_w = 0
# else:
#     epsilon_w = 1
# epsilon = 1
# sigma = 1
# rcut = 2.5
# # sigma12 = sigma ** 12
# # sigma6 = sigma ** 6
# rcut2 = rcut * rcut
# rcutsq = rcut2
# offset = 4.0 * epsilon* (1 / rcut**12 - 1 / rcut**6)
# print("LJ parameters: sigma=%s epsilon=%s rcut=%s offset=%s "%(sigma, epsilon, rcut, offset))
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
            sigmaij = 0.5*(sigmaSizes[i] + sigmaSizes[j])
            sigmaij2 = sigmaij * sigmaij
            sigmaij4 = sigmaij2 * sigmaij2
            sigmaij6 = sigmaij4 * sigmaij2
            sigmaij12 = sigmaij6 * sigmaij2 
            

            dx = x[i] - x[j]
            dx = dx - Lx * np.round(dx/Lx) # minimum image distance

            dy = y[i] - y[j]
            dy = dy - Ly * np.round(dy/Ly)

            dz = z[i] - z[j]
            if bulk == 1:
                dz = dz - Lz * np.round(dz/Lz) #for bulk simulation (if wall is off)
            
            dr2 = dx**2 + dy**2 + dz**2
            #print("dr2:", dr2, "dx:", dx, "dy:", dy, "dz:", dz, "sigmaij2", sigmaij2)
            #inv = inverse
            
            if dr2 < rcutsq * sigmaij2:
                dr2inv = 1.0 / dr2
                dr6inv = dr2inv * dr2inv * dr2inv
                dr12inv = dr6inv * dr6inv

                du = 4 * epsilon * ((sigmaij12 * dr12inv) - (sigmaij6 * dr6inv)) - offset
                u = u + du 

                wij = 4 * epsilon * (12 * sigmaij12 * dr12inv - 6 * sigmaij6 * dr6inv)
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
@jit(nopython=True)
def x_rand():
    r1 = random.uniform(0, Lx)
    r2 = random.uniform(0, Ly)
    r3 = random.uniform(0, Lz)
    return r1, r2, r3
#--------------------------------------------------------------------------
#  function to generate randomc ocnfiguration s.t. distance between 2 particles > minDist
#--------------------------------------------------------------------------
@jit(nopython=True)
def InitConf(minDist, sigmaSizes, x, y, z):
    x[0], y[0], z[0] = x_rand() # choose an arbitrary position for the very 1st particle
    i = 0
    while i < N:
            x_t, y_t, z_t = x_rand() # trial position
            iflag = 1 # flag for accepting trial position in x, y, z list if dist > minDist
            for j in range(i): # look for all possible pairs
                sigmaij = 0.5*(sigmaSizes[i] + sigmaSizes[j])
                sigmaij2 = sigmaij * sigmaij
                dx = x[j] - x_t
                dx = dx - Lx * np.round(dx/Lx) # minimun image distance

                dy = y[j] - y_t
                dy = dy - Ly * np.round(dy/Ly)

                dz = z[j] - z_t
                if bulk == 1:
                    dz = dz - Lz * np.round(dz/Lz)

                dr2 = dx**2 + dy**2 + dz**2
                if(dr2/sigmaij2 < minDist*minDist):
                    iflag = 0 # iflag=0 means don't accept the trial position: see later lines
                    break
            if(iflag==1): # this line will reach (i) by above break statement or (ii) after finishing above for loop
                x[i] = x_t; y[i] = y_t; z[i] = z_t; i = i + 1
                print("Particle", i, "generated")
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
    pe_wall = 0.0
    if bulk == 0:
        pe_wall = force_wall(x,y,z,fx,fy,fz);   

    peTotal = pe + pe_wall

    # velocity update
    for i in range(N):
        vx[i] = vx[i] + 0.5*((fxold[i]+fx[i])/(m))*dt       # the velocities are being overridden in each component
        vy[i] = vy[i] + 0.5*((fyold[i]+fy[i])/(m))*dt     
        vz[i] = vz[i] + 0.5*((fzold[i]+fz[i])/(m))*dt     
    return peTotal

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
#K=(3N-4)(1/2)kbT   (3N-4) comes from degrees of freedom and maing sure that the plots dont shift
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
# Build the wall
#--------------------------------------------------------------------------
@jit (nopython=True)
def force_wall(x, y, z, fx, fy, fz):
    u = 0.0
    for i in range(N):
        if wall_neutrality == 0:
            sigma = sigmaSizes[i]
        else:
            sigma = 1
        #left wall
        dz = z[i] - Lzwall
        #print(dz, z[i], Lzwall)
        #exit()
        du = epsilon_w * (sigma/abs(dz)) ** 9
        u = u + du
        wij = 9 * epsilon_w * (sigma/abs(dz)) ** 9
        wij = wij / (dz ** 2)
        fz[i] = fz[i] + wij * dz
   
    for i in range(N):
        if wall_neutrality == 0:
            sigma = sigmaSizes[i]
        else:
            sigma = 1
        #right wall
        dz = z[i] - Rzwall
        du = epsilon_w * (sigma/abs(dz)) ** 9
        u = u + du
        wij = 9 * epsilon_w * (sigma/abs(dz)) ** 9
        wij = wij / (dz ** 2)
        fz[i] = fz[i] + wij * dz 
    
    return u/N

#--------------------------------------------------------------------------
# reading inpiut file 
#--------------------------------------------------------------------------
def read_input():

    global N, rho, L, Lx, Ly, Lz
    global T, rcut, rcutsq, offset, dt
    global iseed, nsteps, minDist
    global sigMin, sigMax, bulk, Lz
    global wall_neutrality, eql_steps
    global binwidth, graphsCheck

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
                # L = np.power(N/rho, 1.0/3.0)
                # Lx = L; Ly=L; Lz=L
            if(a == "TEMP"): 
                T = float(value)
                print("TEMP = ", T)
            if(a == "RCUT"): 
                rcut = float(value)
                print("RCUT = ", rcut)
                rcutsq = rcut * rcut
                print("RCUTSQ = ", rcutsq)
                offset = 4.0*epsilon*(1/rcut**12 - 1/rcut**6)
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
            if(a == "sigMin"): 
                sigMin = float(value)
                print("sigMin = ", sigMin)
            if(a == "sigMax"): 
                sigMax = float(value)
                print("sigMax = ", sigMax)
            if(a == "bulk"): 
                bulk = int(value)
                print("bulk = ", bulk)
            if(a == "Lz"): 
                Lz = float(value)
                print("Lz = ", Lz)
            if(a == "wall_neutrality"): 
                wall_neutrality = int(value)
                print("wall_neutrality = ", wall_neutrality)
            if(a == "eql_steps"): 
                eql_steps = int(value)
                print("eql_steps = ", eql_steps)
            if(a == "binwidth"): 
                binwidth = float(value)
                print("binwidth = ", binwidth)
            if(a == "graphsCheck"): 
                graphsCheck = int(value)
                print("Graphs = ", graphsCheck)
#--------------------------------------------------------------------------
#Tempurature Control Brown-Clarke
#--------------------------------------------------------------------------
@jit(nopython=True)
def TempBC(x, y, z, vx, vy, vz, fx, fy, fz): #TempBC stands for Tempurature Brown-Clarke
    free = (N-1) * 3
    dt_2 = dt / 2.0
    K = 0.0
    #print("vx:", vx, "fx:", fx)
    #print("free:", free,"dt_2:", dt_2)
    for i in range(N):
       # print("vx:", vx[i], "fx:", fx[i])
        vxi = vx[i] + dt_2 * fx[i] / m
        vyi = vy[i] + dt_2 * fy[i] / m
        vzi = vz[i] + dt_2 * fz[i] / m 
        #print("vxi:",vxi,"vyi:",vyi,"vzi:",vzi)
        K = K + ((vxi * vxi) + (vyi * vyi) + (vzi * vzi))
    
    sysTemp = m * K / free #The system tempurature
    chi = np.sqrt(T / sysTemp)
    #print("sysTemp:", sysTemp, "chi:", chi, "K:", K)

    for i in range(N):
        vx[i] = vx[i] * ( 2.0 * chi - 1.0) + chi * dt * fx[i] / m 
        vy[i] = vy[i] * ( 2.0 * chi - 1.0) + chi * dt * fy[i] / m 
        vz[i] = vz[i] * ( 2.0 * chi - 1.0) + chi * dt * fz[i] / m 

    #updating postions with new velocity
    for i in range(N):
        x[i] = x[i] + dt * vx[i]
        y[i] = y[i] + dt * vy[i]
        z[i] = z[i] + dt * vz[i]

    pe = Force(x,y,z,fx,fy,fz); 
    pe_wall = 0.0
    if bulk == 0:
        pe_wall = force_wall(x,y,z,fx,fy,fz);   

    peTotal = pe + pe_wall

    return peTotal
        
#--------------------------------------------------------------------------
#Polydispersity 
#--------------------------------------------------------------------------

@jit(nopython=True)
def randomSigma(sigmaSizes):
    for i in range(N):
        sigmaSizes[i] = random.uniform(sigMin, sigMax)

#--------------------------------------------------------------------------
# Creating Neighbors List  
#--------------------------------------------------------------------------
@jit(nopython=True)
def nList(): 

    
    nonList = []             #number of neighbors 
    for i in range(N):
        for j in range(i+1, N):
            dx =x[i] - x[j]
            dx = dx - L * round(dx/2)

            dy = y[i] - y[j]
            dy = dy - L * round(dy/2)

            dz = z[i] - z[j]
            dz = dz - L * round(dz/2)

            dr2 = (dx*dx) + (dy*dy) + (dz*dz)
            if dr2 < ((rcut + skin)(rcut + skin))/ (rcut2):
                nonList[i] = nonList[i] + 1
                nonList[j] = nonList[j] + 1
                nlist[nonList[i], i] = j
                nlist[nonList[j], j] = i

#--------------------------------------------------------------------------
# Checking the Neighbor List and the positions 
#--------------------------------------------------------------------------  
@jit(nopython=True)  
def checkList():
    displ = 0
    for i in range(N):
        dx = x[i]-xold[i]
        displ = max(displ, abs(dx))

        dy = y[i] - yold[i]
        displ = max(displ, abs(dy))

        dz = z[i] - zold[i]
        displ = max(displ, abs(dz))

        if displ > (skin/2):        #make skin rcut, becuase this is what 
            nList()

#--------------------------------------------------------------------------
#Check if Z molecule left range of wall
#--------------------------------------------------------------------------
def boundaryZcheck(z):
    for i in range(N):
        if z[i] >   Rzwall:  # for right wall
            print("Z axis molecule went past the right wall")
            exit()
       
        if z[i] <  Lzwall:
            print("Z axis molecule went past the left wall")
            exit()


#--------------------------------------------------------------------------
#Density Modulation
#--------------------------------------------------------------------------
@jit (nopython=True)
def density_mod(z, numParticle):
    for i in range(N):
        if bulk == 1:
            z1 = z[i] - Lz * round(z[i]/Lz - 0.5)
        else:
            z1 = z[i] - Lzwall
        
        ibin = int(z1 / binwidth) #the number of bins
        #if ibin > 19:
        #    print(ibin, z[i], z1)
        #print("i: ", i, "ibin:", ibin, "z", z[i], "x", x[i], "y", y[i], "L: ", L, "Lx:", Lx, "Ly", Ly)
        numParticle[ibin] = numParticle[ibin] + 1 #the number of particles in a bin
        
        #for j in range(ibin):
        #for j in range(N):
        #   numParticle[j] = numParticle[j] + 1

#=========================================================================
#=========================================================================
#  MAIN fnction to call all functiona as required
#=========================================================================
#=========================================================================

read_input() # reading input file. 

if bulk == 1: 
    epsilon_w = 0
else:
    epsilon_w = 1
#rcut = 2.5
# sigma12 = sigma ** 12
# sigma6 = sigma ** 6
rcut2 = rcut * rcut
rcutsq = rcut2
#offset = 4.0 * epsilon* (1 / rcut**12 - 1 / rcut**6)
print("LJ parameters: sigma=%s epsilon=%s rcut=%s offset=%s "%(sigma, epsilon, rcut, offset))

vol = N/rho # volume

if bulk == 1:
    L = np.power(vol, 1 / 3)  # length of the simulation NEED TO CHANGE FOR RECTANGLE
    Lx = L
    Ly = L
    Lz = L
else: 
    Lz = Lz
    Lx = np.sqrt(N/(rho*Lz))
    Ly = Lx
print("Lx = ", Lx, "Ly = ", Ly, "Lz = ", Lz)
#make if statement to end program if Lx and Ly are smaller than rcut*2

Lzwall = -.5
Rzwall = Lz + .5
m = 1
kb = 1

# sigMin = 0.8
# sigMax = 1.2

#neighbors stuff
skin = 0.3

#binwidth = Lz / nbins #add Rzwall and Lzwall to Lz
if bulk == 1:
    nbins = int(Lz / binwidth) + 1
else:
    nbins = int((Rzwall - Lzwall) / binwidth) + 1

Volbin = Lx * Ly * binwidth
totalLength = binwidth * nbins

density_sample = 1000

if Lx < rcut * sigMax * 2 or Ly < rcut * sigMax * 2 or Lz < rcut * sigMax * 2:
    print("Lx or Ly or Lz were less than rcut * sigMax * 2")
    exit()


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
numParticle = np.zeros(nbins)
densities_final_list = np.zeros(N)
sigmaSizes = np.zeros(N)
sigma12 = np.zeros(N)
sigma6 = np.zeros(N)
#offset = np.zeros(N)
nlist = np.zeros(N)

count = 0
 
# opening energy file
file_tag = "N%s-rho%s-T%s-sigMin%s-sigMax%s-bulk%s-Lz%s-binwidth%s"%(N, rho, T, sigMin, sigMax, bulk, Lz, binwidth)
pe_file = file_tag + "energy.txt"
fp = open(pe_file, mode="w")
fp.write("# istep   pe  t_kin   ke\n")

#opening density file
density_file = file_tag + "density.txt"
fp1 = open(density_file, mode="w")
#print("L: ", L, "Lx:", Lx, "Ly", Ly)

random.seed(iseed) # to accept seed for random number generator: Must be at the top of MaIn function

print("Calling randomSigma")
randomSigma(sigmaSizes)


print("Calling InitConf")
InitConf(minDist, sigmaSizes, x, y, z) #initial position  
# plt.figure()
# plt.scatter(x,y)
# plt.scatter(x,z)
# plt.show()
print("Calling InitVel")
InitVel()      #initial velocity

print("Calling Force")
Force(x,y,z,fx,fy,fz) # calling Force first time


if bulk == 0:
    print("Calling force_wall")
    force_wall(x,y,z,fx,fy,fz)

print("Calling copy_fx_to_fxold")
copy_fx_to_fxold(fx,fy,fz,fxold,fyold,fzold) # initialization of fxold=fx first time

start = time.time()

# SIMULATION ITERATION STATRS HERE

#=================================================================
# Pre-equilibration via velocity scaling, otherwise, TempBC oscillates between a high and a low temperature for large dt such as dt=0.005
for istep in range(5000):      #We just decide how many steps we want --> made a variable so we can change it in one place
    pe = Integration(x,y,z,vx,vy,vz,fx,fy,fz,fxold,fyold,fzold); 
    copy_fx_to_fxold(fx,fy,fz,fxold,fyold,fzold);
    velscaling(vx,vy,vz);
    if bulk == 0:
        boundaryZcheck(z);

    if(istep%100==0):
        K = KE(vx,vy,vz)
        Tk = (2*K)/(kb*(3*N-4)) 
        print("Pre-Equilibration istep, pe, temp ", istep, pe , Tk)
#=================================================================


# SIMULATION ITERATION STATRS HERE
for istep in range(nsteps):      #We just decide how many steps we want --> made a variable so we can change it in one place
    if velScale == 1:
        pe = Integration(x,y,z,vx,vy,vz,fx,fy,fz,fxold,fyold,fzold); 
        copy_fx_to_fxold(fx,fy,fz,fxold,fyold,fzold);
        velscaling(vx,vy,vz);
    else:
    #print("vx:", vx[istep], "fx:", fx[istep])
        pe = TempBC(x,y,z,vx,vy,vz,fx,fy,fz); 
#=================================================================
        if bulk == 0:
            boundaryZcheck(z);
#=================================================================








# for istep in range(nsteps):      #We just decide how many steps we want --> made a variable so we can change it in one place
#     if velScale == 1:
#         pe = Integration(x,y,z,vx,vy,vz,fx,fy,fz,fxold,fyold,fzold); 
#         copy_fx_to_fxold(fx,fy,fz,fxold,fyold,fzold);
#         velscaling(vx,vy,vz);
#     else:
#     #print("vx:", vx[istep], "fx:", fx[istep])
#         pe = TempBC(x,y,z,vx,vy,vz,fx,fy,fz); 
    
#     if bulk == 0:
#         boundaryZcheck(z)

    if istep > eql_steps:
        if istep % density_sample == 0:
            count = count + 1
            density_mod(z, numParticle)
            for islab in range(nbins):
                densities_final_list[islab] = numParticle[islab]/Volbin
                
            
   # exit()

    if(istep%100==0):
        K = KE(vx,vy,vz)
        Tk = (2*K)/(kb*(3*N-4)) 
        fp.write("%s %s %s %s \n"%(istep, pe, Tk, K));
        fp.flush();
        print("istep, pe ", istep, pe )
fp.close()
# SIMULATION ITERATION ENDS HERE


# x2 = np.zeros(N)
# y2 = np.zeros(N)
# ax = plt.axes(projection='3d')
# plt.figure()
# for i in range(N):
#     x2[i] = x[i] - Lx * round(x[i]/Lx - 0.5)
#     y2[i] = y[i] - Ly * round(y[i]/Ly - 0.5)
# ax.scatter3D(z, y2, x2, c=x2);
# ax.set_xlabel("z")
# ax.set_ylabel("y")
# ax.set_zlabel("x")
# plt.show()
#exit()

end = time.time()
print("------------------------------")
print("Elapsed time (seconds) = ", end- start)
print("------------------------------")


for islab in range(nbins):
    fp1.write("%s %s \n"%((islab * binwidth + (binwidth/2)), densities_final_list[islab]/count))
    fp1.flush()
fp1.close()
#--------------------------------------------------------------------------
#   Plotting potential energy vs time
#--------------------------------------------------------------------------
import pandas as pd
import matplotlib.pyplot as plt
if graphsCheck == 1:    
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
#   Plotting potential energy vs time
#--------------------------------------------------------------------------
if graphsCheck == 1:
    data1 = pd.read_csv(density_file, sep='\s+',header=None, skiprows=1)
    x1 = data1[0]
    y1 = data1[1]
    plt.plot (x1, y1, '-o')
    plt.xlabel("Distance")
    plt.ylabel("Density")
    plt.show()