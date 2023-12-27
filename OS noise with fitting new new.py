# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 16:09:15 2020

@author: joseph85811PLUS
"""


# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 19:43:13 2020

@author: joseph85811PLUS
"""


import matplotlib.pyplot as plt
import numpy as np
import math
from scipy import optimize
import time
t_min = 0
t_max = 0.01

dt = 1e-5
eta = 0.8509e-3
R = 0.5e-6
gamma = 6*np.pi*eta*R
#gamma = 1 #friction coefficient
    #n = 3 #noise
N = math.ceil((t_max-t_min)/dt)
A = 15
ks = np.zeros(A)
Tact = np.zeros(A)
fitting_Tact = np.zeros(A)
fk = np.zeros(A)
K = np.linspace(0.01,3.01,A)
Ttt = np.zeros(A)
Ttt1 = np.zeros(A)
Ttt2 = np.zeros(A)
TFK = np.zeros(A)
y = np.zeros(N)


T=300


kB = 1.38064852e-23
noise = np.zeros(N) 

yy=np.zeros(N) 
NN=6000
varyy=np.zeros(NN)
varyy1=np.zeros(A)
varyy11=np.zeros(NN)

def test_func(x, a, b):
    return a * np.exp(-(x)**2/b/2)
ttime = np.arange(t_min, t_max, dt)
#%%
start = time.time()
b=11

d=1e-12

a = 20
std=np.sqrt(2*kB*T*gamma/a/dt)

for ii in range(NN-1):
    rng=np.random.randn()
    if rng>0:
            dd=1
    else:
            dd=-1
    for i  in range(N-1):
            
                r=i%a
                if r == 0 :
                    noise[i+1] = np.random.normal(d*(-1)**int(i/a),std)
                    #noise[i+1] = np.random.normal(0,std)
                
                else :
                
                    noise[i+1] = noise[i]
    for i  in range(N-1): 
        yy[i+1]=yy[i]+noise[i]*dt/gamma
        varyy[ii]=yy[N-1]
        varyy11[ii+1]=gamma*yy[N-1]**2/kB/2/NN+varyy11[ii]
    if ii%10000 ==0:
        print(ii)
    plt.plot(ttime, yy, 'b')
plt.show()
Scale1=1e-7
free_count, bins, ignored = plt.hist(varyy/Scale1, 99, density=True)
N1 = np.linspace(min(varyy),max(varyy),99)

params, params_covariance = optimize.curve_fit(test_func, N1/Scale1, free_count,p0=[2, 2])
print('a=',a)
print("fitting var=",params[1]*Scale1**2)
plt.plot(N1/Scale1, test_func(N1/Scale1, params[0], params[1]))
plt.xlabel('x ($1e-7m$)', fontsize=16)
plt.legend(['Fitted function'])
plt.show()
t=np.var(varyy)*gamma/2/(N*dt)
print(dt)
print(np.var(varyy))
print(varyy11[NN-1])
print("fitting t=",np.var(varyy)*gamma/2/(N*dt))
print("t done")

#%%


for ii in range(A):
    v=0
    y = np.zeros(N)
    ks[ii]=K[ii]/(a*dt/(2*gamma))
    for iii in range(NN-1):
        
    
        for i  in range(N-1):
            rng=np.random.randn()
            if rng>0:
                dd=1
            else:
                dd=-1
                
            r=i%a
            if r == 0 :
                noise[i+1] = np.random.normal(d*(-1)**int(i/a),std)
                    
            else :
                    
                noise[i+1] = noise[i]
        
        for i  in range(N-1):
            y[i+1]=y[i]-ks[ii]*y[i]*dt/gamma+noise[i]*dt/gamma
            
        v=y[N-1]**2/NN+v
        
        #plt.plot(ttime, y, 'b')
        if iii%1000==0:
            print(iii/1000)
    varyy1[ii]=v
    """
    plt.show()
    Scale2=1e-9
    free_count, bins, ignored = plt.hist(varyy1, 99, density=True)
    
    N2 = np.linspace(min(varyy1),max(varyy1),99)
        
    #params, params_covariance = optimize.curve_fit(test_func, (N2)/Scale2, free_count,p0=[2, 2])
        
    #plt.plot((N2)/Scale2, test_func((N2)/Scale2, params[0], params[1]),label='Fitted function')
        #plt.plot(N2/Scale2, 0.5*ks[ii]*(N2/Scale2)**2)
       #plt.xlim(-200,200)
    plt.xlabel('x ($1e-7m$)', fontsize=16)
    plt.title(['ks=',round(ks[ii], 8)/1e-5,'e-5' ])
    plt.show()
    print(ii)   
    """ 
    Tact[ii]=ks[ii]*varyy1[ii]
    #fitting_Tact[ii]=params[1]*Scale2**2*ks[ii]
    #print(np.var(y))
    #print("fitting var=",params[1]*Scale2**2)
    #print(" fitting_Tact=",ks[ii]*params[1]*Scale2**2/kB)
    fk[ii]=Tact[ii]/t
    #K[ii]=ks[ii]*a*dt/(2*gamma)
    #Ttt[ii]=(b*(np.tanh(K[ii]))**2+np.tanh(K[ii]))/K[ii]
    sigma2=2*kB*T*gamma/(a*dt)
    Ttt[ii]=np.tanh(K[ii])*(1+d**2*np.tanh(K[ii])/sigma2)/K[ii]
    #Ttt1[ii]=(2*b*(np.tanh(K[ii]))**2+np.tanh(K[ii]))/K[ii]
    #Ttt2[ii]=(0.5*b*(np.tanh(K[ii]))**2+np.tanh(K[ii]))/K[ii]
    
    
end = time.time()

plt.plot (K , fk, 'o')
plt.ylabel('f$_u$(K)', fontsize=16)
plt.xlabel('K', fontsize=16)
plt.plot (K, Ttt)
#plt.plot (K, Ttt/2)


plt.show()
"""
time = np.arange(t_min, t_max, dt)
plt.plot (time, y/10**-6)
plt.xlabel('t ($s$)', fontsize=16)
plt.ylabel(r'x ($\mu$m)', fontsize=16)
plt.grid(True)
plt.show()


plt.plot (time,noise)
plt.xlabel('t ($s$)', fontsize=16)
plt.ylabel('noise ', fontsize=16)
plt.grid(True)
plt.show()

count,bins, ignored = plt.hist(noise,bins=99,density=True)
plt.xlabel('noise',fontsize=16)
plt.ylabel('probability density',fontsize=16)
plt.show()

count,bins, ignored = plt.hist(y/10**-6,bins=99,density=True)
plt.xlabel('x',fontsize=16)
plt.ylabel('probability density',fontsize=16)
plt.show()


print('var =',np.var(y))

Tact=ks[0]*np.var(y)
print('Simulation Tact =',Tact)

fk=Tact/t/kB
print('f(k) =',fk)

K=ks[0]*a*dt/(2*gamma)
print('K) =',K)

TTT=(b*(np.tanh(K))**2+np.tanh(K))/K
print('Theory Tact =',TTT)
"""
print("用時{}秒".format((end-start)))
print('d =',d)
print('std =',std)
