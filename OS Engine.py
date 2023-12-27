# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 12:09:17 2020

@author: USER
"""

import matplotlib.pyplot as plt
import numpy as np
import math
Wab = 0
Wbc = 0
Wcd = 0
Wda = 0    
Qab = 0
Qbc = 0
Qcd = 0
Qda = 0

NQbc = 0
NQcd = 0
NWbc = 0
NWcd = 0

NNQbc = 0
NNQcd = 0
NNWbc = 0
NNWcd = 0
import time
C = 300
t_min = 0
t_max = 0.08
dt = 1e-5 #5e-6
eta = 0.8509e-3
R = 0.5e-6
r = 6*np.pi*eta*R #friction coefficient
    #n = 3 #noise
#gamma = 6*np.pi*eta*R
gamma = 1e-8
N = math.ceil((t_max-t_min)/dt)
y = np.zeros(N)
Y = np.zeros(C)
T0 = 300
#2kbt/
dG = 2e-4
XX2=np.zeros(N)
XX=np.zeros(N)
a = dG/dt

print(a)
go_R=0
go_L=0
noise = np.zeros(N) 
start = time.time()
q=np.zeros(N) 
w=np.zeros(N) 
Q=np.zeros(N) 
W=np.zeros(N) 
P=np.zeros(C)

EEE=np.zeros(C)
for n in range(C):
   
    
    y[0] = y[N-1]
    """
    if n ==0 :
         y[0]=1e-8
    """
    t1 = 200 #assume temperature #400
    t2 = 400 #1000
    kB = 1.38064852e-23
    ks1 = 1e-5 #stiffness #2e-5
    ks2 = 4e-5 #8e-5
    oua1 = 1/(0.02*0.001) #Correlation time 1/ms
    oua2 = 1/(0.06*0.001)
    t = np.zeros(N)
    t[0] = t1
    ks = np.zeros(N)
    ks[0] = ks1
    #T = np.zeros(N) #real temperature
    #Q = np.zeros(N) #heat
    #W = np.zeros(N) #work
    #oua = np.zeros(N)
    
    

    
    wab = 0
    wbc = 0
    wcd = 0
    wda = 0
    qab = 0
    qbc = 0
    qcd = 0
    qda = 0
    
    qab1 = 0
    qab2 = 0
    b=11
    
    Nqbc=0
    Nqcd=0
    Nwbc = 0
    Nwcd = 0
    
    NNqbc=0
    NNqcd=0
    NNwbc = 0
    NNwcd = 0
    
    for i in range(N-1):
            if i < (N/4):
                ks[i+1] = ks[i]+(ks2-ks1)/((N)/4)
            elif (N/4) <= i < (N/2):
                ks[i+1] = ks[i]
            elif (N/2)<= i < (N*3/4):
                ks[i+1]=ks[i]-(ks2-ks1)/((N)/4)
            elif (N*3/4) <= i < (N):
                ks[i+1] = ks[i]
    for i in range(N-1):
            if i < (N/4):
                t[i+1] = t[i]
            elif (N/4) <= i < (N/2):
                t[i+1] = t[i]+(t2-t1)/((N)/4)
            elif (N/2) <= i < (N*3/4):
                t[i+1]= t[i]
            elif (N*3/4) <= i < (N):
                t[i+1] = t[i]-(t2-t1)/((N)/4)
            
   
    
        
    rng=np.random.randn()
    if rng>0:
            dd=1
            go_R=go_R+1
            #phase = 0
    else:
            dd=-1
            go_L=go_L+1
            #phase = 10
    #phase = np.random.randint(20)
    #phase = 17
    #phase = n%(20)
    phase = 19
    
    
    P[n]=phase
    for i  in range(N-1):
        d=1e-4*gamma
        #std=np.sqrt(2*kB*t[i]/dG/gamma)
        std=np.sqrt(2*kB*t[i]*gamma/dG)
        #std=np.sqrt(2*kB*t[i]*dG)
        
        
        
        
        
            
        r=i%a
        if r == phase :
            #noise[i+1] = d*(-1)**int(i/a)
            noise[i+1] = np.random.normal(d*(-1)**int(i/a),std)
        
        else :

                    noise[i+1] = noise[i]
    
    for i in range(phase+1):
        noise[i]=noise[N-1]
    if phase == (a-1):
        newstep=np.random.normal(d*(-1),std)
        #newstep=d*(-1)
        for i in range(phase+1):
            noise[i]=newstep
        
        
    for i  in range(N-1): 
            #y[i+1]=y[i]+noise[i]*dt-ks[i]*y[i]*dt/gamma
            y[i+1]=y[i]+noise[i]*dt/gamma-ks[i]*y[i]*dt/gamma
            r=i%a
    for i  in range(N):        
            XX2[i]=y[i]**2
    
            
    XX=XX+XX2/C
    
    
    
    Ni=20+phase
    """
    for i in range(int(N/4),N-1):
        if y[i]*y[i+1]<=0 :
            a0=i
            break
    for i in range(int(N/4),N-1):
        if y[i]*y[i+1]<=0 and i<=N*3/4:
            b0=i
            
    
            
        
            
    
    
    
    for i in range(N-1):
        if (a0) <= i < (N/2):
            NNqbc = NNqbc+(y[i+1]-y[i])*(ks[i+1]*y[i+1]+ks[i]*y[i])/2
            NNwbc = NNwbc+(ks[i+1]-ks[i])*y[i]**2/2
        elif (N/2) <= i < (b0):
            NNqcd = NNqcd+(y[i+1]-y[i])*(ks[i+1]*y[i+1]+ks[i]*y[i])/2
            NNwcd = NNwcd+(ks[i+1]-ks[i])*y[i]**2/2
    
    
    
    NNQbc = NNQbc + NNqbc
    NNQcd = NNQcd + NNqcd
    NNWbc = NNWbc + NNwbc
    NNWcd = NNWcd + NNwcd
    """     
    
    for i  in range(N-1):
        
        if (N/4+Ni) <= i < (N/2):
            Nqbc = Nqbc+(y[i+1]-y[i])*(ks[i+1]*y[i+1]+ks[i]*y[i])/2
            Nwbc = Nwbc+(ks[i+1]-ks[i])*y[i]**2/2
        
        elif (N/2) <= i < (N*3/4):
            Nqcd = Nqcd+(y[i+1]-y[i])*(ks[i+1]*y[i+1]+ks[i]*y[i])/2
            Nwcd = Nwcd+(ks[i+1]-ks[i])*y[i]**2/2
        
    NQbc = NQbc + Nqbc
    NQcd = NQcd + Nqcd
    NWbc = NWbc + Nwbc
    NWcd = NWcd + Nwcd
    

    
            
    
       
    for i  in range(N-1):   
            if i < (N/4):
                qab = qab+(y[i+1]-y[i])*(ks[i+1]*y[i+1]+ks[i]*y[i])/2
                #qab = qab+((y[i+1]-y[i])/6)*(ks[i]*y[i]+ks[i+1]*y[i+1]+4*(ks[i+1]*y[i+1]+ks[i]*y[i])/2)
                wab = wab+(ks[i+1]-ks[i])*y[i]**2/2
                q[i+1]=qab
                w[i+1]=wab
            
            #elif (N/4) <= i < (N/2):
            elif (N/4) <= i < (N/2):
                #qbc = qbc+(y[i+1]-y[i])*(ks[i+1]*y[i+1]+ks[i]*y[i])/2
                qbc = qbc+((y[i+1]-y[i])/6)*(ks[i]*y[i]+ks[i+1]*y[i+1]+4*(ks[i+1]*y[i+1]+ks[i]*y[i])/2)
                wbc = wbc+(ks[i+1]-ks[i])*y[i]**2/2
                q[i+1]=qbc+qab
                w[i+1]=wbc+wab
            #elif (N/2) <= i < (N*3/4):
            elif (N/2) <= i < (N*3/4):
                qcd = qcd+(y[i+1]-y[i])*(ks[i+1]*y[i+1]+ks[i]*y[i])/2
                wcd = wcd+(ks[i+1]-ks[i])*y[i]**2/2
                q[i+1]=qcd+qbc+qab
                w[i+1]=wcd+wbc+wab
            elif (N*3/4) <= i < (N):
                qda = qda+(y[i+1]-y[i])*(ks[i+1]*y[i+1]+ks[i]*y[i])/2
                wda = wda+(ks[i+1]-ks[i])*y[i]**2/2
                q[i+1]=qda+qcd+qbc+qab
                w[i+1]=wda+wcd+wbc+wab
    Q = Q+q/(C)
    W = W+w/(C) 
            
    Wab = Wab + wab
    Wbc = Wbc + wbc
    Wcd = Wcd + wcd
    Wda = Wda + wda
    Qab = Qab + qab
    Qbc = Qbc + qbc
    Qcd = Qcd + qcd
    Qda = Qda + qda
    
    
    
    
    
    

    if (n+1)%(C/10) ==0:
        
        print(n+1)
        """
    if n==0:
        print(y[0],y[1],y[2],y[3],y[4])
        print(ks[0]*y[0],ks[1]*y[1],ks[2]*y[2],ks[3]*y[3],ks[4]*y[4])
        print(noise[0],noise[1],noise[2],noise[3],noise[4])
        print(ks[0],ks[1],ks[2],ks[3],ks[4])

        """
   
    





#st=(np.log(ks2/ks1)-np.log((1+w1*tc)/(1+w2*tc)))/(1/(1+w2*tc)+np.log(ks2/ks1)-np.log((1+w1*tc)/(1+w2*tc)))

#%%
"""
Qmid=np.zeros(200)
for i in range(200):
    Qmid[i]=(np.max(Q[40*i:40*i+40])+np.min(Q[40*i:40*i+40]))/2
""" 



"""   
tick=np.arange(0,200)
plt.plot(tick, noise[0:200])
plt.show()

plt.plot(np.arange(int(N/4+Ni),int(N*3/4-Ni)), y[int(N/4+Ni):int(N*3/4-Ni)])
plt.show()

plt.hist(Y, 99, density=True)
plt.show()
"""
ttime = np.arange(t_min, t_max, dt)
ttime1 = np.arange(t_min, t_max, dt*40)
#%%
Ax1 = plt.figure().add_subplot(111)

plt.ticklabel_format(style='sci',scilimits=(-1,2), axis='both')   
a,=Ax1.plot(ttime,y,'r',label='X') 
 
Ax1.set_xlabel('time ($s$)',fontsize=16)


plt.ylabel('x', fontsize=16)


Ax2 = Ax1.twinx()
b,=Ax2.plot(ttime,Q,'b',label='Q')
plt.ylabel('Q', fontsize=16)

plt.legend(handles=[a,b],loc='best')
plt.show()
#%%

print('t2-t1 =',t2-t1)
ttime = np.arange(t_min, t_max, dt)
ttime1 = np.arange(t_min, t_max, dt*40)
plt.plot(ttime, y, 'r')
plt.plot(ttime, Q/(1e-13))
#plt.plot(ttime[1], y[1], 'ro')
#plt.annotate((ttime[1],y[1]),xy=(ttime[1],y[1]) )

plt.xlabel('t ($s$)', fontsize=16)
plt.ylabel(r'x ', fontsize=16)
plt.grid(True)
plt.show()





print(Qbc,NQbc,NNQbc)

print(Qcd,NQcd,NNQcd)



print('efficiency =',-(Wab+Wbc+Wcd+Wda)/(Qcd+Qbc))
print('Nefficiency =',-(Wab+NWbc+NWcd+Wda)/(NQcd+NQbc))
#print('00Nefficiency =',-(Wab+NNWbc+NNWcd+Wda)/(NNQcd+NNQbc))

print('pasive efficiency =',((t2-t1)*np.log(ks2/ks1)/((t2-t1)+(t2)*np.log(ks2/ks1))))
print(-(Wab+Wbc+Wcd+Wda)/(Qcd+Qbc)/((t2-t1)*np.log(ks2/ks1)/((t2-t1)+(t2)*np.log(ks2/ks1))))


"""
print('Qab =',Qab/kB/T0/C)
print('theoretical Qab =',qtab/(kB*T0))
print('Qbc =',Qbc/kB/T0/C)
print('theoretical Qbc =',qtbc/kB/T0)
print('Qcd =',Qcd/kB/T0/C)
print('theoretical Qcd =',qtcd/(kB*T0))
print('Qda =',Qda/kB/T0/C)
print('theoretical Qda =',qtda/kB/T0)
print('Wab =',Wab/kB/T0/C)
print('theoretical Wab =',(wtab)/kB/T0)
print('Wbc =',Wbc/kB/T0/C)
print('theoretical Wbc =',wtbc)
print('Wcd =',Wcd/kB/T0/C)
print('theoretical Wcd =',wtcd/kB/T0)
print('Wda =',Wda/kB/T0/C)
print('theoretical Wda =',wtda)
"""
#%%
confine_count, bins, ignored = plt.hist(P, 20, density=True)
print(np.average(P))

plt.show()
#%%

Ax1 = plt.figure().add_subplot(111)
plt.title("(a)",fontsize=20,loc='right')
plt.ticklabel_format(style='sci',scilimits=(-1,2), axis='both')   
a,=Ax1.plot(ttime,noise,'g') 
Ax1.set_xlabel('time ($s$)',fontsize=16)
plt.ylabel('noise', fontsize=16)
Ax1.set_title('Cycle',fontsize=16)

Ax2 = Ax1.twinx()
b,=Ax2.plot(ttime,t,'r',label='T')

plt.yticks([])
Ax3 = Ax2.twinx()
c,=Ax3.plot(ttime,ks/1e-5,label='K')
plt.yticks([])
plt.legend(handles=[c,b],loc='best')
plt.show()
#%%


#%%
plt.ticklabel_format(style='sci',scilimits=(-1,2), axis='both')
plt.plot(ttime, XX, 'b')
#plt.plot(ttime, (d**2*(np.tanh(ks*dG/gamma/2))**2+(2*kB*t/gamma/dG)*np.tanh(ks*dG/gamma/2))*(gamma/ks)**2, 'r')
plt.plot(ttime, (d**2*(np.tanh(ks*dG/gamma/2))**2+(2*kB*t*gamma/dG)*np.tanh(ks*dG/gamma/2))*(1/ks)**2, 'r')
plt.grid(True)
plt.ylabel(r'varience ', fontsize=16)
plt.xlabel('time($s$)', fontsize=16)
plt.show()
#%%

plt.ticklabel_format(style='sci',scilimits=(-1,2), axis='both')
plt.plot(ttime, Q, 'b')
plt.title("(c)",fontsize=20)
#plt.plot(ttime1, Qmid, 'r')
plt.ylabel('Q', fontsize=16)
plt.xlabel('time($s$)', fontsize=16)
plt.grid(True)
plt.show()

plt.ticklabel_format(style='sci',scilimits=(-1,2), axis='both')
plt.plot(ttime, W, 'b')
plt.title("(d)",fontsize=20)
plt.ylabel('W', fontsize=16)
plt.xlabel('time($s$)', fontsize=16)
plt.grid(True)
plt.show()

print("K=",dG*ks1/2/gamma)

print("K=",dG*ks2/2/gamma)

end = time.time()
print("用時{}秒".format((end-start)))
print("走+方向{}次".format(go_R))
print("走-方向{}次".format(go_L))
K=np.arange(0.0001, 5, 0.1)
KK=np.arange(ks1*dG/gamma/2, ks2*dG/gamma/2, 0.1)
#sigma1=2*kB*t1/dG/gamma
#sigma2=2*kB*t2/dG/gamma

sigma1=2*kB*t1*gamma/dG
sigma2=2*kB*t2*gamma/dG

#%%
plt.title("(b)",fontsize=20)
plt.ylabel('f(K)', fontsize=16)
plt.xlabel('K', fontsize=16)
plt.plot(K, np.tanh(K)*(1+d**2*np.tanh(K)/sigma1)/K,'b')
plt.plot(K, np.tanh(K)*(1+d**2*np.tanh(K)/sigma2)/K)
plt.plot(KK, np.tanh(KK)*(1+d**2*np.tanh(KK)/sigma1)/KK,'r')
plt.plot(KK, np.tanh(KK)*(1+d**2*np.tanh(KK)/sigma2)/KK,'r')
plt.grid(True)

plt.show()

#%%
kb=format(kB,'.3E')
Sigma1=format(sigma1,'.3E')
Sigma2=format(sigma2,'.3E')
Gamma=format(gamma,'.3E')
D=format(d,'.3E')
KS1=format(ks1,'.3E')
KS2=format(ks2,'.3E')
T1=format(t1,'.3E')
T2=format(t2,'.3E')
span=format(a)

Step=format(t_max/dG/4)
cycle=format(C)
t_span=format(dt)
t_step=format(dG)
t_cycle=format(t_max)
t_relax1=format(gamma/ks1,'.3E')
t_relax2=format(gamma/ks2,'.3E')
t_t1=format(dt/(gamma/ks1),'.3E')
t_t2=format(dt/(gamma/ks2),'.3E')
K1=format(ks1*dG/2/gamma,'.3E')
K2=format(ks2*dG/2/gamma,'.3E')
TT1=format(gamma/ks1/t_max,'.3E')
TT2=format(gamma/ks2/t_max,'.3E')
eff_act=format(-(Wab+Wbc+Wcd+Wda)/(Qcd+Qbc),'.6E')
eff_pas=format(((t2-t1)*np.log(ks2/ks1)/((t2-t1)+(t2)*np.log(ks2/ks1))),'.6E')

#%%

plt.plot()
plt.yticks([-0.3,1])
plt.xticks([0,1])
plt.text(0, 0.9,"kB={}".format(kb),fontsize=14)
plt.text(0, 0.8,"$\gamma$={}".format(Gamma),fontsize=14)
plt.text(0.4, 0.8,"$\\xi$$_0$={}".format(D),fontsize=14)
plt.text(0, 0.7,"$\sigma$=[{}".format(Sigma1),fontsize=14)
plt.text(0.32, 0.7,",{}]".format(Sigma2),fontsize=14)
plt.text(0.0, 0.6,"k$_1$={}".format(ks1),fontsize=14)
plt.text(0.25, 0.6,"k$_2$={}".format(ks2),fontsize=14)
plt.text(0.0, 0.5,"T$_1$={}".format(t1),fontsize=14)
plt.text(0.2, 0.5,"T$_2$={}".format(t2),fontsize=14)
plt.text(0, 0.4,"n$_s$$_p$$_a$$_n$={}".format(20),fontsize=14)
plt.text(0.3, 0.4,"n$_s$$_t$$_e$$_p$={}".format(Step),fontsize=14)
plt.text(0.65, 0.4,"n$_c$$_y$$_c$$_l$$_e$={}".format(cycle),fontsize=14)
plt.text(0, 0.3,"t$_s$$_p$$_a$$_n$={}".format(t_span),fontsize=14)
plt.text(0.3, 0.3,"t$_s$$_t$$_e$$_p$={}".format(t_step),fontsize=14)
plt.text(0.65, 0.3,"t$_c$$_y$$_c$$_l$$_e$={}".format(t_cycle),fontsize=14)
plt.text(0, 0.2,"t$_r$$_e$$_l$$_a$$_x$=[{}".format(t_relax1),fontsize=14)
plt.text(0.4, 0.2,",{}]".format(t_relax2),fontsize=14)
plt.text(0, 0.1,"k/r*t$_s$$_p$$_a$$_n$=[{}".format(t_t1),fontsize=14)
plt.text(0.5, 0.1,",{}]".format(t_t2),fontsize=14)
plt.text(0, 0.0,"K=k*t$_s$$_t$$_e$$_p$/2r=[{}".format(K1),fontsize=14)
plt.text(0.6, 0.0,",{}]".format(K2),fontsize=14)
plt.text(0, -0.1,"t$_r$$_e$$_l$$_a$$_x$/t$_c$$_y$$_c$$_l$$_e$=[{}".format(TT1),fontsize=14)
plt.text(0.6, -0.1,",{}]".format(TT2),fontsize=14)
plt.text(0.0, -0.2,"$\eta$$_p$$_a$$_s$={}".format(eff_pas),fontsize=14)
plt.text(0.5, -0.2,"$\eta$$_a$$_c$$_t$={}".format(eff_act),fontsize=14)
ax = plt.gca()
ax.axes.xaxis.set_visible(False)
ax.axes.yaxis.set_visible(False)

plt.grid(True)
plt.show()


#%%

np.set_printoptions(threshold=5,edgeitems=5)
print(XX)






















