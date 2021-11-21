import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt

def get_cost(pi,gamma):
    pi=pi[:,1:]
    reward=np.zeros(gamma.shape[0])

    flag=np.zeros(gamma.shape[0])
    for i in range(pi.shape[1]):
        
        bool1=gamma<pi[:,i]
        bool1=bool1.astype(int)
        
        update_indices=bool1-flag
        reward=reward+pfa_lagrange*(1-pi[:,i])*update_indices*(1-flag)+ pi[:,i]*(1-update_indices)*(1-flag)
        
        flag=flag+bool1
        flag[flag>1]=1
        
    return reward

def fast_step_size(input):
  out=1/((1+input+1)**0.9)
  return out

def slow_step_size(input):
    out=30000/((1+input+1))
    return out

#Generates delta for SPSA
def deltas(input):
  out=1/((input+1)**(1/6))
  return out

def declare_attack(gamma,pi_set):
    pi_set=pi_set[:,1:]
    declare=np.zeros(gamma.shape[0])
    compare_history=np.zeros(gamma.shape[0])
    
    for j in range(pi_set.shape[1]):
        declare=declare+1*(1-compare_history)
        
        compare=gamma-pi_set[:,j]<0.01
        compare=compare.astype(int)
        compare_history=compare_history+compare
        compare_history[compare_history>1]=1
        
    declare[declare==100]=-1
    return declare

    
z_set=np.load('z_set.npy')
pi_set=np.load('pi_set.npy')
t_set=np.load('attack_instance_set.npy')

num_iters=2000
gamma_set=np.ones(z_set.shape[0])*0.5
batch_size=100 #Enter a divisor of 10,000

inp=np.arange(num_iters)
fast_steps=fast_step_size(inp)
slow_steps=slow_step_size(inp)
delta=deltas(inp)

pfa_lagrange_plot=[]
pfa_plot=[]
mean_delay_plot=[]
gamma_plot=[]

pfa=np.array([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8])#0.11,0.2,

for k in range(pfa.size):
    print(k)
    
    pfa_lagrange=500
    gamma=np.array([0.5])
    pfa_obtained=0
    mean_delay=0
    
    for i in range(num_iters):
        
        for j in range(int(10000/batch_size)):
        
            perturb= (np.random.binomial(1,0.5,gamma.size)-0.5)*2
        
            gamma_positive=gamma+delta[i]*perturb
            gamma_positive[gamma_positive>1]=1
            gamma_positive[gamma_positive<0]=0
        
            gamma_negative=gamma-delta[i]*perturb
            gamma_negative[gamma_negative>1]=1
            gamma_negative[gamma_negative<0]=0
            temp=get_cost(pi_set[j*batch_size:(j+1)*batch_size,:],gamma_positive)-get_cost(pi_set[j*batch_size:(j+1)*batch_size,:],gamma_negative)
            grad=temp/(2*delta[i]*perturb)
        
            gamma=gamma-fast_steps[i]*np.mean(grad)
            gamma[gamma>1]=1
            gamma[gamma<0]=0
        
        
        t_alarm=declare_attack(gamma,pi_set)
        delays= t_alarm- t_set
        delays_without_FL= np.sum(delays>=0) #number of NO false alarms
        delays[delays<0]=0     #removing false arlarm delays
        
        mean_delay=np.sum(np.abs(delays))/delays_without_FL
        pfa_obtained=(z_set.shape[0]-delays_without_FL)/z_set.shape[0]
        
        
        pfa_lagrange=pfa_lagrange+slow_steps[i]*(pfa_obtained-pfa[k])
        if i%10==0:
            print(pfa_lagrange, pfa_obtained, gamma)
    
    pfa_lagrange_plot.append(pfa_lagrange)
    pfa_plot.append(pfa_obtained)
    mean_delay_plot.append(mean_delay)
    gamma_plot.append(gamma)
    print('delay:', mean_delay, 'pfa:',pfa_obtained, 'gamma:',gamma)

plt.plot(pfa_plot,mean_delay_plot,marker='o',linewidth=1.5)
plt.xlabel('Probability of False Alarm')
plt.ylabel('Mean Delay')
plt.title('Mean Delay vs PFA: Our Algorithm')
plt.grid(True)
plt.savefig('plot.png', dpi=600)

pfa_lagrange_plot=np.array(pfa_lagrange_plot)
pfa_plot=np.array(pfa_plot)
mean_delay_plot=np.array(mean_delay_plot)

np.save('pfa_lagrange_plot_sgd',pfa_lagrange_plot)
np.save('pfa_plot_sgd',pfa_plot)
np.save('mean_delay_plot_sgd.npy',mean_delay_plot)
np.save('gamma_plot_sgd',gamma_plot)