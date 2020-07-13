import numpy as np
import random
import math
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.animation import FuncAnimation
from scipy import signal as sig
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体,使plot图像可以显示中文
matplotlib.rcParams['axes.unicode_minus'] = False    # 解决保存图像是负号'-'显示为方块的问题
long=300
def ElectromagneticSignal(c,tao):
    '''
    航空瞬变电磁信号
    '''
    sums = 0
    t=np.linspace(1,long+1,long)
    n = 1000
    for i in range(1,n+1):
            sums += np.exp(-(i*i)*(t / tao))
    signal = (c / tao) * sums
    
    return signal
def GaussianNoise(μ,sigma):
    '''
    高斯噪声
    '''
    noise=np.random.normal(μ,sigma,long)
    return noise
def RandomNoise(a,b):
    '''
    随机噪声
    '''
    Ex=(a+b)/2
    Sx=np.sqrt((b+a)**2/12)
    noise=np.random.normal(Ex,Sx,long)
    return noise
def AtmosphericNoise(μ,sigma):
    '''
    天电噪声
    '''
    w=GaussianNoise(μ,sigma)
    p,q=2,2#ARMA(2,2)
    a = [1.0 , 1.45 , 0.6]#自回归系数ai = {1.0 , 1.45 , 0.6}
    b = [1.0 , -0.2 , -0.1]#滑动平均系数bi = {1.0 , -0.2 , -0.1}
    x=np.zeros(long)
    x[0]=b[0]*w[0]
    for n in range(1,long):
        sum=0
        for i in range (1,p+1):
            if (n==1):
                sum+=-a[1]*x[n-1]
                break
            else:
                sum+=-a[i]*x[n-i]
        for j in range (1,q+1):
            if (n==1):
                sum+=b[1]*w[n-1]
                break
            else:
                sum+=b[j]*w[n-j]
        x[n]=sum
    return x
def elect_plus_gaosi():
    a=np.array(ElectromagneticSignal(60,10))
    b=np.array(GaussianNoise(1,0.5))
    return(a+b),a,b
def elect_plus_random():
    a=np.array(ElectromagneticSignal(60,10))
    b=np.array(RandomNoise(0,1))
    return(a+b),a,b
def elect_plus_atmose():
    a=np.array(ElectromagneticSignal(60,10))
    b=np.array(AtmosphericNoise(1,3))
    return(a+b),a,b
def ffttransfor(x):
    signal = np.asarray(x, dtype=float)
    a = signal.shape[0]
    b = np.arange(a)
    c = b.reshape((a, 1))
    d = np.exp(-2j * np.pi * c * b / a)
    f = np.dot(d, signal)
    return f
def SNR(ori,nois):
    Ps=[]
    Pn=[]
    for d in range(len(ori)):
        Ps.append((ori[d])**2)
        Pn.append((nois[d])**2)    
    snr=10*np.log10(np.mean(Ps)/np.mean(Pn))
    return snr
class FIlter():
    def __init__(self,signal):
        self.signal=signal
        self.step=2
        self.Fc=100
    def FIR(self,x_1):  
        b, a = sig.butter(8,2.0*self.Fc/1000, 'lowpass')  
        filtedData = sig.filtfilt(b, a,x_1) 
        return filtedData
    def Mean(self,org):
        orgin=[]
        for j in org:
            orgin.append(j)
        filte=[]
        for i in range(self.step):
            orgin.insert(0,orgin[0])
            orgin.append(orgin[-1])
        for j in range(len(org)):
            a=orgin[j:j+2*self.step]
            b=np.mean(a)
            filte.append(b)
        return filte
    def Median(self,org):
        orgin=[]
        for j in org:
            orgin.append(j)
        filte = []
        for i in range(self.step):
            orgin.insert(0,orgin[0])
            orgin.append(orgin[-1])
        for k in range(len(org)):
            a=orgin[k:k+2*self.step]
            a.sort() 
            filte.append(a[self.step])
        return filte
    def Kalman(self):
        shape = self.signal.shape[0]
        x_predict = np.zeros(shape)  # 信号的预测值
        Perro = np.zeros(shape)  # 信号的误差估计
        x_update = np.zeros(shape)  # x的最优值
        P = np.zeros(shape)  # 协方差的误差估计
        K = np.zeros(shape)  # 观测值权重
        R, Q = 0.000001, 0.000000001#不确定度R Q是一些无法确认的干扰噪音
        P[0], x_update[0] = 1.0, self.signal[0]
        for i in range(1, shape):
            x_predict[i] = x_update[i - 1]
            Perro[i] = P[i - 1] + Q
            # 更新过程
            K[i] = Perro[i] / (Perro[i] + R)
            x_update[i] = x_predict[i] + K[i] * (self.signal[i] - x_predict[i])
            P[i] = (1 - K[i]) * Perro[i]
        return x_update

def ani(signal):
    result_list=[]
    t_list=[]
    for i in range(100):
        if i==1:
            a=input()
        t_list.append(i)
        result_list.append(signal[i])
        plt.title('天电噪声')
        plt.plot(t_list,result_list,'b')
        plt.pause(0.01)
ani(AtmosphericNoise(1,3))

# signal,ori,noise=elect_plus_atmose()
# fil=FIlter(signal)
# kalm=fil.Kalman()
# snrsignal=SNR(signal,noise)
# snrfil=SNR(kalm,noise)
# print('电磁信号+高斯噪声SNR',snrsignal)
# print('滤波后SNR',snrfil)

#plt.plot(signal,label='电磁信号+天电噪声')
# plt.figure()
# plt.title('电磁信号+随机噪声频谱分析')
# plt.plot(fftsignal)
# plt.xlabel('Time')
# plt.figure()
# plt.title('电磁信号频谱分析')
# plt.plot(fftori)
# plt.xlabel('Time')
# plt.figure()
# plt.title('随机噪声频谱分析')
# plt.plot(fftnoise)
# plt.xlabel('Time')
# plt.figure()
# plt.title('电磁信号+随机噪声滤波后频谱分析')
# plt.plot(fftfilter)
# plt.xlabel('Time')
#plt.plot(noise)
#count,x,ignored=plt.hist(noise,300,density=True)
#plt.plot(kalm,label='卡尔曼滤波后')
# plt.figure()
# plt.plot(ffttransfor(signal))
# print(SNR(signal,noise))
# print(SNR(kalm,noise))
#plt.legend()
plt.show()
