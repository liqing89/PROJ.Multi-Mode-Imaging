'''
该函数用来成像
'''
import numpy as np
import read_data
import scipy.io as sio
#from skimage import exposure
import matplotlib.pyplot as plt

class imaging:
    def __init__(self,path,beta):
        self.data,para = read_data.load_data(path)
        # 光速
        self.c = para['c']
        # 载频
        self.fc = para['fc']
        # 波长
        self.lamda = para['lamda']
        # 线性调频率
        self.Kr = para['Kr']
        # 脉宽
        self.Tp = para['Tp']
        # 雷达收发间隔
        self.PRF = para['PRF']
        # 采样率
        self.fs = para['fs']
        # 带宽
        self.Br = para['Br']
        # 起始斜距
        self.R0 = para['R0']
        # 飞行速度
        self.Vr = para['Vr']
        # 天线长度
        self.La = para['La']
        # 成像模式
        self.scanMode = para['scanMode']
        # 方位向范围
        self.azmScale = para['azmScale']-100
        # 距离向范围
        self.rngScale = para['rngScale']-100
        # 距离分辨率
        self.dr = self.c/2/self.fs
        # 方位向分辨率
        self.dazm = self.Vr/self.PRF
        # 凯撒窗系数
        self.beta = beta
    def Rangewindow(self):
        '''
        生成距离维的窗
        '''
        Na,Nr = self.data.shape
        # window = np.hamming(np.ceil(self.Br/self.fs*Nr))
        window = np.kaiser(np.ceil(self.Br/self.fs*Nr), self.beta)
        # 补0
        Length_w = len(window)
        window = np.concatenate((np.zeros(int(np.ceil((Nr-Length_w)/2))),window))
        window = np.concatenate((window,np.zeros(int(np.floor((Nr-Length_w)/2)))))
        window = np.fft.fftshift(window).reshape(1,-1)
        return window
    def Azimuthwindow(self):
        '''
        生成方位维的窗
        '''
        Na,Nr = self.data.shape
        # window = np.hamming(np.ceil(2*self.Vr*self.lamda/self.La/self.lamda/self.PRF*Na))
        window = np.kaiser(np.ceil(2*self.Vr*self.lamda/self.La/self.lamda/self.PRF*Na), self.beta)
        # 补0
        Length_w = len(window)
        window = np.concatenate((np.zeros(int(np.ceil((Na-Length_w)/2))),window))
        window = np.concatenate((window,np.zeros(int(np.floor((Na-Length_w)/2)))))
        window = np.fft.fftshift(window).reshape(-1,1)
        return window
    
    def show_image(self,mode,save_path):
        '''
        截取成像区域并显示
        '''
        Na,Nr = self.image.shape
        Nr_start = int(self.Tp*self.fs/2)+int(50/self.dr)
        Nr_end = Nr_start + int(self.rngScale/self.dr)
        Na_start = int(Na/2) - int(self.azmScale/self.dazm/2)
        Na_end = Na_start + int(self.azmScale/self.dazm)
        I = self.image[Na_start:Na_end,Nr_start:Nr_end]

        if mode == 1:
            # 条带模式下背景自适应量化
            I = np.abs(I) # 取图像幅值
            I = I/I.max() #归一化
            I = 20*np.log10(np.abs(I)) #dB显示
            I = np.clip(I,-60,0) #截断最值
            I = (I + 60)/60*255 # 线性量化
            sio.savemat(save_path,{'image':I.astype(np.uint8) }) #转成uint8输出

        elif mode == 2:
            # 用线性显示
            # date:2023-09-06
            I = np.abs(I) # 幅度
            I = I/I.max() #归一化
            # 对图像中低于阈值的强度值进行拉伸，将他们映射到0到255的强度范围内
            p1, p99 = np.percentile(I, (1, 99.5))
            I = np.clip(I,p1,p99)
            # 1.线性映射 0到255
            # I = (I - p1) / (p99 - p1) * 255
            # 2.固定背景均值的量化值为bg_uint
            bg_uint = 50;
            mean_val = np.mean(I[np.logical_and(I<np.max(I)*0.1,I>np.min(I)*10)]) # 背景强度均值
            N = np.mean(np.round([bg_uint * (p99 - p1) / (mean_val -p1)])) # 按照背景固定为25灰度计算，图像灰度取值量化上下限为0-N
            M = I.max() 
            # 压缩量化映射范围
            I = (I - p1) / (p99 - p1) * N # 将图像投影到0-N范围内
            I[0,0] = (M - p1) / (p99 - p1) * 255 #图像总量化范围为0-255不变
            # 上限截断
            # pct = 0.1
            # I = np.where(I > I.max()*pct, I.max()*pct, I)
            # 使用伽马校正进一步增强对比度
            # I = (I/255)**(1/1.5)*255
            sio.savemat(save_path,{'image':I.astype(np.uint8) })


    def imaging(self):
        if self.scanMode == 1:
            self.image = self.tiaodai()
    def tiaodai(self):
        # 读取数据
        ss = self.data
        # 读取数据宽度
        Na,Nr = ss.shape
        # CS变标
        f_fast = np.fft.fftshift(np.linspace(-self.fs/2,self.fs/2-self.fs/Nr,Nr).reshape(1,-1))
        f_slow = np.fft.fftshift(np.linspace(-self.PRF/2,self.PRF/2-self.PRF/Na,Na).reshape(-1,1))
        cos_theta = np.sqrt(1-(self.lamda*f_slow/2/self.Vr)**2)
        R_start = self.R0 - self.Tp/2*self.c/2
        R_ref = self.R0 + (Nr-self.Tp*self.fs)/2*self.c/2/self.fs
        t_fast = (2*R_start/self.c + np.linspace(0,Nr-1,Nr).reshape(1,-1)/self.fs).reshape(1,-1)
        Km = self.Kr/(1-self.Kr*self.lamda*R_ref*f_slow**2/2/self.Vr**2/self.fc**2/cos_theta**3)
        ss = np.fft.fft(ss,axis=0)
        ss = ss*np.exp(1j*np.pi*Km*(1/cos_theta-1)*(t_fast-2*R_ref/self.c/cos_theta)**2)
        # 距离徙动和脉冲压缩
        ss = np.fft.fft(ss,axis=1)
        # 加窗
        ss = ss * self.Rangewindow() * self.Azimuthwindow()
        ss = ss*np.exp(1j*np.pi/Km*cos_theta*f_fast**2)*np.exp(1j*4*np.pi/self.c*R_ref*(1/cos_theta-1)*f_fast)
        ss = np.fft.ifft(ss,axis=1)
        # 脉冲压缩
        ss = ss*np.exp(-1j*4*np.pi*Km/self.c**2*(1-cos_theta)*(t_fast/cos_theta-2*R_ref/self.c/cos_theta)**2)
        R = t_fast*self.c/2
        ss = ss*np.exp(1j*4*np.pi*R/self.lamda*cos_theta)
        ss = np.fft.ifft(ss,axis=0)
        return ss
    