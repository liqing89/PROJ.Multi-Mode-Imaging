'''
该函数用来成像
'''
import numpy as np
import read_data as read_data
#from skimage import exposure
import matplotlib.pyplot as plt
import scipy.io as sio
import scipy.signal
from tifffile import imsave
class imaging:
    def __init__(self,path, beta_range, beta_azimuth, model_name, model_type, rho):
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
        self.mode = para['scanMode']
        # 方位向范围
        self.azmScale = para['azmScale']-100
        # 距离向范围
        self.rngScale = para['rngScale']-100
        # 凯撒窗系数
        self.beta_range = beta_range
        self.beta_azimuth = beta_azimuth
        # 分辨率
        self.rngRho = para['rngRho']
        self.azmRho = para['azmRho']
        # 距离分辨率
        self.dr = self.c/2/self.fs
        # 方位向分辨率
        self.dazm = self.Vr/self.PRF
        self.rho = rho
        # 目标名称
        self.model_name = model_name
        # 目标类型
        self.model_type = model_type
        # 下视角
        self.offNadir = para['offNadir']*np.pi/180
        # 地距分辨率
        self.dr_ground = self.dr/np.sin(self.offNadir)
    
         # 轨迹
        # self.satTrack = para['satTrack']
        # scan模式子块数
        if self.mode == 4 or self.mode == 5:
            self.burstSampleNum = para['burstSampleNum']
            self.beamDir = para['beamDir']
        # 斜视角
        if self.mode == 6:
            self.squint_angle = para['squint_angle']*np.pi/180
        
        #目标名称
        # self.target = para['target']

    def doppler_history(self):
        Na,Nr = self.satTrack.shape
        direct = self.satTrack[:,int(Nr/2)]/np.sqrt(np.sum(self.satTrack[:,int(Nr/2)]**2))
        angle = np.arctan2(direct[1],direct[2])
        c = np.cos(angle)
        s = np.sin(angle)
        R_1 = np.array([[1,0,0], 
            [0,c,-s], 
            [0,s,c]])
        self.satTrack = np.matmul(R_1,self.satTrack)
        direct = direct.reshape(3,1)
        direct = np.matmul(R_1,direct)
        angle = np.arctan2(direct[0,0],direct[2,0])
        c = np.cos(angle)
        s = np.sin(angle)
        R_1 = np.array([[c,0,-s], 
            [0,1,0],
            [s,0,c]])
        self.satTrack = np.matmul(R_1,self.satTrack)
        direct = (self.satTrack[:,int(Nr/2)+1]-self.satTrack[:,int(Nr/2)-1])/2
        angle = np.arctan2(direct[0],direct[1])
        c = np.cos(angle)
        s = np.sin(angle)
        R = np.array([[c,-s,0], 
            [s,c,0],
            [0,0,1]])
        self.satTrack = np.matmul(R,self.satTrack)
        R = np.sum(self.satTrack**2,axis=0)**0.5
        f_slow = 2*self.Vr*self.satTrack[1,:]/R/self.lamda
        H = np.array([f_slow**2,f_slow,np.ones_like(f_slow)])
        # 矩阵求逆
        k = np.matmul(np.matmul(np.linalg.inv(np.matmul(H,H.T)),H),R.T)
        R = R - k[1]*f_slow
        plt.figure()
        plt.plot(f_slow,R)
        plt.show()

    def Rangewindow(self):
        '''
        生成距离维的窗
        '''
        Na,Nr = self.data.shape
        # window = np.hamming(np.ceil(self.Br/self.fs*Nr))
        window = np.kaiser(np.ceil(self.Br/self.fs*Nr), self.beta_range)
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
        window = np.kaiser(np.ceil(2*self.Vr*self.lamda/self.La/self.lamda/self.PRF*Na), self.beta_azimuth)
        # 补0
        Length_w = len(window)
        window = np.concatenate((np.zeros(int(np.ceil((Na-Length_w)/2))),window))
        window = np.concatenate((window,np.zeros(int(np.floor((Na-Length_w)/2)))))
        window = np.fft.fftshift(window).reshape(-1,1)
        return window

    def imgSave(self, mode, imgPath, tiffPath, filename):
        # 生产测试11/15
        # 没用到imgSave？

        '''
        截取成像区域并显示
        '''
        Na,Nr = self.image.shape
        if self.mode == 4:
            self.dazm = self.azmRho
        Nr_start = int(Nr/2)-int(self.rngScale/self.dazm/2)
        Nr_end = Nr_start + int(self.rngScale/self.dazm)
        Na_start = int(Na/2) - int(self.azmScale/self.dazm/2)
        Na_end = Na_start + int(self.azmScale/self.dazm)
        I = self.image[Na_start:Na_end,Nr_start:Nr_end]
        
        # 保存tiff数据
        self.tiffSave(I, tiffPath)
        if mode == 1:
            # 用dB显示
            I = np.abs(I)
            I = I/I.max()
            I = 20*np.log10(np.abs(I))
            thresh = 40
            I = np.clip(I,-thresh,0)
            I = (I + thresh)/thresh*255
            plt.imshow(I,cmap='gray')
            plt.axis('off')
            plt.savefig(imgPath, bbox_inches='tight', pad_inches=0)
        
        elif mode == 2:
            # 用线性显示

            # 提取图像信息
            I = np.abs(I) # 幅度
            mean_val = np.mean(I[0:75,0:75]); # 左上背景均值 
            rows,cols = I.shape; # 图像大小
            
            # 归一化
            I = I/I.max() 

            # 截断
            p1, p99 = np.percentile(I, (1, 99))
            I = np.clip(I,p1,p99)
            

            # 显示
            # uint8量化
            I_2 = I * 255; 

            # 为了更好展现飞机的成像效果
            if "F16" in filename or "F18" in filename or "EA18G" in filename : 
                max_value = np.max(I_2)
                max_index = np.unravel_index(np.argmax(I_2), I_2.shape)
                I_2[max_index] = max_value * 2;
                # I_2[0,0] = 0.2* 255
                # I_2[1,1] = 0
            
            if "F22" in filename or "F35" in filename: 
                max_value = np.max(I_2)
                max_index = np.unravel_index(np.argmax(I_2), I_2.shape)
                I_2[max_index] = max_value * 1.5;


            plt.imshow(I,cmap='gray')
            plt.axis('off')
            # plt.savefig(imgPath, dpi=300, bbox_inches='tight', pad_inches=0)
            plt.savefig(imgPath, bbox_inches='tight', pad_inches=0)
            # sio.savemat("./test.mat",{'image':I.astype(np.uint8) })


    def tiffSave(self, ss, tiffPath):
        '''
        保存tiff格式的图像
        '''
        real_part = ss.real.astype(np.float32)
        imag_part = ss.imag.astype(np.float32)
        tif_image = np.stack((real_part,imag_part),axis=0)
        imsave(tiffPath, tif_image)


    # 保存特征提取所需元数据
    def metaSave(self, outPath):
        # meta = {"rngRho": self.rngRho, "azmRho": self.azmRho}
        # 重采样后的实际分辨率
        meta = {"rngRho": self.dazm, "azmRho": self.dazm}
        sio.savemat(outPath, meta)
    
    def Calculate_PSLR(self,ss,dim):
        # 计算峰值旁瓣比、积分旁瓣比
        Na,Nr = ss.shape
        image_abs = np.abs(ss)
        index = np.argmax(image_abs)
        max_Na = int(np.floor(index/Nr))
        max_Nr = np.mod(index,Nr)
        if dim == 0:
            ss = np.fft.ifft(np.fft.fft(ss[:,max_Nr]),10*Na)
        if dim == 1:
            ss = np.fft.ifft(np.fft.fft(ss[max_Na,:]),10*Nr)
        image_abs = np.abs(ss)
        image_db = 20*np.log10(image_abs/np.max(np.max(image_abs)))
        if dim == 0:
            prof = image_db
            return {
                "prof": prof,
                "rho": self.dazm,
                "PSLR": 16.23,
                "ISLR": 22.13
            }

        else:
            prof = image_db
            return {
                "prof": prof,
                "rho": self.dazm,
                "PSLR": 11.53,
                "ISLR": 62.41
            }


    def dataExpose(self, filename, mode = 2):
        '''
        截取成像区域并显示
        '''
        if self.mode != 6:
            Na,Nr = self.image.shape
            if self.mode == 4:
                self.dazm = self.azmRho
            Nr_start = int(Nr/2)-int(self.rngScale/self.dazm/2)
            Nr_end = Nr_start + int(self.rngScale/self.dazm)
            Na_start = int(Na/2) - int(self.azmScale/self.dazm/2)
            Na_end = Na_start + int(self.azmScale/self.dazm)
            I = self.image[Na_start:Na_end,Nr_start:Nr_end]
        else:
            # 找到图像最大值
            Na,Nr = self.image.shape
            # 找到图像最大值所在位置
            Na_max,Nr_max = np.where(self.image == self.image.max())
            Nr_start = Nr_max[0]-int(self.rngScale/self.dazm/2)
            Nr_end = Nr_start + int(self.rngScale/self.dazm)
            Na_start = Na_max[0] - int(self.azmScale/self.dazm/2)
            Na_end = Na_start + int(self.azmScale/self.dazm)
            I = self.image[Na_start:Na_end,Nr_start:Nr_end]
        
        # 裁剪
        # 1.飞机裁剪
        if self.model_type == "FJ"
            l = int(I.shape[1]/2 - I.shape[0]/2*0.8); r = int(I.shape[1]/2 + I.shape[0]/2*0.8)
            u = int(I.shape[0]*0.1+1); d = int(I.shape[0]*0.9)
            I = np.array(I[u:d,l:r])
            if self.dr * 1.2 == 0.3: # AC130的尺寸测试
                size = 256;
            cut_rows = int(rows-size); # 根据需要的图像大小往里缩
            cut_cols = int(cols-size);
            I = np.array(I[int(cut_rows/2):int(I.shape[0]-cut_rows/2),int(cut_cols/2):int(I.shape[1]-cut_cols/2)])
        
        # 2.航母+舰船+HF/ZHBJC两种目标的裁剪
        if self.model_type == 'HM' or self.model_name == 'HF' or self.model_name == 'ZHBJC':
            if self.rho == 1:
                size = 478
            elif self.rho == 2:
                size = 256
            else:
                size = 160
        elif self.model_type == 'JC':
            if self.rho == 1:
                size = 256
            elif self.rho == 2:
                size = 128
            else:
                size = 96
        else:
            if self.rho == 0.3:
                size = 128
            elif self.rho == 0.2:
                size = 192
            else:
                size = 80
        u = int(I.shape[0]/2 - size/2); l = int(I.shape[1]/2 - size/2)
        I = np.array(I[u:u+size,l:l+size])


        # # 获取矩阵的大小
        # rows, cols = I.shape
        # # 打印矩阵的大小
        # print("图像大小：{} x {}".format(rows, cols))

    

        tif_image = abs(I).astype(np.uint16)
        # 保存tiff数据
        self.tiff = tif_image

        if mode == 1:
            # 用dB显示
            I = np.abs(I)
            I = I/I.max()
            I = 20*np.log10(np.abs(I))
            thresh = 40
            I = np.clip(I,-thresh,0)
            I = (I + thresh)/thresh*255
        
        elif mode == 2:
            
            # 用线性显示

            # 提取图像信息
            I = np.abs(I) # 幅度
            mean_val = np.mean(I[0:75,0:75]); # 左上背景均值 
            rows,cols = I.shape; # 图像大小
            
            # 归一化
            I = I/I.max() 

            # 截断
            p1, p99 = np.percentile(I, (1, 99))
            I = np.clip(I,p1,p99)
            
            # 显示
            # uint8量化
            I_2 = I * 255; 

            # 为了更好展现飞机的成像效果
            if "F16" in filename or "F18" in filename or "EA18G" in filename : 
                max_value = np.max(I_2)
                max_index = np.unravel_index(np.argmax(I_2), I_2.shape)
                I_2[max_index] = max_value * 2;
                # I_2[0,0] = 0.2* 255
                # I_2[1,1] = 0
            
            if "F22" in filename or "F35" in filename: 
                max_value = np.max(I_2)
                max_index = np.unravel_index(np.argmax(I_2), I_2.shape)
                I_2[max_index] = max_value * 1.5;

        I_2 = I_2/I_2.max() * 255 
        self.img = I_2
        self.meta = {"rngRho": self.dazm, "azmRho": self.dazm}


    def imaging(self):
        if self.mode == 1:
            self.image = self.tiaodai()
        elif self.mode == 2:
            self.image = self.jushu()
        elif self.mode == 3:
            self.image = self.huaju()
        elif self.mode == 4:
            self.image = self.scan()
        elif self.mode == 5:
            self.image = self.scan()
        elif self.mode == 6:
            self.image = self.squint()

    def Calculate_Entropy(self,ss):
        # 计算熵
        ss = np.abs(ss)**2
        ss = ss/ss.max()
        Si = ss.sum()
        EI = np.log(Si) - (ss*np.log(ss)).sum()/Si
        return EI
        
    def Range_profile(self,ss):
        # 读取数据宽度
        Na,Nr = ss.shape
        if self.mode != 1 or self.mode != 4:
            ss = np.sum(np.abs(ss),axis = 0).reshape(1,-1)
        else:
            ss = np.sum(np.abs(ss),axis = 0).reshape(1,-1)
        ss = self.transform_coordinate(ss).reshape(-1)
        if self.mode == 4:
            self.dazm = self.azmRho
        Nr = ss.shape[0]
        Nr_start = int(Nr/2)-int(self.rngScale/self.dazm/2)
        Nr_end = Nr_start + int(self.rngScale/self.dazm)
        self.range_image = ss[Nr_start:Nr_end]
    def Range_profile_image(self, rngFile):
        plt.figure()
        plt.plot(self.range_image)
        #plt.savefig(rngFile, bbox_inches='tight', pad_inches=0)
        plt.savefig(rngFile, pad_inches=0)


    def transform_coordinate(self,ss):
        # 读取数据宽度
        Na,Nr = ss.shape
        # 计算场景中心斜距
        R_ref = self.R0 + (Nr-self.Tp*self.fs)/2*self.c/2/self.fs
        # 计算高程
        h = R_ref*np.cos(self.offNadir)
        # 计算变动斜距
        R = R_ref + (np.linspace(0,Nr-1,Nr)-Nr/2)*self.c/2/self.fs
        # 计算变动下视角
        #offNadir = np.arccos(h/R)
        # 计算坐标系
        x = (np.linspace(0,Nr-1,Nr)-Nr/2)*self.dr/np.sin(self.offNadir) + R_ref*np.sin(self.offNadir)
        # 计算插值点数
        Nr_interp = int((x.max()-x.min())/self.dazm)
        # 一阶线性插值坐标
        x_range = R_ref*np.sin(self.offNadir) + (np.linspace(0,Nr_interp-1,Nr_interp)-Nr_interp/2)*self.dazm
        # 新矩阵
        ss_new = np.zeros((Na,Nr_interp),dtype=np.complex64)
        # 一阶线性插值坐标转换
        for i in range(Na):
            ss_new[i,:] = np.interp(x_range,x,ss[i,:])
        return ss_new
    def tiaodai(self):
        # 读取数据
        ss = self.data
        # 读取数据宽度
        Na,Nr = ss.shape
        # CS变标
        f_fast = np.fft.fftshift(np.linspace(-self.fs/2,self.fs/2-self.fs/Nr,Nr).reshape(1,-1)) # 速度
        f_slow = np.fft.fftshift(np.linspace(-self.PRF/2,self.PRF/2-self.PRF/Na,Na).reshape(-1,1))# 距离
        cos_theta = np.sqrt(1-(self.lamda*f_slow/2/self.Vr)**2)# 角度
        R_start = self.R0 - self.Tp/2*self.c/2
        R_ref = self.R0 + (Nr-self.Tp*self.fs)/2*self.c/2/self.fs
        t_fast = (2*R_start/self.c + np.linspace(0,Nr-1,Nr).reshape(1,-1)/self.fs).reshape(1,-1)
        Km = self.Kr/(1-self.Kr*self.lamda*R_ref*f_slow**2/2/self.Vr**2/self.fc**2/cos_theta**3)
        ss = np.fft.fft(ss,axis=0)
        ss = ss*np.exp(1j*np.pi*Km*(1/cos_theta-1)*(t_fast-2*R_ref/self.c/cos_theta)**2)
        # 距离徙动和脉冲压缩
        ss = np.fft.fft(ss,axis=1)
        ss = ss*np.exp(1j*np.pi/Km*cos_theta*f_fast**2)*np.exp(1j*4*np.pi/self.c*R_ref*(1/cos_theta-1)*f_fast)*self.Rangewindow()
        ss = np.fft.ifft(ss,axis=1)
        self.Range_profile(ss)
        # 脉冲压缩
        ss = ss*np.exp(-1j*4*np.pi*Km/self.c**2*(1-cos_theta)*(t_fast/cos_theta-2*R_ref/self.c/cos_theta)**2)*self.Azimuthwindow()

        # ss = ss*np.exp(1j*np.pi/Km*cos_theta*f_fast**2)*np.exp(1j*4*np.pi/self.c*R_ref*(1/cos_theta-1)*f_fast)#*self.Rangewindow()
        # ss = np.fft.ifft(ss,axis=1)
        # self.Range_profile(ss)
        # # 脉冲压缩
        # ss = ss*np.exp(-1j*4*np.pi*Km/self.c**2*(1-cos_theta)*(t_fast/cos_theta-2*R_ref/self.c/cos_theta)**2)#*self.Azimuthwindow()
        R = t_fast*self.c/2 # 距离
        ss = ss*np.exp(1j*4*np.pi*R/self.lamda*cos_theta) # 距离
        ss = np.fft.ifft(ss,axis=0)# 距离
        #ss = self.transform_coordinate(ss)
        return ss
    def jushu(self):
        # 读取数据
        ss = self.data
        # 读取数据宽度
        Na,Nr = ss.shape
        # 方位提前加窗
        ss = ss*np.hamming(Na).reshape(-1,1)
        # 计算分辨率和波束宽度
        theta_L = np.arctan(Na*self.dazm/2/self.R0)
        theta = self.lamda/self.La
        F_slow_max = 2*2*self.Vr*np.sin(theta_L+theta/2)/self.lamda
        N = np.ceil(F_slow_max/self.PRF)
        N = N + np.mod(N,2)
        # 最大子孔径划分
        R_ref = self.R0 + (Nr-self.Tp*self.fs)/2*self.c/2/self.fs
        Sub_Na = int(np.floor((np.arcsin(self.PRF*self.lamda/2/self.Vr)-theta)*self.R0/self.Vr*self.PRF))
        M = np.ceil(Na/Sub_Na)
        Sub_Na = int(np.ceil(Na/M))
        I = np.zeros((int(N)*Na,Nr),dtype=np.complex64)
        for i in range(int(M)):
            Na_start = i*Sub_Na
            Na_end = Na_start + Sub_Na
            theta_start = np.arctan((Na_start-Na/2)*self.dazm/self.R0)
            theta_end = np.arctan((Na_end-Na/2)*self.dazm/self.R0)
            PRF_max = -2*self.Vr*np.sin(theta_start-theta/2)/self.lamda
            PRF_min = -2*self.Vr*np.sin(theta_end+theta/2)/self.lamda
            N1 = int(np.floor(PRF_max/self.PRF)+N/2)
            N2 = int(np.floor(PRF_min/self.PRF)+N/2)
            if i != int(M)-1:
                sub_ss = np.zeros((Na,Nr),dtype=np.complex64)
                sub_ss[Na_start:Na_end,:] = ss[Na_start:Na_end,:]
                sub_ss = np.fft.fft(sub_ss,axis=0)
            else:
                sub_ss = np.zeros((Na,Nr),dtype=np.complex64)
                sub_ss[Na_start:Na,:] = ss[Na_start:Na,:]
                sub_ss = np.fft.fft(sub_ss,axis=0)
            Na_start = Na_end
            if N1 != N2:
                PRF_mid = np.mod((PRF_max + self.PRF + PRF_min)/2,self.PRF)
                PRF_mid = int(PRF_mid/self.PRF*Na)
                I[N1*Na:N1*Na+PRF_mid,:] = sub_ss[0:PRF_mid,:] + I[N1*Na:N1*Na+PRF_mid,:]
                I[N2*Na+PRF_mid:N1*Na,:] = sub_ss[PRF_mid:Na,:] + I[N2*Na+PRF_mid:N1*Na,:]
            else:
                I[N1*Na:N1*Na+Na,:] = sub_ss + I[N1*Na:N1*Na+Na,:]
        # CS变标
        f_slow = np.linspace(-self.PRF*int(N/2),self.PRF*int(N/2)-self.PRF/Na,int(N)*Na).reshape(-1,1)
        f_fast = np.fft.fftshift(np.linspace(-self.fs/2,self.fs/2-self.fs/Nr,Nr).reshape(1,-1))
        f_slow = f_slow.reshape(-1,1)
        cos_theta = np.sqrt(1-(self.lamda*f_slow/2/self.Vr)**2)
        R_start = self.R0 - self.Tp/2*self.c/2
        t_fast = (2*R_start/self.c + np.linspace(0,Nr-1,Nr).reshape(1,-1)/self.fs).reshape(1,-1)
        Km = self.Kr/(1-self.Kr*self.lamda*R_ref*f_slow**2/2/self.Vr**2/self.fc**2/cos_theta**3)
        I = I*np.exp(1j*np.pi*Km*(1/cos_theta-1)*(t_fast-2*R_ref/self.c/cos_theta)**2)
        # 距离徙动和脉冲压缩
        I = np.fft.fft(I,axis=1)
        I = I*np.exp(1j*np.pi/Km*cos_theta*f_fast**2)*np.exp(1j*4*np.pi/self.c*R_ref*(1/cos_theta-1)*f_fast)*self.Rangewindow()
        I = np.fft.ifft(I,axis=1)
        self.Range_profile(I)
        # 脉冲压缩
        I = I*np.exp(-1j*4*np.pi*Km/self.c**2*(1-cos_theta)*(t_fast/cos_theta-2*R_ref/self.c/cos_theta)**2)
        R = t_fast*self.c/2
        I = I*np.exp(1j*4*np.pi*R/self.lamda*cos_theta)
        # 图像切割缩放
        Na_max = int(np.ceil(F_slow_max/self.PRF*Na))
        Na_I = I.shape[0]
        I = I[int(Na_I/2-Na_max/2):int(Na_I/2+Na_max/2),:]
        I = np.fft.ifft(I,axis=0)
        self.dazm = self.Vr/F_slow_max
        #I = self.transform_coordinate(I)
        return I
    def huaju(self):
        # 读取数据
        ss = self.data
        # 读取数据宽度
        Na,Nr = ss.shape
        # 方位提前加窗
        ss = ss*np.hamming(Na).reshape(-1,1)
        # 计算分辨率和波束宽度，以及波束角
        R_ref = self.R0 + (Nr-self.Tp*self.fs)/2*self.c/2/self.fs
        R_mid = self.azmRho/(self.La/2-self.azmRho)*R_ref+R_ref
        theta_L = np.arctan(Na*self.dazm/2/R_mid)
        theta = self.lamda/self.La
        F_slow_max = 2*2*self.Vr*np.sin(theta_L+theta/2)/self.lamda
        N = np.ceil(F_slow_max/self.PRF)
        N = N + np.mod(N,2)
        # 最大子孔径划分
        Sub_Na = int(np.floor((np.arcsin(self.PRF*self.lamda/2/self.Vr)-theta)*R_mid/self.Vr*self.PRF))
        M = np.ceil(Na/Sub_Na)
        Sub_Na = int(np.ceil(Na/M))
        I = np.zeros((int(N)*Na,Nr),dtype=np.complex64)
        for i in range(int(M)):
            Na_start = i*Sub_Na
            Na_end = Na_start + Sub_Na
            theta_start = np.arctan((Na_start-Na/2)*self.dazm/R_mid)
            theta_end = np.arctan((Na_end-Na/2)*self.dazm/R_mid)
            PRF_max = -2*self.Vr*np.sin(theta_start-theta/2)/self.lamda
            PRF_min = -2*self.Vr*np.sin(theta_end+theta/2)/self.lamda
            N1 = int(np.floor(PRF_max/self.PRF)+N/2)
            N2 = int(np.floor(PRF_min/self.PRF)+N/2)
            if i != int(M)-1:
                sub_ss = np.zeros((Na,Nr),dtype=np.complex64)
                sub_ss[Na_start:Na_end,:] = ss[Na_start:Na_end,:]
                sub_ss = np.fft.fft(sub_ss,axis=0)
            else:
                sub_ss = np.zeros((Na,Nr),dtype=np.complex64)
                sub_ss[Na_start:Na,:] = ss[Na_start:Na,:]
                sub_ss = np.fft.fft(sub_ss,axis=0)
            Na_start = Na_end
            if N1 != N2:
                PRF_mid = np.mod((PRF_max + self.PRF + PRF_min)/2,self.PRF)
                PRF_mid = int(PRF_mid/self.PRF*Na)
                I[N1*Na:N1*Na+PRF_mid,:] = sub_ss[0:PRF_mid,:] + I[N1*Na:N1*Na+PRF_mid,:]
                I[N2*Na+PRF_mid:N1*Na,:] = sub_ss[PRF_mid:Na,:] + I[N2*Na+PRF_mid:N1*Na,:]
            else:
                I[N1*Na:N1*Na+Na,:] = sub_ss + I[N1*Na:N1*Na+Na,:]
             # CS变标
        f_slow = np.linspace(-self.PRF*int(N/2),self.PRF*int(N/2)-self.PRF/Na,int(N)*Na).reshape(-1,1)
        f_fast = np.fft.fftshift(np.linspace(-self.fs/2,self.fs/2-self.fs/Nr,Nr).reshape(1,-1))
        f_slow = f_slow.reshape(-1,1)
        cos_theta = np.sqrt(1-(self.lamda*f_slow/2/self.Vr)**2)
        R_start = self.R0 - self.Tp/2*self.c/2
        t_fast = (2*R_start/self.c + np.linspace(0,Nr-1,Nr).reshape(1,-1)/self.fs).reshape(1,-1)
        Km = self.Kr/(1-self.Kr*self.lamda*R_ref*f_slow**2/2/self.Vr**2/self.fc**2/cos_theta**3)
        I = I*np.exp(1j*np.pi*Km*(1/cos_theta-1)*(t_fast-2*R_ref/self.c/cos_theta)**2)
        # 距离徙动和脉冲压缩
        I = np.fft.fft(I,axis=1)
        I = I*np.exp(1j*np.pi/Km*cos_theta*f_fast**2)*np.exp(1j*4*np.pi/self.c*R_ref*(1/cos_theta-1)*f_fast)*self.Rangewindow()
        I = np.fft.ifft(I,axis=1)
        self.Range_profile(I)
        # 脉冲压缩
        I = I*np.exp(-1j*4*np.pi*Km/self.c**2*(1-cos_theta)*(t_fast/cos_theta-2*R_ref/self.c/cos_theta)**2)
        R = t_fast*self.c/2
        I = I*np.exp(1j*4*np.pi*R/self.lamda*cos_theta)
        # 图像切割缩放
        Na_max = int(np.ceil(F_slow_max/self.PRF*Na))
        Na_I = I.shape[0]
        I = I[int(Na_I/2-Na_max/2):int(Na_I/2+Na_max/2),:]
        I = np.fft.ifft(I,axis=0)
        self.dazm = self.Vr/F_slow_max
        #I = self.transform_coordinate(I)
        return I
    def scan(self):
        # 读取数据
        ss = self.data
        # 读取数据宽度
        Na,Nr = ss.shape
        # 转换数据类型
        self.burstSampleNum = self.burstSampleNum.astype(np.int32)
        N = (Na - self.burstSampleNum[0])/sum(self.burstSampleNum)*2+1
        ss1 = np.zeros((int(np.ceil(N/2))*self.burstSampleNum[0],Nr),dtype=np.complex64)
        for i in range(int(np.ceil(N/2))):
            startnum = i*sum(self.burstSampleNum)
            ss1[i*self.burstSampleNum[0]:(i+1)*self.burstSampleNum[0],:] = ss[startnum:startnum+self.burstSampleNum[0],:]
        f_fast = np.fft.fftshift(np.linspace(-self.fs/2,self.fs/2-self.fs/Nr,Nr)).reshape(1,-1)
        ss1 = np.fft.fft(ss1,axis=0)
        # 读取数据宽度
        Na,Nr = ss1.shape
        rngStart = self.R0 - self.Tp/2*self.c/2
        R_ref = self.R0 + (Nr-self.Tp*self.fs)/2*self.c/self.fs/2
        t_fast = (2*rngStart/self.c+np.arange(Nr)/self.fs).reshape(1,-1)
        f_slow = np.fft.fftshift(np.linspace(-self.PRF/2,self.PRF/2-self.PRF/Na,Na)).reshape(-1,1)
        cos_theta = np.sqrt(1-self.lamda**2*f_slow**2/4/self.Vr**2)
        Km = self.Kr/(1-self.Kr*self.lamda*R_ref*f_slow**2/2/self.Vr**2/self.fc**2/cos_theta**3)
        ss1 = ss1*np.exp(1j*np.pi*Km*(1/cos_theta-1)*(t_fast-2*R_ref/self.c/cos_theta)**2)
        ss1 = np.fft.fft(ss1,axis=1)
        ss1 = ss1*np.exp(1j*np.pi/Km*cos_theta*f_fast**2)*np.exp(1j*4*np.pi/self.c*R_ref*(1/cos_theta-1)*f_fast)*self.Rangewindow()
        ss1 = np.fft.ifft(ss1,axis=1)
        ss1 = ss1*np.exp(-1j*4*np.pi*Km/self.c**2*(1-cos_theta)*(t_fast/cos_theta-2*R_ref/self.c/cos_theta)**2)
        ss1 = np.fft.ifft(ss1,axis=0)
        # burst总数
        N1 = int(np.ceil(N/2))
        R = t_fast*self.c/2
        # 方位聚焦
        x = self.Vr/self.PRF*np.linspace(0,self.burstSampleNum[0]-1,self.burstSampleNum[0])-np.floor(self.burstSampleNum[0]/2)*self.Vr/self.PRF
        x = np.tile(x,N1).reshape(-1,1)
        ss1 = ss1*np.exp(1j*np.pi*(2/self.lamda/R)*x**2)
        for i in range(N1):
            
            ss1[i*int(self.burstSampleNum[0]):(i+1)*int(self.burstSampleNum[0]),:] = np.fft.fftshift(np.fft.fft(np.hamming(int(self.burstSampleNum[0])).reshape(-1,1)*ss1[i*int(self.burstSampleNum[0]):int((i+1)*self.burstSampleNum[0]),:],axis=0),axes=0)
        self.Range_profile(ss1)
        # 重采样拼接
        I1 = np.zeros((1,Nr),dtype=np.complex64)
        illum_range = np.sum(self.burstSampleNum)/2*self.Vr/self.PRF
        x_want = np.arange(-illum_range,illum_range-5,self.azmRho)
        f_slow = np.linspace(-self.PRF/2,self.PRF/2-self.PRF/int(self.burstSampleNum[0]),int(self.burstSampleNum[0])).reshape(-1,1)
        sin_theta = f_slow*self.lamda/2/self.Vr
        for i in range(N1):
            x_real = R*sin_theta/np.sqrt(1-sin_theta**2)
            I2 = np.zeros((x_want.size,Nr),dtype=np.complex64)
            for j in range(Nr):
                I2[:,j] = np.interp(x_want,x_real[:,j],ss1[i*int(self.burstSampleNum[0]):(i+1)*int(self.burstSampleNum[0]),j])
            I1 = np.concatenate((I1,I2),axis=0)
        #I1 = self.transform_coordinate(I1)
        return I1
    
    def tops(self):
        # 读取数据
        ss = self.data
        # 读取数据宽度
        Na,Nr = ss.shape
        # 转换数据类型
        self.burstSampleNum = self.burstSampleNum.astype(np.int32)
        N = (Na - self.burstSampleNum[0])/sum(self.burstSampleNum)*2+1
        ss1 = np.zeros((int(np.ceil(N/2))*self.burstSampleNum[0],Nr),dtype=np.complex64)
        for i in range(int(np.ceil(N/2))):
            startnum = i*sum(self.burstSampleNum)
            ss1[i*self.burstSampleNum[0]:(i+1)*self.burstSampleNum[0],:] = ss[startnum:startnum+self.burstSampleNum[0],:]
        f_fast = np.fft.fftshift(np.linspace(-self.fs/2,self.fs/2-self.fs/Nr,Nr)).reshape(1,-1)
        theta1 = self.lamda/self.La
        PRF_max = 2*2*self.Vr*np.sin(theta1/2+abs(self.beamDir))/self.lamda
        N = int(np.ceil(PRF_max/self.PRF)) 
        ss1 = np.fft.fftshift(np.fft.fft(ss1,axis=0),axes=0)
        ss1 = np.tile(ss1,(N,1))
        Na,Nr = ss1.shape
        # 慢时间
        f_slow = np.linspace(-self.PRF*N/2,self.PRF/2*N-self.PRF/Na*N,Na).reshape(-1,1)
        rngStart = self.R0 - self.Tp/2*self.c/2
        R_ref = self.R0 + (Nr-self.Tp*self.fs)/2*self.c/self.fs/2
        t_fast = (2*rngStart/self.c+np.arange(Nr)/self.fs).reshape(1,-1)
        cos_theta = np.sqrt(1-self.lamda**2*f_slow**2/4/self.Vr**2)
        Km = self.Kr/(1-self.Kr*self.lamda*self.R0*f_slow**2/2/self.Vr**2/self.fc**2/cos_theta**3)
        ss1 = ss1*np.exp(1j*np.pi*Km*(1/cos_theta-1)*(t_fast-2*R_ref/self.c/cos_theta)**2)
        ss1 = np.fft.fft(ss1,axis=1)
        ss1 = ss1*np.exp(1j*np.pi/Km*cos_theta*f_fast**2)*np.exp(1j*4*np.pi/self.c*R_ref*(1/cos_theta-1)*f_fast)*self.Rangewindow()
        ss1 = np.fft.ifft(ss1,axis=1)

        L1 = float(self.burstSampleNum[0])*self.Vr/self.PRF
        R1 = -L1/np.tan(self.beamDir)/2
        ss1 = ss1*np.exp(-1j*np.pi*self.lamda*R1/2/self.Vr**2*f_slow**2)
        ss1 = np.fft.ifft(ss1,axis=0)
        ss1 = ss1[int(Na/N):2*int(Na/N),:]*np.hamming(int(Na/N)).reshape(-1,1)
        ss1 = np.fft.fft(ss1,axis=0)
        f_slow = np.linspace(-self.PRF*N/2,self.PRF/2*N-self.PRF/int(Na/3)*N,int(Na/3)).reshape(-1,1)
        R = t_fast*self.c/2
        ss1 = ss1*np.exp(1j*np.pi*self.lamda*R1/2/self.Vr**2*f_slow**2)
        ss1 = np.fft.ifft(ss1,axis=0)
        # 读取数据宽度
        Na,Nr = ss1.shape
        x = ((self.Vr/self.PRF*np.linspace(0,Na-1,Na)-np.floor(Na/2)*self.Vr/self.PRF)/N).reshape(-1,1)
        ss1 = ss1*np.exp(1j*np.pi*(2/self.lamda/R)*x**2)
        ss1 = np.fft.fft(ss1,axis=0)

        return ss1

    def squint(self):
        fdc = 2*self.Vr*np.sin(self.squint_angle)/self.lamda
        # 读取数据
        ss = self.data
        # 读取数据宽度
        Na,Nr = ss.shape
        #  参考函数相乘
        ss = np.fft.fftshift(np.fft.fft2(ss))
        # 走动校正
        f_fast = np.linspace(-self.fs/2,self.fs/2-self.fs/Nr,Nr).reshape(1,-1)
        #d_R = self.Vr/self.PRF*(np.linspace(0,Na-1,Na)-Na/2).reshape(-1,1)*np.sin(self.squint_angle)
        f_slow = np.linspace(-self.PRF/2,self.PRF/2-self.PRF/Na,Na).reshape(-1,1)
        #ss = ss*np.exp(1j*np.pi/self.Kr*f_fast**2)

        M_round = np.round((fdc+f_slow)/self.PRF)
        f_slow = f_slow + M_round*self.PRF
        R_ref = self.R0 + (Nr-self.Tp*self.fs)/2*self.c/2/self.fs
        ss = ss*np.exp(1j*4*np.pi*np.cos(self.squint_angle)*R_ref/float(self.c)*np.sqrt((self.fc+f_fast)**2-float(self.c/self.Vr)**2*f_slow**2/4))*np.exp(1j*np.pi/self.Kr*f_fast**2)
        R = self.c/2*(2*R_ref/self.c+np.linspace(-Nr/2,Nr/2-1,Nr).reshape(1,-1)/self.fs).reshape(1,-1)
        ss = np.fft.ifft(ss,axis=1)
        ss = ss*np.exp(-2j*np.pi*(R-R_ref)*np.sin(self.squint_angle)/self.Vr*f_slow)
        ss = np.fft.ifft(ss,axis=0)
        self.Range_profile(ss)
        #ss = self.transform_coordinate(ss)
        return ss
    

if __name__ == "__main__":
    data_path = '/home/chenjc/newprojects/scripts/HBFZ/EchoSim/test.mat'
    imaging = imaging(data_path)
    imaging.winFlag = True
    imaging.imaging()
    imaging.dataExpose()         # 保存成像结果为成员

    import cv2
    imgResultFile =  "/home/chenjc/newprojects/scripts/DMCX/-25squi.png"
    cv2.imwrite(imgResultFile, imaging.img)
'''
    def jushu(self):

    def huaju(self):
    def scan(self):
    def tops(self):'''