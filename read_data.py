'''
文件导入程序
'''
import scipy.io as sio
import numpy as np
def load_data(path):
    '''
    该函数用来读取.mat文件
    和参数文件
    '''
    data = sio.loadmat(path)
    # 用字典保存下面的变量
    c = data['c'][0][0]
    fc = data['fc'][0][0]
    # 波长
    lamda = c/fc
    # 脉宽
    Tp = data['Tp'][0][0]
    # 雷达收发间隔
    PRF = data['PRF'][0][0]
    # 采样率
    fs = data['fs'][0][0]
    # 带宽
    Br = data['B'][0][0]
    # 线性调频率
    Kr = Br/Tp
    # 开始采样时间
    R0 = data['rngStart'][0][0]
    # 飞行速度
    Vr = data['V'][0][0]
    # 天线长度
    La = data['La'][0][0]
    # 方位向范围
    azmScale = data['azmScale'][0][0]
    # 距离向范围
    rngScale = data['rngScale'][0][0]
    # 数据模式
    scanMode = data['scanMode'][0][0]
    # 存储数据
    data = np.array(data['data'],dtype=np.complex64)
    # 存储雷达参数
    #radar_parameters = {'c':c,'fc':fc,'Kr':Kr,'Tp':Tp,'PRF':PRF,'fs':fs,'lamda':lamda,'Br':Br,'R0':R0,'Vr':Vr,'La':La,'azmScale':azmScale,'rngScale':rngScale}
    radar_parameters = {'c':c,'fc':fc,'Kr':Kr,'Tp':Tp,'PRF':PRF,'fs':fs,'lamda':lamda,'Br':Br,'R0':R0,'Vr':Vr,'La':La,'scanMode':scanMode,'azmScale':azmScale,'rngScale':rngScale}
    return data,radar_parameters