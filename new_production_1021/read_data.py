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
    # 距离分辨率
    rngRho = data['rngRho'][0][0]
    # 方位分辨率
    azmRho = data['azmRho'][0][0]
    # 下视角
    offNadir = data['offNadiAng'][0][0]
    # burst采样点数
    burstSampleNum = data['burstSampleNum'][:,0]
    # 目标名称
    # target = data['target'][0][0]
    if scanMode == 4 or scanMode == 5:
        burstSampleNum = data['burstSampleNum'][:,0]
    # 斜视角
    if scanMode == 6:
        squint_angle = data['squiAng'][0][0]
    # 波束角度
    beamDir = np.array(data['beamDir'])[0][0]
    # 轨迹
    satTrack = np.array(data['satTrack'])
    # 存储数据
    data = np.array(data['data'],dtype=np.complex64)
    # 存储雷达参数
    #radar_parameters = {'c':c,'fc':fc,'Kr':Kr,'Tp':Tp,'PRF':PRF,'fs':fs,'lamda':lamda,'Br':Br,'R0':R0,'Vr':Vr,'La':La,'azmScale':azmScale,'rngScale':rngScale}
    radar_parameters = {'c':c,'fc':fc,'Kr':Kr,'Tp':Tp,'PRF':PRF,'fs':fs,'lamda':lamda,'Br':Br,'R0':R0,'Vr':Vr,\
                        'La':La,'scanMode':scanMode,'azmScale':azmScale,'rngScale':rngScale,'rngRho': rngRho,\
                        'azmRho': azmRho, 'burstSampleNum':burstSampleNum, 'offNadir':offNadir}
    if scanMode == 4 or scanMode == 5:
        radar_parameters = {'c':c,'fc':fc,'Kr':Kr,'Tp':Tp,'PRF':PRF,'fs':fs,'lamda':lamda,'Br':Br,'R0':R0,'Vr':Vr,
                            'La':La,'scanMode':scanMode,'azmScale':azmScale,'rngScale':rngScale, 'rngRho': rngRho, 'azmRho':azmRho,'burstSampleNum':burstSampleNum,'offNadir':offNadir,'beamDir':beamDir,'satTrack':satTrack}
    if scanMode == 6:
        radar_parameters = {'c':c,'fc':fc,'Kr':Kr,'Tp':Tp,'PRF':PRF,'fs':fs,'lamda':lamda,'Br':Br,'R0':R0,'Vr':Vr,
                        'La':La,'scanMode':scanMode,'azmScale':azmScale,'rngScale':rngScale, 'rngRho': rngRho, 'azmRho':azmRho,'offNadir':offNadir,'beamDir':beamDir,'squint_angle':squint_angle,'satTrack':satTrack}
    return data,radar_parameters