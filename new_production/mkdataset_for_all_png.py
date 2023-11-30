import numpy as np
from utils import dictToXmlFile, xmlFileToDict, CurrTime, addGeoInfoToTiff
from dataProcess import Extractor # 11-18更新为data Process 与lij的代码合并
import tifffile as tiff
import subprocess
import os
import shutil
from txtProcess import * # 电磁建模结果处理文件
import scipy.io as sio # 后面三个为成像库
from read_data import * 
from Five_mode_imaging_V2 import *
import matplotlib.image as mpimg
from utils import zip_folder

def mkdataset(model_name, model_info_folder, model_xml_file, current_save_folder, current_mid_folder, distributionx, distributiony, RayH, RayW,
              target_type, x_cut, y_cut, scanMode, Rho, beta_range, beta_azimuth, wave_band, polarization, squiAng):

    # 参数设置

    if wave_band == 'X':
        fc = 9.6e9
    else:
        fc = 6e9
        
    rngRho = Rho
    azmRho = Rho

    # 初始化文件目录
    if os.path.exists(current_save_folder):
        shutil.rmtree(current_save_folder)
    os.makedirs(current_save_folder)

    # 生成缓存文件夹
    if os.path.exists(current_mid_folder):
        shutil.rmtree(current_mid_folder)
    save_mid_result_folder = current_mid_folder
    os.makedirs(save_mid_result_folder)

    # 暂存变量，不用进行更改，方便程序运行
    save_pov_folder_name = save_mid_result_folder + '/pov/'
    save_txt_folder_name = save_mid_result_folder + '/txt/'
    save_mat_folder_name =  save_mid_result_folder + '/mat/'
    save_echo_folder_name =  save_mid_result_folder + '/echo/'
    save_img_folder_name = save_mid_result_folder + '/img/'

    # 生成缓存文件夹
    os.makedirs(save_pov_folder_name)
    os.makedirs(save_txt_folder_name)
    os.makedirs(save_mat_folder_name)
    os.makedirs(save_echo_folder_name)
    os.makedirs(save_img_folder_name)

    print('开始生产目标：' + model_name + '，目标类型：' + target_type + '，极化：' + polarization + '，波段：' + wave_band + '，分辨率：' + str(Rho))
    povfilename = model_info_folder + model_name +'.pov'
    xmlfilename = model_info_folder + model_name +'.xml'

    # 取出几何模型
    with open(povfilename,'r',encoding='utf-8') as f1:
        lines = []
        for line in f1.readlines():
            lines += [line]
    f1.close()

    # 设置回波仿真参数
    workMode = 1        # 工作体制（单通道）
    simSatMode = 1      # 卫星解算方式（自定义）
    echoMethod = 1      # 回波生成方式（快速fft）
    initParams = {"scanMode": scanMode,\
                "workMode": workMode,\
                "rngRho": rngRho,\
                "azmRho": azmRho,\
                "simSatMode": simSatMode,\
                "echoMethod": echoMethod}
    
    # 其他参数
    squiAng = 0        # 斜视情况所需参数
    threadNum = 48     # CPU并行线程数
    sightRange = 1e4   # 卫星与场景的最短距离，自定义和实际轨道均有效，此时仿真轨道非实际轨道
    targetV = np.array([0, 0, 0], dtype="double").reshape(-1,1)       # 将速度矢量转化为np列向量

    # 误差参数
    echoWinStartEdge = 0;       # 回波窗起始沿误差 ns级, 正值对应信号前移
    sysDelay = 0;               # 系统延时测量误差        正值对应信号右移
    innoPhaseErr = 0;            # 电离层主要引起相位误差  弧度
    tropPhaseErr = 0;            # 对流层主要引起时延以及幅度下降
    tropAmpDec = 1;              # 幅度下降因子

    addonParams = {
        "squiAng": squiAng,
        "threadNum": threadNum,
        "sightRange": sightRange,
        "targetV": targetV,
        "echoWinStartEdge": echoWinStartEdge,
        "sysDelay": sysDelay,
        "innoPhaseErr": innoPhaseErr,
        "tropPhaseErr": tropPhaseErr,
        "tropAmpDec": tropAmpDec
    }
    initParams.update(addonParams)

    # 根据不同模式设置天线长度
    if scanMode == 1:
        La = 2*azmRho
    elif scanMode == 2:
        La = 3
    elif scanMode == 3:
        La = 4
    elif scanMode == 4:
        La = 4
    elif scanMode == 5:
        La = 4
    elif scanMode == 6:
        La = 2*azmRho

    # 距离扫描所需参数
    subStripNum = 2
    subStripRng = 25e3
    # 多通道时所需参数
    channelNum = 8

    extra = {"La": La, "subStripNum": subStripNum, "subStripRng": subStripRng,\
            "channelNum": channelNum}

    # 轨道参数设置
    orbital = {"majAxis": 6893.38359077456e3,\
            "eccRatio": 0.0013485366190770,\
            "incAng": 97.4477903825153,\
            "ascdLng": 295.304778902499,\
            "perigee": 66.7207893275409}

    # 场景约束
    scn = {"tgLat": 28.06194738,\
           "tgLng": 112.89267163,\
           "tgHeight": 54.8}

    # 雷达参数
    radar = {"c": 3e8, "fc": fc, "Tp": 5e-6, "riseRatio": 16}

    initParams.update(extra)
    initParams.update(orbital)
    initParams.update(scn)
    initParams.update(radar)

    # 回波仿真环境
    echoSimPath = "/home/chenjc/newprojects/scripts/HBFZ/EchoSimProd/build"
    # 中间参数文件结果
    settingsPath = save_mid_result_folder + '/settings.mat'        

    # 俯仰角设置
    for incidentAngle in range(17,56,70): # 俯仰角（18，56，5）（改）
        # 根据下视角和方位角计算相机位置
        # 相机初始位置,即方位角为0时
        Z = 5000*np.cos(incidentAngle/180 * np.pi)
        Y = 5000*np.sin(incidentAngle/180 * np.pi)
        # 调整场景大小、射线数量以及裁剪大小
        distributiony_change = np.sqrt((distributiony+1e6/np.cos(incidentAngle/180 * np.pi))**2-(1e6)**2)-1e6/np.cos(incidentAngle/180 * np.pi)*np.sin(incidentAngle/180 * np.pi)
        RayH_change = np.sqrt((RayH+1e6/np.cos(incidentAngle/180 * np.pi))**2-(1e6)**2)-1e6/np.cos(incidentAngle/180 * np.pi)*np.sin(incidentAngle/180 * np.pi)
        x_cut_change = np.sqrt((x_cut+1e6/np.cos(incidentAngle/180 * np.pi))**2-(1e6)**2)-1e6/np.cos(incidentAngle/180 * np.pi)*np.sin(incidentAngle/180 * np.pi)
        
        for j in range(0,36,10): # 方位角0-360度 20度一个（改）
            # 方位角设置
            current_degree = j*10 # 方位角间隔2*10度（改）
            # print('current degree:'+str(current_degree));
            # 设置存储文件
            current_name = str(incidentAngle)+'_'+str(current_degree)

            # 步骤1：初始化pov、xml文件

            '''
            update：2023-11-19
            lij原版代码：Y0 = Y*np.cos((current_degree+90)/180 * np.pi)
            现在逆时针转回90度（和调试代码保持一致）
            '''

            # 当前角度
            Z0 = Z
            Y0 = Y*np.cos((current_degree+90)/180 * np.pi)
            X0 = Y*np.sin((current_degree+90)/180 * np.pi)*(-1)

            # pov
            # 当前角度
            Z0 = Z
            Y0 = Y*np.cos((current_degree+90)/180 * np.pi)
            X0 = Y*np.sin((current_degree+90)/180 * np.pi)*(-1)
            # 初始化pov文件
            content = ['#include "colors.inc"\n#include "finish.inc"\nglobal_settings{SAR_Output_Data 1 SAR_Intersection 1}\n']
            camera = ['#declare Cam = camera {\northographic\n']
            cameraPos = 'location <' + str(X0) + ',' + str(Z0) + ',' + str(Y0) + '>' + '\n'
            camera.append(cameraPos)
            camera.append('look_at < 0 , 0 , 0 >\n')
            camera.append('right ' + str(distributionx) + '*x\n')
            camera.append('up ' + str(distributiony_change) + '*y\n')
            camera.append('}\n')
            content = content + camera
            content.append('camera{Cam}\n')
            content.append('light_source {\n0*x\ncolor rgb <1,1,1>\nparallel\ntranslate <'+str(X0)+','+str(Z0)+','+str(Y0)+'>\npoint_at < 0 , 0 , 0 >\n}\n')
            content.append('plane {\n<0,1,0>\n0\ntexture {\npigment { color rgb<1, 1, 1> }\nfinish {reflection {0.2} ambient 0 diffuse 0.001 specular 0.5 roughness 0.0033}\n}\n}\n')
            # 存储pov文件
            content = content + lines
            savefilename = save_pov_folder_name+current_name+'.pov'
            with open(savefilename,'w',encoding='utf-8') as f:
                f.writelines(content)
            f.close()

            # 步骤2：对pov文件进行电磁建模
            runCommand = 'povray '+ savefilename +' -D +W'+str(int(1.2*RayW))+' +H'+str(int(1.2*RayH_change))
            result = subprocess.run(runCommand, shell=True, capture_output=True, text=True, cwd=save_txt_folder_name, check=True)

            reviseFilenameCommand = 'mv Contributions.txt ' + current_name + '.txt'
            result = subprocess.run(reviseFilenameCommand, shell=True, capture_output=True, text=True, cwd=save_txt_folder_name, check=True)

            # 步骤3：对建模结果进行裁剪处理
            current_txt_name = save_txt_folder_name+current_name+'.txt'
            txtProParams = [current_degree,incidentAngle, squiAng, target_type, polarization, x_cut_change, y_cut, current_txt_name]
            elecResult = txtProcess(txtProParams)
            savedic = {"data": elecResult, "offNadiAng": incidentAngle}
            modelingResultPath = save_mat_folder_name + current_name + '.mat'
            sio.savemat(modelingResultPath, savedic)

            # 步骤4：进行回波仿真
            echoResultPath = save_echo_folder_name + current_name + '.mat'
            # 这一步一旦散射矩阵被当做整数解析就会出问题！
            modelResult = sio.loadmat(modelingResultPath)
            modelResult["data"] = np.array(modelResult["data"], dtype="double")
            initParams.update(modelResult)
            # 保存参数
            sio.savemat(settingsPath, initParams)
            # 运行回波程序
            try:
                print("开始回波仿真...")
                echoSimCmd = ["./main", settingsPath, echoResultPath]
                result = subprocess.run(echoSimCmd, capture_output=False, text=True, check=True, cwd=echoSimPath)       # 屏蔽输出避免跨平台开发导致的编码问题
                # print(result.stdout)
            except Exception as e:
                # print(e)
                print("回波仿真异常！")
                raise
            else:
                print("回波仿真完成！")
            
            # 步骤5：进行成像处理并存为.png文件
            save_png_path = current_save_folder + '/' + current_name + '.png'
            # 成像
            imaging_current = imaging(echoResultPath, beta_range, beta_azimuth, model_name, target_type, Rho) #
            imaging_current.imaging()
            imaging_current.dataExpose(model_name,2) # 保存成像结果为成员
            mpimg.imsave(save_png_path, np.array(imaging_current.img, dtype=np.uint8), cmap=plt.cm.gray)   