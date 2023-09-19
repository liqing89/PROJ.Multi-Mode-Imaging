import tqdm
import numpy as np
import subprocess
import os
import shutil
from txtProcess import * # 电磁建模结果处理文件
import scipy.io as sio # 后面三个为成像库
from read_data import * 
from Five_mode_imaging import *
import matplotlib.image as mpimg

def mkdataset(model_name, distribution, Ray, target, cut, scanMode, Rho, POV_folder, folder, beta):

    # 参数设置
    distributionx = distribution
    distributiony = distribution

    RayH = Ray
    RayW = Ray

    x_cut = cut
    y_cut = cut

    rngRho = Rho
    azmRho = Rho

    main_folder = folder + model_name

    # 暂存变量，不用进行更改，方便程序运行
    if os.path.exists(main_folder):
        shutil.rmtree(main_folder)
    mk_save_main_floder = 'mkdir ' + model_name
    result = subprocess.run(mk_save_main_floder, shell=True, capture_output=True, text=True, cwd=folder, check=True)

    save_pov_folder_name = main_folder + '/pov/'
    save_txt_folder_name = main_folder + '/txt/'
    save_mat_folder_name =  main_folder + '/mat/'
    save_echo_folder_name =  main_folder + '/echo/'
    save_img_folder_name = main_folder + '/img/'
    save_png_folder_name = main_folder + '/png/'

    # 步骤1：初始化多俯仰角、多方位角pov文件
    print('开始生产目标：' + model_name + '，目标类型：' + target)
    filename = POV_folder + model_name +'.pov'

    if os.path.exists(save_pov_folder_name):
        shutil.rmtree(save_pov_folder_name)
    mk_save_pov_floder = 'mkdir pov'
    result = subprocess.run(mk_save_pov_floder, shell=True, capture_output=True, text=True, cwd=main_folder, check=True)

    for incidentAngle in range(20,55,5):
        # 俯仰角设置

        # 根据下视角和方位角计算相机位置
        # 相机初始位置,即方位角为0时
        Z = 700*np.cos(incidentAngle/180 * np.pi)
        Y = 700*np.sin(incidentAngle/180 * np.pi)
        # 调整场景大小
        distributiony_change = np.sqrt((distributiony+1e6/np.cos(incidentAngle/180 * np.pi))**2-(1e6)**2)-1e6/np.cos(incidentAngle/180 * np.pi)*np.sin(incidentAngle/180 * np.pi)
        
        # 方位角设置
        for j in range(0,18,18):
            current_main_degree = j*20
            flag = 0
            # for i in range(current_main_degree*2-5,current_main_degree*2+6):
            for i in range(1):
                #当前角度
                current_degree = current_main_degree
                Z0 = Z
                Y0 = Y*np.cos(current_degree/180 * np.pi)
                X0 = Y*np.sin(current_degree/180 * np.pi)*(-1)

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

                with open(filename,'r',encoding='utf-8') as f1:
                    lines = []
                    for line in f1.readlines():
                        lines += [line]
                f1.close()

                content = content + lines

                savefilename = save_pov_folder_name+str(incidentAngle)+'_'+str(current_main_degree)+'_'+ str(flag) +'.pov'

                with open(savefilename,'w',encoding='utf-8') as f:
                    f.writelines(content)
                f.close()
                flag += 1
            print('已完成俯仰角 {} 度, 方位角 {} 度的pov文件初始化。'.format(incidentAngle, 20*j))

    # 步骤2：对pov文件进行电磁建模
    if os.path.exists(save_txt_folder_name):
        shutil.rmtree(save_txt_folder_name)
    mk_save_txt_floder = 'mkdir txt'
    result = subprocess.run(mk_save_txt_floder, shell=True, capture_output=True, text=True, cwd=main_folder, check=True)

    for k in range(20,55,5):
        RayH_change = np.sqrt((RayH+1e6/np.cos(k/180 * np.pi))**2-(1e6)**2)-1e6/np.cos(k/180 * np.pi)*np.sin(k/180 * np.pi)
        for i in range(0,18,18):
            main_degree = 20*i
            current_name_prefix = save_pov_folder_name+str(k)+'_'+str(main_degree)
            for j in range(1):
                current_file_name = current_name_prefix + '_' + str(j) + '.pov'

                runCommand = 'povray '+ current_file_name +' -D +W'+str(int(1.2*RayW))+' +H'+str(int(1.2*RayH_change))
                result = subprocess.run(runCommand, shell=True, capture_output=True, text=True, cwd=save_txt_folder_name, check=True)

                reviseFilenameCommand = 'mv Contributions.txt Contributions_'+str(k)+'_'+str(main_degree) + '_' + str(j) + '.txt'
                result = subprocess.run(reviseFilenameCommand, shell=True, capture_output=True, text=True, cwd=save_txt_folder_name, check=True)
            print('已完成俯仰角 {} 度, 方位角 {} 度的电磁建模。'.format(k, 20*i))

    # 步骤3：对建模结果进行裁剪处理
    if os.path.exists(save_mat_folder_name):
        shutil.rmtree(save_mat_folder_name)
    mk_save_mat_floder = 'mkdir mat'
    result = subprocess.run(mk_save_mat_floder, shell=True, capture_output=True, text=True, cwd=main_folder, check=True)

    for k in range(20,55,5):
        x_cut_change = np.sqrt((x_cut+1e6/np.cos(k/180 * np.pi))**2-(1e6)**2)-1e6/np.cos(k/180 * np.pi)*np.sin(k/180 * np.pi)
        for i in range(0,18,18):
            main_degree = 20*i
            current_name_prefix = save_txt_folder_name + 'Contributions_' + str(k) + '_' + str(main_degree)
            for j in range(1):
                current_file_name = current_name_prefix + '_' + str(j) + '.txt'
                txtProParams = [k, 0, target, 'HH', x_cut_change, y_cut, current_file_name]
                elecResult = txtProcess(txtProParams)
                savedic = {"data": elecResult, "offNadiAng": k}
                resultMatFile = "{}mat_{}_{}_{}.mat".format(save_mat_folder_name, k, main_degree, j)
                sio.savemat(resultMatFile, savedic)
            print('已完成俯仰角 {} 度, 方位角 {} 度的电磁建模裁剪工作。'.format(k, 20*i))
    
    # 步骤4：进行回波仿真
    if os.path.exists(save_echo_folder_name):
        shutil.rmtree(save_echo_folder_name)
    mk_save_echo_floder = 'mkdir echo'
    result = subprocess.run(mk_save_echo_floder, shell=True, capture_output=True, text=True, cwd=main_folder, check=True)

    for k in range(20,55,5):
        for i in range(0,18,18):
            for j in range(1):
                main_degree = 20*i
                modelingResultPath = save_mat_folder_name + 'mat_' + str(k) + '_' + str(main_degree) + '_' + str(j) + '.mat'
                echoResultPath = save_echo_folder_name + 'echo_' + str(k) + '_' + str(main_degree) + '_' + str(j) + '.mat'
                settingsPath = main_folder + '/settings.mat'        # 中间参数文件结果

                workMode = 1        # 工作体制（单通道）
                simSatMode = 1      # 卫星解算方式（自定义）
                echoMethod = 1      # 回波生成方式（快速fft）
                initParams = {"scanMode": scanMode,\
                            "workMode": workMode,\
                            "rngRho": rngRho,\
                            "azmRho": azmRho,\
                            "simSatMode": simSatMode,\
                            "echoMethod": echoMethod}

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
                # 斜视情况所需参数
                squiAng = 30

                extra = {"La": La, "subStripNum": subStripNum, "subStripRng": subStripRng,\
                        "channelNum": channelNum, "squiAng": squiAng}

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
                radar = {"c": 3e8, "fc": 9.6e9, "Tp": 5e-6, "riseRatio": 16}

                initParams.update(extra)
                initParams.update(orbital)
                initParams.update(scn)
                initParams.update(radar)

                # 这一步一旦散射矩阵被当做整数解析就会出问题！
                modelResult = sio.loadmat(modelingResultPath)
                modelResult["data"] = np.array(modelResult["data"], dtype="double")
                initParams.update(modelResult)

                # 保存参数
                sio.savemat(settingsPath, initParams)

                echoSimPath = "/home/chenjc/newprojects/EchoSim/EchoSimV2/build"
                # 运行回波程序
                try:
                    print("开始回波仿真...")
                    echoSimCmd = ["./main", settingsPath, echoResultPath]
                    result = subprocess.run(echoSimCmd, capture_output=True, text=True, check=True, cwd=echoSimPath)
                    print(result.stdout)
                except Exception as e:
                    print(e)
                    print("回波仿真异常！")
                    raise
                else:
                    print("回波仿真完成！")
    
    # 步骤5：成像处理
    if os.path.exists(save_img_folder_name):
        shutil.rmtree(save_img_folder_name)
    mk_save_img_floder = 'mkdir img'
    result = subprocess.run(mk_save_img_floder, shell=True, capture_output=True, text=True, cwd=main_folder, check=True)

    for k in range(20,55,5):
        for i in range(0,18,18):
            for j in range(1):
                main_degree = 20*i
                data_path = save_echo_folder_name + 'echo_' + str(k) + '_' + str(main_degree) + '_' + str(j) + '.mat'
                save_path = save_img_folder_name + 'img_' + str(k) + '_' + str(main_degree) + '_' + str(j) + '.mat'
                imaging_current = imaging(data_path, beta)
                imaging_current.imaging()
                imaging_current.show_image(2, save_path)
            print('已完成俯仰角 {} 度, 方位角 {} 度的回波成像工作。'.format(k, 20*i))

    # 步骤6：将成像结果存为png文件
    if os.path.exists(save_png_folder_name):
        shutil.rmtree(save_png_folder_name)
    mk_save_png_floder = 'mkdir png'
    result = subprocess.run(mk_save_png_floder, shell=True, capture_output=True, text=True, cwd=main_folder, check=True)

    for k in range(20,55,5):
        for i in range(0,18,18):
            for j in range(1):
                main_degree = 20*i
                data_path = save_img_folder_name + 'img_' + str(k) + '_' + str(main_degree) + '_' + str(j) + '.mat'
                save_path = save_png_folder_name + 'png_' + str(k) + '_' + str(main_degree) + '_' + str(j) + '.png'
                data = sio.loadmat(data_path)['image']
                # 裁剪
                l = int(data.shape[1]/2 - data.shape[0]/2*0.8); r = int(data.shape[1]/2 + data.shape[0]/2*0.8)
                u = int(data.shape[0]*0.1+1); d = int(data.shape[0]*0.9)
                image = np.array(data[u:d,l:r], dtype=np.uint8)
                mpimg.imsave(save_path, image, cmap=plt.cm.gray)
            print('已完成俯仰角 {} 度, 方位角 {} 度的成像结果储存。'.format(k, 20*i))