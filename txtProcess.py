#!/usr/bin/python3
# -*- coding: UTF-8 -*-
import numpy as np
'''
------------------------------------
该脚本用于对RaySAR电磁建模结果进行后处理
作者:邓嘉
日期:2023/04/23
版本:V1.0

日期:2023/06/13
版本:V2.0

日期:2023/08/24
版本:V3.0
------------------------------------
'''

'''
------------------------------------
输入参数部分
------------------------------------
'''
def processForSingle(index_single, X, Y, Z, Intens, povname, IncidentAngle,Azimuth,filename):
    # 输入参数 一次散射点的下标 处理好的电磁建模结果
    X1 = X[index_single]
    Y1 = Y[index_single]
    Z1 = Z[index_single]
    Intens1 = Intens[index_single]
    # print(max(Intens1), min(Intens1))
    if povname == 'JC' or povname == 'HM': # 舰船 / 航母
        for i in range(len(Z1)):
            if Z1[i] < 0.5:
                # 如果这个点是地面的话，散射强度下降为0.00016，如果是目标的话，目标散射强度下降1/4
                Intens1[i] = (1+np.random.rand(1)*0.5)*0.004
            else:
                if Intens1[i] < 0.04:
                    Intens1[i] = (1+np.random.rand(1)*0.2)*0.01
                else:
                    Intens1[i] = Intens1[i]/4 # (1+np.random.rand(1)*0.1)*0.1

    elif povname == 'TK': # 坦克
        background = 0.005 
        vz = max(Z1);
        for i in range(len(Z1)):
            if Z1[i] < 0.5:
                # 如果这个点是地面的话，散射强度下降为0.00016，如果是目标的话，目标散射强度下降1/4
                Intens1[i] = (1+np.random.rand(1)*0.5)*background
            elif Z1[i] < vz:
                if Intens1[i] < 0.32:
                    Intens1[i] = (1+np.random.rand(1)*0.2)*0.08
                else:
                    Intens1[i] = Intens1[i]/4
            else:
                if Intens1[i] < 0.64:
                    Intens1[i] = (1+np.random.rand(1)*0.2)*0.16
                else:
                    Intens1[i] = Intens1[i]/4

    elif povname == 'FJ': # 飞机一次散射
        if "B2" in filename: # B2 隐身飞机 一次散射  
            for i in range(len(Z1)):
                if Z1[i] < 1: # ground
                    Intens1[i] =  0.1 * Intens1[i] * np.cos(50/180*np.pi)/np.cos(IncidentAngle/180*np.pi)  #自适应背景
                else: # plane
                    Intens1[i] =  0.05 * Intens1[i] * np.cos(50/180*np.pi)/np.cos(IncidentAngle/180*np.pi)  #自适应背景


        elif "F16" in filename or "F18" in filename or "EA18G" in filename: # 小型战斗机一次散射
            count_max = 0;
            for i in range(len(Z1)):
                beta = 0.8 # 幂次
                if Azimuth == 0: #机头正对来波
                    if X1[i] > -20 and X1[i] < -5 and Y1[i] > -5 and Y1[i] < 5 and Z1[i]>1 and count_max< 15 : # 锁定机头
                        count_max = count_max + 1;
                        cos_incident = np.cos(IncidentAngle) # 20度俯仰角最强 依次减弱
                        Intens1[i] = Intens1[i] * 0.1 * (cos_incident)  /  ( Intens1[i] ** (beta) ) 
                    else: # 机头强散射点外一次散射
                        if Z1[i] < 1: # ground
                            Intens1[i] =  0.5 * Intens1[i] * np.cos(50/180*np.pi)/np.cos(IncidentAngle/180*np.pi)  #自适应背景
                        else: # plane
                            Intens1[i] =  0.2 * Intens1[i] * np.cos(50/180*np.pi)/np.cos(IncidentAngle/180*np.pi)  #自适应背景
                else: # 机头非正对来波
                    if Z1[i] < 1: # ground
                        Intens1[i] =  0.5 * Intens1[i] * np.cos(50/180*np.pi)/np.cos(IncidentAngle/180*np.pi)  #自适应背景
                    else: # plane
                        Intens1[i] =  0.2 * Intens1[i] * np.cos(50/180*np.pi)/np.cos(IncidentAngle/180*np.pi)  #自适应背景

        elif "F22" in filename or "F35" in filename: # F22和F35 隐身战机 一次散射
            for i in range(len(Z1)):
                if Z1[i] < 1: # ground
                    Intens1[i] =  0.2 * Intens1[i] * np.cos(50/180*np.pi)/np.cos(IncidentAngle/180*np.pi)  #自适应背景
                else: # plane
                    Intens1[i] =  0.1 * Intens1[i] * np.cos(50/180*np.pi)/np.cos(IncidentAngle/180*np.pi)  #自适应背景


        else: # 普通飞机一次散射
            for i in range(len(Z1)):
                if Z1[i] < 1: # ground
                    Intens1[i] =  1 * Intens1[i] * np.cos(50/180*np.pi)/np.cos(IncidentAngle/180*np.pi)  #自适应背景
                else: # plane
                    Intens1[i] =  0.8 * Intens1[i] * np.cos(50/180*np.pi)/np.cos(IncidentAngle/180*np.pi)  #自适应背景

       
    else:
        pass 
    return X1, Y1, Z1, Intens1

def processForDouble(index_double, X, Y, Z, Intens, povname,IncidentAngle,Azimuth,filename):
    X2 = X[index_double]
    Y2 = Y[index_double]
    Z2 = Z[index_double]
    Intens2 = Intens[index_double]
    # print(max(Intens2), min(Intens2))
    if povname == 'JC' or povname == 'HM': # 舰船二次散射的处理方式
        for i in range(len(Intens2)):
            if Z2[i] < 0.5:
                Intens2[i] = (1+np.random.rand(1)*0.5)*0.004
            else:
                if Intens2[i] < 0.05:
                    Intens2[i] = (1+np.random.rand(1)*0.2)*0.08
                else:
                    Intens2[i] = (1+np.random.rand(1)*0.2)*0.8

    elif povname == 'TK':    # 坦克二次散射的处理方式
        
        for i in range(len(Intens2)):
            if Z2[i] < 0.5:
                Intens2[i] = (1+np.random.rand(1)*0.5)*0.008
            else:
                if Intens2[i] < 0.05:
                    Intens2[i] = (1+np.random.rand(1)*0.2)*0.05
                else:
                    Intens2[i] = (1+np.random.rand(1)*0.2)*0.35

    elif povname == 'FJ':  # 分别讨论三类隐身飞机：B2 / F22 / F35 的二次散射
        
        if "B2" in filename: # 1. B2隐身飞机二次散射的处理方式
            count_max = 0; # 限制高亮点次数
            for i in range(len(Intens2)):

            # 左手系旋转矩阵
                R = np.array([[np.cos(-Azimuth*np.pi/180), -np.sin(-Azimuth*np.pi/180)],
                                [np.sin(-Azimuth*np.pi/180), np.cos(-Azimuth*np.pi/180)]])
                        
                [X2_origin,Y2_origin] = np.dot(R, np.array([X2[i], Y2[i]]))

                # 电磁散射
                beta = 0.8 # 幂次
                if X2_origin > 0 and X2_origin < 10 and Y2_origin > -10 and Y2_origin < 10: # 锁定排气管,x-axis指向机头
                    if count_max < 2:
                        Intens2[i] = Intens2[i] * 0.05 /  ( Intens2[i] ** (beta) ) 
                        count_max = count_max + 1
                    else:
                        Intens2[i] = Intens2[i] * 0.015 /  ( Intens2[i] ** (beta) )
                else:
                    pass

        elif "B1B" in filename:
            count_max = 0; # 限制高亮点次数
            beta = 0.8;
            for i in range(len(Z2)):
                if Z2[i] < 1: # ground
                    Intens2[i] = Intens2[i] * 0.06 /  ( Intens2[i] ** (beta) ) # 0.06->0.08
                else:  # plane
                    if count_max < 10:   # 限制高亮点次数
                        Intens2[i] = Intens2[i] * 0.1 /  ( Intens2[i] ** (beta) ) # 0.06->0.08
                        count_max = count_max + 1
                    else:
                        Intens2[i] = Intens2[i] * 0.06 /  ( Intens2[i] ** (beta) ) # 0.06->0.08

        
        elif "F16" in filename or "F18" in filename or "EA18G" in filename: # 小型战斗机二次散射的处理方式
            for i in range(len(Intens2)):
                    Intens2[i] =  Intens2[i] * 0.1

        elif "F22" in filename or "F35" in filename: # F22/F35隐身飞机二次散射的处理方式
            count_max = 0;
            for i in range(len(Intens2)):
                if count_max<10:
                    count_max = count_max + 1;
                    Intens2[i] =  Intens2[i] * 0.5
                else:
                    Intens2[i] =  Intens2[i] * 0.1
        
        else: # 4. 普通飞机二次散射的处理方式
            for i in range(len(Intens2)):
                beta = 0.8
                Intens2[i] = Intens2[i] * 0.06 /  ( Intens2[i] ** (beta) ) # 0.06->0.08
                # 散射强度低的时候 这里不能加入随机数 会引起类相干斑的躁点

            

    else:  # JC/FJ/TK 讨论结束
        pass

    return X2, Y2, Z2, Intens2

def  processForTriple(index_Triple, X, Y, Z, Intens, povname,IncidentAngle,filename):
    X3 = X[index_Triple]
    Y3 = Y[index_Triple]
    Z3 = Z[index_Triple]
    Intens3 = Intens[index_Triple]
    # print(max(Intens3), min(Intens3))
    if povname == 'JC' or povname == 'HM':
        # 舰船三次散射的处理方式
        # Intens3 = Intens[index_Triple]+max(Intens[index_Triple])
        # for j in range(len(Intens3)):
        #     if Intens3[j] < 0.003 and Z3[j] > 1:
        #         Intens3[j] = 0.003 + np.random.rand(1)*0.0005
        #     else:
        #         Intens3[j] = 0.00016
        Intens3 = Intens[index_Triple] + max(Intens[index_Triple])
        for j in range(len(Intens3)):
            if Z3[j] > 0.5:
                if Intens3[j] < 0.09:
                    Intens3[j] = (1+np.random.rand(1)*0.1)*0.09
                else:
                    Intens3[j] = (1+np.random.rand(1)*0.1)*0.15
            else:
                Intens3[j] = (1+np.random.rand(1)*0.5)*0.004
    elif povname == 'TK':
        # 坦克三次散射的处理方式
        Intens3 = Intens[index_Triple] + max(Intens[index_Triple])
        for j in range(len(Intens3)):
            if Z3[j] > 0.5:
                if Intens3[j] < 0.05:
                    Intens3[j] = (1+np.random.rand(1)*0.1)*0.09
                else:
                    Intens3[j] = (1+np.random.rand(1)*0.1)*0.15
            else:
                Intens3[j] = (1+np.random.rand(1)*0.5)*0.008
    elif povname == 'FJ':
        if "F22" in filename or "F35" in filename or "EA18G" in filename: #小型战斗机三次散射
            for j in range(len(Intens3)):
                Intens3[j] =  1 * Intens3[j] * np.cos(50/180*np.pi)/np.cos(IncidentAngle/180*np.pi) ;
        else:
            pass
    
    else:
        pass
    return X3, Y3, Z3, Intens3
    
def txtProcess(params):
    
    Azimuth,IncidentAngle,squiAng,target,polarMethod,rayXScale,rayYScale,filename = params
    
    # 注：这里删掉了X_scale和Y_scale，不再由用户控制场景大小，而是通过目标类型来设置场景大小

    X_cut_min = -rayXScale/2.0
    X_cut_max = rayXScale/2.0
    Y_cut_min = -rayYScale/2.0
    Y_cut_max = rayYScale/2.0

    '''
    ------------------------------------
    数据处理部分
    ------------------------------------
    '''
    import numpy as np
    from collections import Counter

    data = np.loadtxt(filename, dtype=np.float32)
    Az = np.array(data[:,0])
    Ra = np.array(data[:,1])
    El = np.array(data[:,2])
    Intens = np.array(data[:,3])
    Flag = np.array(data[:,4])

    # 删除散射点强度为0的点
    index_no0 = np.array(np.where(Intens!=0))
    Az = Az[index_no0]
    Ra = Ra[index_no0]
    El = El[index_no0]
    Intens = Intens[index_no0]
    Intens = np.squeeze(Intens)
    Flag = Flag[index_no0]

    # 旋转点云，转换为场景坐标系
    cIncidentAngle = 90 - IncidentAngle
    A = np.array([[np.cos(cIncidentAngle * np.pi /180), np.sin(cIncidentAngle * np.pi /180)],
                [-np.sin(cIncidentAngle * np.pi /180), np.cos(cIncidentAngle * np.pi /180)]])
    A = np.squeeze(A)
    p_after = np.dstack((Ra,El))
    p_after = np.squeeze(p_after)
    p_after = np.dot(A,p_after.T)
    p_after = p_after.T
    p_after = np.dstack((p_after[:,0], Az, p_after[:,1]))
    p_after = np.squeeze(p_after)

    # 将目标移到场景中心
    # X = p_after[:,0]-np.mean(p_after[:,0])
    Y = p_after[:,1]
    
    Z = p_after[:,2]
    counter = Counter(Z)
    Z_bias = counter.most_common(1)
    Z = Z - Z_bias[0][0]

    index_1 = np.array(np.where(Z>-0.01))
    index_2 = np.array(np.where(Z<0.01))
    index = np.array(np.intersect1d(index_1, index_2))
    X = p_after[:,0]-np.mean(p_after[:,0][index])

    # 对地面进行裁剪
    Z_cut_min = -0.01
    Z_cut_max = 500

    index_x1 = np.array(np.where(X>X_cut_min))
    index_x2 = np.array(np.where(X<X_cut_max))
    index_x = np.array(np.intersect1d(index_x1, index_x2))

    index_y1 = np.array(np.where(Y>Y_cut_min))
    index_y2 = np.array(np.where(Y<Y_cut_max))
    index_y = np.array(np.intersect1d(index_y1, index_y2))

    index_z1 = np.array(np.where(Z>Z_cut_min))
    index_z2 = np.array(np.where(Z<Z_cut_max))
    index_z = np.array(np.intersect1d(index_z1, index_z2))

    index_xy = np.array(np.intersect1d(index_x,index_y))
    index = np.array(np.intersect1d(index_xy,index_z))

    index_scatter1_flag = np.array(np.where(Flag==1))
    index_scatter1 = np.array(np.intersect1d(index,index_scatter1_flag))
    index_scatter2_flag = np.array(np.where(Flag==2))
    index_scatter2 = np.array(np.intersect1d(index,index_scatter2_flag))
    index_scatter3_flag = np.array(np.where(Flag==3))
    index_scatter3 = np.array(np.intersect1d(index,index_scatter3_flag))

    X1, Y1, Z1, Intens1 = processForSingle(index_scatter1, X, Y, Z, Intens, target,IncidentAngle,Azimuth,filename)
    X2, Y2, Z2, Intens2 = processForDouble(index_scatter2, X, Y, Z, Intens, target,IncidentAngle,Azimuth,filename)
    X3, Y3, Z3, Intens3 = processForTriple(index_scatter3, X, Y, Z, Intens, target,IncidentAngle,filename)

    # 查看3种散射次数分别的建模结果
    # electronics1 = np.dstack((Y1,-X1,Z1,Intens1))
    # electronics1 = np.squeeze(electronics1)
    # sio.savemat("/home/liq/pro/Debug/mat_1_B2.mat", {"data": electronics1})
    # electronics2 = np.dstack((Y2,-X2,Z2,Intens2))
    # electronics2 = np.squeeze(electronics2)
    # sio.savemat("/home/liq/pro/Debug/mat_2_B2.mat", {"data": electronics2})
    # electronics3 = np.dstack((Y3,-X3,Z3,Intens3))
    # electronics3 = np.squeeze(electronics3)
    # sio.savemat("/home/liq/pro/Debug/mat_3_B2.mat", {"data": electronics3})
    
    # 输出电磁建模位置坐标结果
    X = np.concatenate((X1,X2,X3),axis=0)
    Y = np.concatenate((Y1,Y2,Y3),axis=0)
    Z = np.concatenate((Z1,Z2,Z3),axis=0)

    # 斜视角情况
    squiAng = -squiAng
    Xnew = X * np.cos(squiAng * np.pi/180) - Y * np.sin(squiAng * np.pi/180)
    Ynew = X * np.sin(squiAng * np.pi/180) + Y * np.cos(squiAng * np.pi/180)

    # 极化情况
    if polarMethod == 'VV':
        Intens1 = np.power(Intens1, 14/15)
        
    # 输出散射强度建模结果
    Intens = np.concatenate((Intens1,Intens2,Intens3),axis=0)

    # 输出电磁建模结果
    electronics = np.dstack((Ynew,-Xnew,Z,Intens))
    # electronics = np.dstack((Xnew,Ynew,Z,Intens))
    electronics = np.squeeze(electronics)
    # sio.savemat("/home/lij/PILIANGHUAEXPERIMENT/T72_debug_save/mat_20_0_0.mat", {"data": electronics})

    # 用于matlab做图像渲染
    np.savetxt("/home/chenjc/newprojects/DCFZ/txtData/results.txt", electronics, fmt='%.8f')
    
    return electronics

if __name__ == "__main__":
    import scipy.io as sio
    # 电磁建模
    pitchAngel = 40
    Azimuth = 0
    target = "FJ"
    txtFile = "/home/liq/pro/35Targets/B2/txt/Contributions_40_0_0.txt"
    # resultMatFile = "/home/lij/pro/Debug/test_B2.mat"
    txtProParams = [Azimuth, pitchAngel, 0, target, 'HH', 147, 95, txtFile]
    elecResult = txtProcess(txtProParams)
    # savedic = {"data": elecResult, "offNadiAng": pitchAngel}
    # savedic["data"] = np.array(savedic["data"], dtype="double")
    # # 回波仿真
    # settingsPath0 = '/home/lij/PILIANGHUAEXPERIMENT/T72/settings.mat'
    # initParams = sio.loadmat(settingsPath0)
    # # print(initParams["data"].shape)
    # # print(elecResult.shape)
    # initParams.update(savedic)
    # settingsPath = '/home/lij/PILIANGHUAEXPERIMENT/txtProcess_debug/settings.mat'
    # sio.savemat(settingsPath, initParams)
    # echoSimPath = "/home/chenjc/newprojects/EchoSim/EchoSimV2/build"
    # import subprocess
    # echo_path = '/home/lij/PILIANGHUAEXPERIMENT/txtProcess_debug/echo.mat'
    # echoSimCmd = ["./main", settingsPath, echo_path]
    # result = subprocess.run(echoSimCmd, capture_output=True, text=True, check=True, cwd=echoSimPath)
    # # 成像处理
    # from Five_mode_imaging import *
    # imaging_current = imaging(echo_path)
    # imaging_current.imaging()
    # img_path = '/home/lij/PILIANGHUAEXPERIMENT/txtProcess_debug/img.mat'
    # imaging_current.show_image(2, img_path)
    # # 存为.png
    # data = sio.loadmat(img_path)['image']
    # image = np.array(data[:,125:475], dtype=np.uint8).transpose()
    # import matplotlib.image as mpimg
    # png_path = '/home/lij/PILIANGHUAEXPERIMENT/txtProcess_debug/result_VV.png'
    # mpimg.imsave(png_path, 1-image, cmap=plt.cm.Greys)