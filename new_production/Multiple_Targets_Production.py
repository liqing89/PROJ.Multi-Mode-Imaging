from mkdataset_for_all_file import * # 生产脚本（改）
# 注意mkdataset_for_all_file在./new_production_1021/路径下！
import time

# 生产目标信息列表(生产时唯一需要修改的部分)
filename = '/home/liq/pro/new_production/NAME.txt'
# 扫描模式设置
scanMode = 1
# 斜视角设置
squiAng = 0

# 读取目标
data = np.loadtxt(filename, dtype=str)
model_Name = data[:,0]
model_Type = data[:,1]
# 设置运行文件夹
save_folder = '/home/data1/user/PiLiangHuaExperiment/Dataset_for_Production/'
# 设置存储中间结果文件夹
# mid_folder = '/home/data2/Generation_mid_results/'
mid_folder = '/home/data1/user/PiLiangHuaExperiment/Dataset_for_Production_MidResult/'
# 设置XML模板文件
model_xml_file_head = '/home/lij/PILIANGHUAEXPERIMENT/xmlModel_'
# 清空时间记录
# with open(time_record,'a+',encoding='utf-8') as f:
#     f.truncate(0)
# f.close()
# 开始生产
for i in range(len(model_Name)):
    # T0 = time.time()
    # 分辨率设置
    if model_Type[i] == 'JC' or model_Type[i] == 'HM':
        flag = 'JC'
        Rho_list = [2] # [3, 2, 1]
    elif model_Type[i] == 'FJ' or model_Type[i] == 'TK':
        flag = model_Type[i]
        Rho_list = [0.3] # 小飞机分辨率（改 [0.5, 0.3, 0.2]）
        if model_Name[i] == 'C17' or model_Name[i] == 'B52' or model_Name[i] == 'B2' or model_Name[i] == 'B1B' or model_Name[i] == 'AC130' or model_Name[i] == 'KC135':
            Rho_list = [0.3] # 大飞机分辨率（改 [1, 0.5, 0.3]）
    else:
        pass

    model_info_folder = '/home/lij/PILIANGHUAEXPERIMENT/35_Targets_POV_and_XML/' + model_Name[i] + '/'
    model_xml_file = model_xml_file_head + flag + '.xml'
    for polarization in ['HH']: # 极化方式（改['HH', 'VV']）
        for wave_band in ['X']: # 波段（改['X', 'C']）
            for Rho in Rho_list:
                T2 = time.time()
                if model_Type[i] == 'JC':
                    if model_Name[i] == 'ZHBJC':
                        distributionx = 455
                        distributiony = 455
                    elif model_Name[i] == 'HF':
                        distributionx = 505
                        distributiony = 505
                    else:
                        distributionx = 275
                        distributiony = 275
                    x_cut = distributionx-5
                    y_cut = distributiony-5
                    RayH = y_cut/Rho
                    RayW = x_cut/Rho
                    beta_range = 3
                    beta_azimuth = 2
                elif model_Type[i] == 'HM':
                    distributionx = 505
                    distributiony = 505
                    if model_Name[i] == 'HMBD':
                        distributionx = 2005
                        distributiony = 2005
                    x_cut = distributionx-5
                    y_cut = distributiony-5
                    RayH = y_cut/Rho
                    RayW = x_cut/Rho
                    beta_range = 3
                    beta_azimuth = 2
                elif model_Type[i] == 'FJ': #要改
                    distributionx = 95
                    distributiony = 95
                    x_cut = distributionx-5
                    y_cut = distributiony-5
                    RayH = y_cut/Rho
                    RayW = x_cut/Rho
                    beta_range = 4
                    beta_azimuth = 2.5
                elif model_Type[i] == 'TK':
                    distributionx = 35
                    distributiony = 35
                    x_cut = distributionx-5
                    y_cut = distributiony-5
                    RayH = y_cut/Rho
                    RayW = x_cut/Rho
                    beta_range = 4
                    beta_azimuth = 2.5
                else:
                    pass
                current_save_folder = save_folder + model_Type[i] + '/' + model_Name[i] + '/' + polarization + '/' + wave_band + '/' + str(Rho) + '_' + str(Rho) + '/'
                current_mid_folder = mid_folder + model_Type[i] + '/' + model_Name[i] + '/' + polarization + '/' + wave_band + '/' + str(Rho) + '_' + str(Rho)
                mkdataset(model_Name[i], model_info_folder, model_xml_file, current_save_folder, current_mid_folder, distributionx, distributiony, RayH, RayW, model_Type[i], x_cut, y_cut, scanMode, Rho, beta_range, beta_azimuth, wave_band, polarization, squiAng)
                # T3 = time.time()
                # with open(time_record,'a+',encoding='utf-8') as f:
                #     f.write('%s极化，%s波段，分辨率%s，生产时间：%dh %dm %ds\n' % (polarization, wave_band, str(Rho), int(int(T3-T2)/3600), int((int(T3-T2)%3600)/60), int((int(T3-T2)%3600)%60)))
                # f.close()
    # T1 = time.time()
    # with open(time_record,'a+',encoding='utf-8') as f:
    #     f.write('%s生产时间：%dh %dm %ds\n' % (model_Name[i], int(int(T1-T0)/3600), int((int(T1-T0)%3600)/60), int((int(T1-T0)%3600)%60)))
    # f.close()

from utils import zip_folder
zip_folder(save_folder[:-1], save_folder[:-1]+'.zip')

