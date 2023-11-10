import numpy as np
from utils import dictToXmlFile, xmlFileToDict, CurrTime
from dataProcess import Extractor
import tifffile as tiff
import subprocess
import os
import shutil
from txtProcess import * # 电磁建模结果处理文件
import scipy.io as sio # 后面三个为成像库
from new_production_1021.read_data import * 
from Five_mode_imaging_V2 import *
import matplotlib.image as mpimg

def mkdataset(model_name, model_info_folder, model_xml_file, current_save_folder, current_mid_folder, distributionx, distributiony, RayH, RayW,
              target_type, x_cut, y_cut, scanMode, Rho, beta_range, beta_azimuth, wave_band, polarization, squiAng):

    # 参数设置  
    rngRho = Rho
    azmRho = Rho

    if target_type == 'HM' or target_type == 'JC':
        military_base = '梅波特海军基地'
        base_coordinate = '30.38, -81.41'
    elif target_type == 'FJ':
        military_base = '内利斯空军基地'
        base_coordinate = '36.24, -115.05'
    else:
        military_base = '埃格林空军基地'
        base_coordinate = '30.46, -86.55'

    if target_type == 'HM':
        target_flag1 = 'jc'
    else:
        target_flag1 = target_type.lower()
    
    if target_type == 'TK' or target_type == 'HM':
        target_flag2 = 'jc'
    else:
        target_flag2 = target_type.lower()
                

    # 初始化文件目录
    if os.path.exists(current_save_folder):
        shutil.rmtree(current_save_folder)
    os.makedirs(current_save_folder)

    save_echo_folder_name =  current_mid_folder + '/echo/'

    print('开始生产目标：' + model_name + '，目标类型：' + target_type + '，极化：' + polarization + '，波段：' + wave_band + '，分辨率：' + str(Rho))
    xmlfilename = model_info_folder + model_name +'.xml'

    # 俯仰角设置
    for incidentAngle in range(18,56,5):
        for j in range(0,36,1):
            # 方位角设置
            current_degree = j*10
            # 设置存储文件
            current_name = str(incidentAngle)+'_'+str(current_degree)
            current_save_name = current_save_folder + current_name
            os.makedirs(current_save_name)
            # 取出xml模板
            model_xml_dict = xmlFileToDict(model_xml_file)
            current_xml_dict = xmlFileToDict(xmlfilename)
            # 更新target_info
            timestamp16 = CurrTime().timestamp16
            model_xml_dict["ndm"]["body"]["target_info"].update({"target_id": timestamp16})
            model_xml_dict["ndm"]["body"]["target_info"].update({"target_name": current_xml_dict["ndm"]["body"]["target_info"]["target_model"]+'_'+timestamp16})
            model_xml_dict["ndm"]["body"]["target_info"].update({"nation": current_xml_dict["ndm"]["body"]["target_info"]["country"]})
            if target_type != 'TK':
                model_xml_dict["ndm"]["body"]["target_info"].update({"level_3_class": current_xml_dict["ndm"]["body"]["target_info"]["label3"]})
                model_xml_dict["ndm"]["body"]["target_info"].update({"level_4_class": current_xml_dict["ndm"]["body"]["target_info"]["label4"]})
            model_xml_dict["ndm"]["body"]["target_info"].update({"target_model": current_xml_dict["ndm"]["body"]["target_info"]["target_model"]})
            model_xml_dict["ndm"]["body"]["target_info"].update({"military_base": military_base})
            model_xml_dict["ndm"]["body"]["target_info"].update({"base_coordinate": base_coordinate})
            model_xml_dict["ndm"]["body"]["target_info"].update({"aspect_angle": current_degree-(97.4477903825153-90)})
            model_xml_dict["ndm"]["body"]["target_info"].update({"model_path": model_name+'.glb'})
            model_xml_dict["ndm"]["body"]["target_info"].update({target_flag1+'_length': current_xml_dict["ndm"]["body"]["target_info"][target_flag2+'_length']})
            model_xml_dict["ndm"]["body"]["target_info"].update({target_flag1+'_width': current_xml_dict["ndm"]["body"]["target_info"][target_flag2+'_width']})
            model_xml_dict["ndm"]["body"]["target_info"].update({target_flag1+'_height': current_xml_dict["ndm"]["body"]["target_info"][target_flag2+'_height']})
            if target_flag1 == 'jc':
                model_xml_dict["ndm"]["body"]["target_info"].update({"draft": current_xml_dict["ndm"]["body"]["target_info"]["depth"]})
                model_xml_dict["ndm"]["body"]["target_info"].update({"loaded_displacement": current_xml_dict["ndm"]["body"]["target_info"]["full_loaded_displacement"]})
                model_xml_dict["ndm"]["body"]["target_info"].update({"standard_displacement": current_xml_dict["ndm"]["body"]["target_info"]["tonnage"]})
            # 更新image_info
            model_xml_dict["ndm"]["body"]["image_info"].update({"chip_id": timestamp16})
            model_xml_dict["ndm"]["body"]["image_info"].update({"chip_name": model_xml_dict["ndm"]["body"]["target_info"]["target_name"]+'.tiff'})
            model_xml_dict["ndm"]["body"]["image_info"].update({"chip_path": model_xml_dict["ndm"]["body"]["target_info"]["target_name"]+'.tiff'})
            model_xml_dict["ndm"]["body"]["image_info"]["relation"].update({"incident_angle": incidentAngle})
            model_xml_dict["ndm"]["body"]["image_info"]["relation"].update({"incident_direction": current_degree})
            model_xml_dict["ndm"]["body"]["image_info"]["sar_payload"].update({"wave_band": wave_band})
            model_xml_dict["ndm"]["body"]["image_info"]["sar_payload"].update({"polar_mode": polarization})
            model_xml_dict["ndm"]["body"]["image_info"]["sar_payload"].update({"range_resolution": rngRho})
            model_xml_dict["ndm"]["body"]["image_info"]["sar_payload"].update({"azimuth_resolution": azmRho})
            # 进行成像处理并存为.tiff文件
            save_tiff_path = current_save_name + '/' + model_xml_dict["ndm"]["body"]["target_info"]["target_name"] + '.tiff'
            save_range_path = current_save_name + '/' + model_xml_dict["ndm"]["body"]["target_info"]["target_name"] + '_hrrp.txt'
            # 成像
            echoResultPath = save_echo_folder_name + current_name + '.mat'
            imaging_current = imaging(echoResultPath, beta_range, beta_azimuth)
            imaging_current.imaging()
            imaging_current.dataExpose() # 保存成像结果为成员
            # 一维距离向保存  
            np.savetxt(save_range_path, np.c_[np.array(abs(imaging_current.range_image))], fmt='%f',delimiter='\n')
            plt.figure()
            plt.plot(imaging_current.range_image)
            plt.savefig(current_save_name + '/'+'yiweijulixiang.png', pad_inches=0)
            # 更新image_info
            model_xml_dict["ndm"]["body"]["image_info"].update({"imaging_time": CurrTime().localeTime})
            model_xml_dict["ndm"]["body"]["image_info"].update({"resolution": imaging_current.dazm})
            # 更新inversion_info
            model_xml_dict["ndm"]["body"]["inversion_info"]["electromagnetic_scattering_characteristic"].update({"HRRP_path": model_xml_dict["ndm"]["body"]["target_info"]["target_name"]+'_hrrp.txt'})
            # 储存.tiff
            tiff.imwrite(save_tiff_path, imaging_current.tiff)
            mpimg.imsave(current_save_name + '/'+current_name+'.png', np.array(imaging_current.img, dtype=np.uint8), cmap=plt.cm.gray)
            # 进行目标特征提取
            save_scatter_Path = current_save_name + '/' + model_xml_dict["ndm"]["body"]["target_info"]["target_name"] + '_attr.txt'
            try:
                # 参数设置
                dazm = imaging_current.dazm
                imgMeta =  {"azimuth_resolution": dazm, "distance_resolution": dazm}
                # 调用特征提取算法并存储散射中心txt
                worker = Extractor()
                worker.performExtract(save_tiff_path, imgMeta, target_type, save_scatter_Path)
                plt.imshow(worker.scatterImg)
                plt.axis('off')
                plt.savefig(current_save_name + '/'+'sanzhezhongxin.png', bbox_inches='tight', pad_inches=0)
                plt.close()
                # 封装提取的特征
                meta = {
                    "retrieval_length": worker.tgInfo['length'],
                    "retrieval_width": worker.tgInfo['width'],
                    "area": float(worker.tgInfo['length']) * float(worker.tgInfo['width']),
                    "aspect_angle": worker.tgInfo['rotateAng'],
                }
                # 更新inversion_info
                model_xml_dict["ndm"]["body"]["inversion_info"].update(meta)
                model_xml_dict["ndm"]["body"]["inversion_info"].update({"incidence_angle": incidentAngle})
                model_xml_dict["ndm"]["body"]["inversion_info"]["electromagnetic_scattering_characteristic"].update({"attribute_scatter": model_xml_dict["ndm"]["body"]["target_info"]["target_name"] + '_attr.txt'})
                # 更新存储xml文件
                savexmlname = current_save_name + '/' + current_name + '.xml'
                dictToXmlFile(savexmlname, model_xml_dict)
            except Exception as e:
                # print(e)
                continue