#!/usr/bin/python3
# @Author: Jc Chen
# @Title: 
# @Modified: 
import re
from datetime import datetime

import xml.dom.minidom as minidom
import xmltodict
import os
import tifffile
import numpy as np
import math
from osgeo import gdal, osr
import matplotlib.pyplot as plt
from zipfile import ZipFile 

def zip_folder(folder_path, zip_name):
    with ZipFile(zip_name, 'w') as zipf:
        for foldername, subfolders, filenames in os.walk(folder_path):
            for filename in filenames:
                file_path = os.path.join(foldername, filename)
                arcname = os.path.relpath(file_path, folder_path)
                zipf.write(file_path, arcname)

def addGeoInfoToTiff(tiffPath, outPath, cenLng, cenLat, xRes, yRes):

    image_data = tifffile.imread(tiffPath)
    Na, Nr = image_data.shape
    lng_pix = xRes * 360 /(2*math.pi*6371e3)
    lat_pix = yRes * 360 /(2*math.pi*6371e3*math.cos(math.radians(cenLat)))

    startLng = cenLng - Na/2 * lng_pix
    startLat = cenLat - Nr/2 * lat_pix

    driver = gdal.GetDriverByName('GTiff')
    # 创建GeoTIFF数据集
    dataset = driver.Create(outPath, image_data.shape[1], image_data.shape[0], 1, gdal.GDT_UInt16)
    # 将图像数据写入数据集
    dataset.GetRasterBand(1).WriteArray(image_data)
    # 设置地理坐标信息
    projection = osr.SpatialReference()
    projection.ImportFromEPSG(4326)  # 设置坐标系（EPSG 4326代表WGS 84，经纬度坐标系）
    dataset.SetProjection(projection.ExportToWkt())

    # 设置地理变换参数（如果有）
    geoTransform = [startLng, lng_pix, 0, startLat, 0, lat_pix]
    #栅格数据的六参数。
        # geoTransform[0]：左上角像素经度
        # geoTransform[1]：影像宽度方向上的分辨率(经度范围/像素个数)
        # geoTransform[2]：x像素旋转, 0表示上面为北方
        # geoTransform[3]：左上角像素纬度
        # geoTransform[4]：y像素旋转, 0表示上面为北方
        # geoTransform[5]：影像宽度方向上的分辨率(纬度范围/像素个数)
    dataset.SetGeoTransform(geoTransform)
    # 保存文件
    dataset.FlushCache()
    # 关闭文件
    dataset = None

def idParser(fileName):
    str_com = re.split(r'[_.]+', fileName)
    id = str_com[-3] + '_' + str_com[-2]
    return id

def getCurrentTimeInfo():
    current_time = datetime.now()
    id_compact = current_time.strftime('%y%m%d')
    id_full = current_time.strftime('%y%m%d_%H%M%S')
    id_locale = current_time.strftime('%Y-%m-%d %H:%M:%S')
    timestamp = int(datetime.timestamp(current_time)*1000)          # 13位时间戳
    return [timestamp, id_compact, id_full, id_locale]

class CurrTime:
    def __init__(self) -> None:
        current_time = datetime.now()
        self.ymdTime = current_time.strftime('%y%m%d')
        self.idTime = current_time.strftime('%y%m%d_%H%M%S')
        self.localeTime = current_time.strftime('%Y%m%d%H%M%S')
        self.timestamp13 = str(int(datetime.timestamp(current_time)*1000))         # 13位时间戳
        self.timestamp16 = str(int(datetime.timestamp(current_time)*1e6))          # 16位时间戳

# dict转xml并保存文件
def dictToXmlFile(xmlFilePath, data):

    # Function to create XML nodes recursively
    def dict_to_xml(doc, parent, data):
        for key, value in data.items():
            if isinstance(value, dict):
                # For nested dictionaries, create an element and call the function recursively
                element = doc.createElement(key)
                parent.appendChild(element)
                dict_to_xml(doc, element, value)
            else:
                # For text values, create a text node
                element = doc.createElement(key)
                parent.appendChild(element)
                text = doc.createTextNode(str(value) if value is not None else '')
                element.appendChild(text)

    # Create the minidom Document object
    doc = minidom.Document()

    # Convert the dictionary to XML
    dict_to_xml(doc, doc, data)

    # Convert the Document object to a formatted string with indentation
    pretty_xml_str = doc.toprettyxml(indent="\t", encoding='utf-8').decode()

    # Save the XML to a file
    with open(xmlFilePath, 'w', encoding='utf-8') as file:
        file.write(pretty_xml_str)


def xmlFileToDict(xmlFilePath):
    
    # Read the XML file
    with open(xmlFilePath, 'r', encoding='utf-8') as file:
        xml_str = file.read()

    # Convert the XML to a dictionary
    xml_dict = xmltodict.parse(xml_str)

    return xml_dict


def createFolderWithId(rootPath, folderName):
    counter = 1
    folderNameWithId = folderName
    while os.path.exists(os.path.join(rootPath, folderNameWithId)):
        folderNameWithId = f"{folderName}_{counter}"
        counter += 1
    
    newFolderPath = os.path.join(rootPath, folderNameWithId)
    os.makedirs(newFolderPath)
    return newFolderPath

def getTiffRender(tiffPath, lowerBound = 1, upperBound = 99):
    tiffData = tifffile.imread(tiffPath)
    img = np.sqrt(tiffData[0,:,:]**2 + tiffData[1,:,:]**2)
    img = np.abs(img)
    pL, pU = np.percentile(img, (lowerBound, upperBound))
    img = np.clip(img, pL, pU)
    img = (img - pL)/(pU - pL) * 255
    # 使用伽马校正进一步增强对比度
    img = (img/255)**(1/1.5)*255
    return img

def saveRngLineFromImg(imgPath, outPath):
    img = plt.imread(imgPath)


    
    ss = np.sum(np.abs(img[:,:,0]),axis = 0).reshape(-1)
    
    plt.figure()
    plt.plot(ss)
    plt.savefig(outPath, pad_inches=0)
    plt.close()

    # ss = self.transform_coordinate(ss).reshape(-1)
    # if self.mode == 4:
    #     self.dazm = self.azmRho
    # Nr = ss.shape[0]
    # Nr_start = int(Nr/2)-int(self.rngScale/self.dazm/2)
    # Nr_end = Nr_start + int(self.rngScale/self.dazm)
    # self.range_image = ss[Nr_start:Nr_end]

    
if __name__ == "__main__":
    imgPath = "/home/chenjc/newprojects/database/tztq/tztq_batch_279/imgSlice.png"
    saveRngLineFromImg(imgPath, './rngTest.png')

# # original version
# #!/usr/bin/python3
# # @Author: Jc Chen
# # @Title: 
# # @Modified: 
# import re
# from datetime import datetime

# import xml.dom.minidom as minidom
# import xmltodict
# import os
# import tifffile
# import numpy as np
# import matplotlib.pyplot as plt

# def idParser(fileName):
#     str_com = re.split(r'[_.]+', fileName)
#     id = str_com[-3] + '_' + str_com[-2]
#     return id

# def getCurrentTimeInfo():
#     current_time = datetime.now()
#     id_compact = current_time.strftime('%y%m%d')
#     id_full = current_time.strftime('%y%m%d_%H%M%S')
#     id_locale = current_time.strftime('%Y-%m-%d %H:%M:%S')
#     timestamp = int(datetime.timestamp(current_time)*1000)          # 13位时间戳
#     return [timestamp, id_compact, id_full, id_locale]

# class CurrTime:
#     def __init__(self) -> None:
#         current_time = datetime.now()
#         self.ymdTime = current_time.strftime('%y%m%d')
#         self.idTime = current_time.strftime('%y%m%d_%H%M%S')
#         self.localeTime = current_time.strftime('%Y%m%d%H%M%S')
#         self.timestamp13 = str(int(datetime.timestamp(current_time)*1000))         # 13位时间戳
#         self.timestamp16 = str(int(datetime.timestamp(current_time)*1e6))          # 16位时间戳

# # dict转xml并保存文件
# def dictToXmlFile(xmlFilePath, data):

#     # Function to create XML nodes recursively
#     def dict_to_xml(doc, parent, data):
#         for key, value in data.items():
#             if isinstance(value, dict):
#                 # For nested dictionaries, create an element and call the function recursively
#                 element = doc.createElement(key)
#                 parent.appendChild(element)
#                 dict_to_xml(doc, element, value)
#             else:
#                 # For text values, create a text node
#                 element = doc.createElement(key)
#                 parent.appendChild(element)
#                 text = doc.createTextNode(str(value) if value is not None else '')
#                 element.appendChild(text)

#     # Create the minidom Document object
#     doc = minidom.Document()

#     # Convert the dictionary to XML
#     dict_to_xml(doc, doc, data)

#     # Convert the Document object to a formatted string with indentation
#     pretty_xml_str = doc.toprettyxml(indent="\t", encoding='utf-8').decode()

#     # Save the XML to a file
#     with open(xmlFilePath, 'w', encoding='utf-8') as file:
#         file.write(pretty_xml_str)


# def xmlFileToDict(xmlFilePath):
    

#     # Read the XML file
#     with open(xmlFilePath, 'r', encoding='utf-8') as file:
#         xml_str = file.read()

#     # Convert the XML to a dictionary
#     xml_dict = xmltodict.parse(xml_str)

#     return xml_dict


# def createFolderWithId(rootPath, folderName):
#     counter = 1
#     folderNameWithId = folderName
#     while os.path.exists(os.path.join(rootPath, folderNameWithId)):
#         folderNameWithId = f"{folderName}_{counter}"
#         counter += 1
    
#     newFolderPath = os.path.join(rootPath, folderNameWithId)
#     os.makedirs(newFolderPath)
#     return newFolderPath

# def getTiffRender(tiffPath, lowerBound = 1, upperBound = 99):
#     tiffData = tifffile.imread(tiffPath)
#     img = np.sqrt(tiffData[0,:,:]**2 + tiffData[1,:,:]**2)
#     img = np.abs(img)
#     pL, pU = np.percentile(img, (lowerBound, upperBound))
#     img = np.clip(img, pL, pU)
#     img = (img - pL)/(pU - pL) * 255
#     # 使用伽马校正进一步增强对比度
#     img = (img/255)**(1/1.5)*255
#     return img

# def saveRngLineFromImg(imgPath, outPath):
#     img = plt.imread(imgPath)


    
#     ss = np.sum(np.abs(img[:,:,0]),axis = 0).reshape(-1)
    
#     plt.figure()
#     plt.plot(ss)
#     plt.savefig(outPath, pad_inches=0)
#     plt.close()

#     # ss = self.transform_coordinate(ss).reshape(-1)
#     # if self.mode == 4:
#     #     self.dazm = self.azmRho
#     # Nr = ss.shape[0]
#     # Nr_start = int(Nr/2)-int(self.rngScale/self.dazm/2)
#     # Nr_end = Nr_start + int(self.rngScale/self.dazm)
#     # self.range_image = ss[Nr_start:Nr_end]

    
# if __name__ == "__main__":
#     imgPath = "/home/chenjc/newprojects/database/tztq/tztq_batch_279/imgSlice.png"
#     saveRngLineFromImg(imgPath, './rngTest.png')
    



    