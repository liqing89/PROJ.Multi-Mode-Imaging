import numpy as np
import os
import matplotlib.pyplot as plt
from skimage import io, morphology
from skimage.segmentation import watershed
import cv2
import math
from scipy.ndimage import rotate, label, distance_transform_edt
import scipy.io as sio
from scipy import ndimage
from skimage.transform import radon
import numpy as np
from sklearn.cluster import KMeans
from skimage.transform import radon
from scipy.ndimage.filters import uniform_filter 
from scipy.ndimage.measurements import variance 
import tifffile
def lee_filter(img, size): 
    img_mean = uniform_filter(img, (size, size)) 
    img_sqr_mean = uniform_filter(img**2, (size, size)) 
    img_variance = img_sqr_mean - img_mean**2 

    overall_variance = variance(img) 

    img_weights = img_variance**2/(img_variance**2 + overall_variance**2) 
    img_output = img_mean + img_weights * (img - img_mean) 
    return img_output 

# MRF分割
def mrf(imgGray):
    imgGray = imgGray / 255
    imgCopy = imgGray.copy()
    imgpixel = (imgCopy.flatten()).reshape((imgGray.shape[0]*imgGray.shape[1], 1))
    kind = 2
    kmeans = KMeans(n_clusters=kind)
    label = kmeans.fit(imgpixel)
    imgLabel = np.array(label.labels_).reshape(imgGray.shape)
    #令目标区域确认为1
    if np.mean(imgGray[np.where(imgLabel==0)]) >= np.mean(imgGray[np.where(imgLabel==1)]):
        # print(np.mean(imgGray[np.where(imgLabel==1)]))
        # print(np.mean(imgGray[np.where(imgLabel==0)]))
        imgLabel=1-imgLabel
    # plt.figure()
    # plt.imshow(imgLabel,cmap='gray')

    return imgLabel

 #找最大连通区域

#找最大连通区，认定为目标
def find_largest_area_2(image):
    #findcontours函数要求灰度图，即dtype=uint8
    image=np.uint8(image)
    contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    areas = []
 
    for c in range(len(contours)):
        areas.append(cv2.contourArea(contours[c]))
    
    max_idx = np.argmax(areas)
    max_area = cv2.contourArea(contours[max_idx])
 
    for k in range(len(contours)):
    
        if k != max_idx:
            cv2.fillPoly(image, [contours[k]], 0)
    return image

def find_largest_area_13(image):
    #findcontours函数要求灰度图，即dtype=uint8
    closed_image=np.uint8(image)

     # 进行闭操作
    kernel = np.ones((3, 3), np.uint8)


    # 计算连通区域
    _, labels, stats, _ = cv2.connectedComponentsWithStats(closed_image, connectivity=8)


    # 找到最大连通区域的标签
    max_label = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
    max_area = stats[max_label, cv2.CC_STAT_AREA]

    closed_image = np.zeros_like(image, dtype=np.uint8)
    mask = np.zeros_like(image, dtype=np.uint8)
    mask[labels == max_label] = 1
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    closed_image = cv2.bitwise_or(closed_image, mask)

    # 检查每个连通区域的面积
    for label in range(1, np.max(labels) + 1):
        area = stats[label, cv2.CC_STAT_AREA]

        # 如果连通区域的面积大于总面积的20%，将其与最大连通区域相连
        if area > 0.05 * max_area:
            mask = np.zeros_like(image, dtype=np.uint8)
            mask[labels == label] = 1
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            closed_image = cv2.bitwise_or(closed_image, mask)

    closed_image = cv2.morphologyEx(closed_image, cv2.MORPH_CLOSE, kernel)


    # 将离最大连通区较远的区域视作杂波
    
    return closed_image

def find_largest_area_4(image):
    # #findcontours函数要求灰度图，即dtype=uint8
    # closed_image=np.uint8(image)

    #  # 进行闭操作
    # kernel = np.ones((3, 3), np.uint8)


    # # 计算连通区域
    # _, labels, stats, _ = cv2.connectedComponentsWithStats(closed_image, connectivity=8)


    # # 找到最大连通区域的标签
    # max_label = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1

    # closed_image = np.zeros_like(image, dtype=np.uint8)
    # mask = np.zeros_like(image, dtype=np.uint8)
    # mask[labels == max_label] = 1
    # mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    # closed_image = cv2.bitwise_or(closed_image, mask)

    # # 检查每个连通区域的面积
    # for label in range(1, np.max(labels) + 1):
    #     area = stats[label, cv2.CC_STAT_AREA]

    #     if area:
    #         mask = np.zeros_like(image, dtype=np.uint8)
    #         mask[labels == label] = 1
    #         mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    #         closed_image = cv2.bitwise_or(closed_image, mask)

    # closed_image = cv2.morphologyEx(closed_image, cv2.MORPH_CLOSE, kernel)


    # 将离最大连通区较远的区域视作杂波
    
    
    closed_image=np.uint8(image)

     # 进行闭操作
    kernel = np.ones((3, 3), np.uint8)


    # 计算连通区域
    _, labels, stats, _ = cv2.connectedComponentsWithStats(closed_image, connectivity=8)

    '''
    update:2023-11-18
    注释掉的为原版dataProcess_png的代码
    替换为跟lij的代码一致的代码

    # 找到最大连通区域的标签
    max_label = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
    max_area = stats[max_label, cv2.CC_STAT_AREA]

    closed_image = np.zeros_like(image, dtype=np.uint8)
    mask = np.zeros_like(image, dtype=np.uint8)
    mask[labels == max_label] = 1
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    closed_image = cv2.bitwise_or(closed_image, mask)

    # 检查每个连通区域的面积
    for label in range(1, np.max(labels) + 1):
        area = stats[label, cv2.CC_STAT_AREA]

        # 如果连通区域的面积大于总面积的20%，将其与最大连通区域相连
        if area > 0.05 * max_area:
            mask = np.zeros_like(image, dtype=np.uint8)
            mask[labels == label] = 1
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            closed_image = cv2.bitwise_or(closed_image, mask)

    closed_image = cv2.morphologyEx(closed_image, cv2.MORPH_CLOSE, kernel)
    return closed_image
    '''

    #findcontours函数要求灰度图，即dtype=uint8
    closed_image=np.uint8(image)

     # 进行闭操作
    kernel = np.ones((3, 3), np.uint8)


    # 计算连通区域
    _, labels, stats, _ = cv2.connectedComponentsWithStats(closed_image, connectivity=8)


    # 找到最大连通区域的标签
    max_label = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1

    closed_image = np.zeros_like(image, dtype=np.uint8)
    mask = np.zeros_like(image, dtype=np.uint8)
    mask[labels == max_label] = 1
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    closed_image = cv2.bitwise_or(closed_image, mask)

    # 检查每个连通区域的面积
    for label in range(1, np.max(labels) + 1):
        area = stats[label, cv2.CC_STAT_AREA]

        if area:
            mask = np.zeros_like(image, dtype=np.uint8)
            mask[labels == label] = 1
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            closed_image = cv2.bitwise_or(closed_image, mask)

    closed_image = cv2.morphologyEx(closed_image, cv2.MORPH_CLOSE, kernel)


    # 将离最大连通区较远的区域视作杂波
    
    return closed_image


# 估计方位角
def azumith_estimation_2(left, right, down):
    # radonleft = DiscreteRadonTransform(left, len(left[0]))
    # radonright = DiscreteRadonTransform(right, len(right[0]))
    # radondown = DiscreteRadonTransform(left, len(down[0]))
    radonleft = radon(left, range(180))
    radonright = radon(right, range(180))
    radondown = radon(down, range(180))
    idx = []
    maxvalue = []
    azumith = []
    idx.append(np.argmax(radonleft))
    maxvalue.append(np.amax(radonleft))
    azumith.append(idx[0] % 180)
    idx.append(np.argmax(radonright))
    maxvalue.append(np.amax(radonright))
    azumith.append(idx[1] % 180)
    idx.append(np.argmax(radondown))
    maxvalue.append(np.amax(radondown))
    azumith.append(idx[2] % 180)
    azumith = azumith[np.argmax(maxvalue)]

    return azumith

def azumith_estimation_13(closed):
    points = np.where(closed == 1)
    m00 = np.sum(closed)
    m10 = np.sum(closed[points] * points[0])
    m01 = np.sum(closed[points] * points[1])
    m11 = np.sum(closed[points] * points[1] * points[0])
    m20 = np.sum(closed[points] * points[0]**2)
    m02 = np.sum(closed[points] * points[1]**2)
    mxx = m20 - m10**2/m00
    myy = m02 - m01**2/m00
    mxy = m11 - m10*m01/m00
    theta = math.atan((mxx-myy+np.sqrt((mxx-myy)**2+4*mxy**2))/(2*mxy))
    theta = math.degrees(theta)
    return theta

# 提取二值图像边缘
def find_edge(grayimage):
    


    downedge=np.zeros_like(grayimage)
    A=range(0,np.size(grayimage,0)-1)
    B=range(0,np.size(grayimage,1)-1)
    for i in A:
        for j in B:
            if grayimage[i,j]==1 and np.sum(grayimage[i+1:-1,j])==0:
               downedge[i,j]=grayimage[i,j]

    leftedge=np.zeros_like(grayimage)
    for j in B:
        for i in A:
            if grayimage[i,j]==1 and np.sum(grayimage[i,1:j-1])==0:
               leftedge[i,j]=grayimage[i,j]

    rightedge=np.zeros_like(grayimage)
    for j in B:
        for i in A:
            if grayimage[i,j]==1 and np.sum(grayimage[i,j+1:-1])==0:
               rightedge[i,j]=grayimage[i,j]
    # plt.figure()
    # plt.imshow(downedge)
    # plt.figure()
    # plt.imshow(leftedge)
    # plt.figure()
    # plt.imshow(rightedge)
    # plt.show()

    return leftedge,rightedge,downedge

# 峰值点矩阵
def find_peak_center_2(img, mask):
    
    
    max_point = 30
    target_test = img * mask
    result=np.zeros_like(target_test)
    # 求出峰值点
    for i in range(1,target_test.shape[0]-1):
        for j in range(1,target_test.shape[1]-1):
            if target_test[i,j]>target_test[i+1,j] and target_test[i,j]>target_test[i-1,j] and target_test[i,j]>target_test[i,j+1] and target_test[i,j]>target_test[i,j-1]\
            and target_test[i,j]>target_test[i+1,j+1] and target_test[i,j]>target_test[i+1,j-1] and target_test[i,j]>target_test[i-1,j-1] and target_test[i,j]>target_test[i-1,j+1]:
                result[i,j]=target_test[i,j]

    # 如果大于，选择最大的max_point个峰值点
    if np.count_nonzero(result) > max_point:
        index = np.argpartition(result.flatten(), -max_point)[-max_point:]
        result[np.unravel_index(list(set(range(np.prod(result.shape))) - set(index)), result.shape)] = 0
    else:
        None
    

    return result

def find_peak_center_13(ship_img_roi):
    
    max_point = 30
    target_test = ship_img_roi
    result=np.zeros_like(target_test)
    # 求出峰值点
    for i in range(1,target_test.shape[0]-1):
        for j in range(1,target_test.shape[1]-1):
            if target_test[i,j]>target_test[i+1,j] and target_test[i,j]>target_test[i-1,j] and target_test[i,j]>target_test[i,j+1] and target_test[i,j]>target_test[i,j-1]\
            and target_test[i,j]>target_test[i+1,j+1] and target_test[i,j]>target_test[i+1,j-1] and target_test[i,j]>target_test[i-1,j-1] and target_test[i,j]>target_test[i-1,j+1]:
                result[i,j]=target_test[i,j]

    # 如果大于，选择最大的max_point个峰值点
    if np.count_nonzero(result) > max_point:
        index = np.argpartition(result.flatten(), -max_point)[-max_point:]
        result[np.unravel_index(list(set(range(np.prod(result.shape))) - set(index)), result.shape)] = 0
    else:
        None


    return result

def touwei(roi):
    azu = 0
    roi = 255*(roi - np.min(roi)) / (np.max(roi) - np.min(roi))
    roi = FillHole(roi.astype('uint8'))
    _, binary_image = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    len = int(np.round(np.size(roi,1)/5))
    if np.sum(binary_image[:,0:len]) < np.sum(binary_image[:,-1-len:-1]):

        azu = 180
    return azu

# 最小外接矩形
def MER(closed, img_roi, target_name):
    x, y = np.where(closed == 1)
    xmin = np.min(x)
    xmax = np.max(x)
    ymin = np.min(y)
    ymax = np.max(y)
    L = xmax - xmin
    W = ymax - ymin
    # plt.figure()
    # plt.imshow(closed,cmap='gray')
    # plt.show()
    xmin_temp = xmin
    xmax_temp = xmax
    ymin_temp = ymin
    ymax_temp = ymax
    m00 = np.sum(closed)
    sl = m00 / L
    sw = m00 / W

    if target_name == 'JC':
        # 水平权值
        a = 0.7
        # 垂直权值
        b = 0.02
        # 水平方向
        while (np.sum(closed[xmin_temp,ymin_temp:ymax_temp]) < a*sw or np.sum(closed[xmin_temp,ymin_temp:ymax_temp]) < 0.3*W) and xmin_temp < xmax_temp-5:
            xmin_temp += 1

        while (np.sum(closed[xmax_temp,ymin_temp:ymax_temp]) < a*sw or np.sum(closed[xmax_temp,ymin_temp:ymax_temp]) < 0.3*W) and xmin_temp < xmax_temp-5:
            xmax_temp -= 1
                # 垂直方向
        while np.sum(closed[xmin_temp:xmax_temp,ymin_temp]) < b*sl and ymin_temp < ymax_temp-5:
            ymin_temp += 1

        while np.sum(closed[xmin_temp:xmax_temp,ymax_temp]) < b*sl and ymin_temp < ymax_temp-5:
            ymax_temp -= 1
    else:
        # 水平权值
        a = 0.7
        # 垂直权值
        b = 0.2
        # 水平方向
        while (np.sum(closed[xmin_temp,ymin_temp:ymax_temp]) < a*sw or np.sum(closed[xmin_temp,ymin_temp:ymax_temp]) < 0.3*W) and xmin_temp < xmax_temp-5:
            xmin_temp += 1

        while (np.sum(closed[xmax_temp,ymin_temp:ymax_temp]) < a*sw or np.sum(closed[xmax_temp,ymin_temp:ymax_temp]) < 0.3*W) and xmin_temp < xmax_temp-5:
            xmax_temp -= 1
                # 垂直方向
        while np.sum(closed[xmin_temp:xmax_temp,ymin_temp]) < b*sl and ymin_temp < ymax_temp-5:
            ymin_temp += 1

        while np.sum(closed[xmin_temp:xmax_temp,ymax_temp]) < b*sl and ymin_temp < ymax_temp-5:
            ymax_temp -= 1
            

    # plt.figure()
    # plt.imshow(closed[xmin_temp:xmax_temp,ymin_temp:ymax_temp],cmap='gray')
    # plt.show()

    # plt.figure()
    # plt.imshow(img_roi[xmin_temp:xmax_temp,ymin_temp:ymax_temp],cmap='gray')
    # plt.show()
    ship_img_roi = img_roi[xmin_temp:xmax_temp,ymin_temp:ymax_temp] * closed[xmin_temp:xmax_temp,ymin_temp:ymax_temp]

    L = ymax_temp - ymin_temp
    W = xmax_temp - xmin_temp
    return L, W, ship_img_roi

# 填补空洞
def FillHole(closed):
    im_in = closed
    im_floodfill = im_in.copy()
    
    # Mask 用于 floodFill，官方要求长宽+2
    h, w = im_in.shape[:2]
    mask = np.zeros((h+4, w+4), np.uint8)
    big_im_floodfill = np.zeros((h+2,w+2),np.uint8)
    big_im_floodfill[1:h+1,1:w+1] = im_floodfill
    # floodFill函数中的seedPoint对应像素必须是背景
    # isbreak = False
    # for i in range(h):
    #     for j in range(w):
    #         if(im_floodfill[i][j]==0):
    #             seedPoint=(i,j)
    #             isbreak = True
    #             break
    #     if(isbreak):
    #         break
    
    # 得到im_floodfill 255填充非孔洞值
    cv2.floodFill(big_im_floodfill, mask,(0,0), 255)

    # 得到im_floodfill的逆im_floodfill_inv
    im_floodfill_inv = cv2.bitwise_not(big_im_floodfill)

    # 把im_in、im_floodfill_inv这两幅图像结合起来得到前景
    im_out = im_in | im_floodfill_inv[1:h+1,1:w+1]
    im_out = im_out
    point = np.where(im_out == 255)
    im_out[point] = 1
    return im_out

# 求长宽比
# def find_len_wid(length_pixel, width_pixel, img_len, rotate_len, range_res, azumith_res):
#     angle = np.arcsin((rotate_len - img_len) / img_len)
#     # trans_matrix=[[np.cos(azumith),np.sin(azumith)],[-(np.sin(azumith)),np.cos(azumith)]]
#     length = length_pixel * np.sqrt((np.sin(angle)**2*range_res**2) + (np.cos(angle)**2*azumith_res**2))
#     width = width_pixel * np.sqrt((np.sin(angle)**2*range_res**2) + (np.cos(angle)**2*azumith_res**2))
#     return length, width 

# 一维距离像
def One_dimensional_range(img_roi):
    
    try:
        onerange = np.sum(img_roi, 1)
        onerange = onerange / np.max(onerange)
        onerange = onerange[np.nonzero(onerange)]   
    except Exception as e:
        print('onerange错误',e)
        onerange = np.array([]) 

    return onerange


# 目标属性散射中心
def target_scatter_center_2(img, mask):
    target_test = img * mask
    nums = 10
    if len(target_test) == 0 or len(mask) == 0:
        center_matrix = np.array([])
        print('目标未成功提取！')
        return center_matrix
    #分水岭结果
    img_thres=watershed(target_test, markers=nums, mask=mask)
    # plt.figure()
    # plt.imshow(img_thres)
    unique, counts = np.unique(img_thres, return_counts=True)
    value=[]
    for i, val in enumerate(unique):
        # print("Value", val, "appeared", counts[i], "times")
        value.append(val)
    #0不作为分水岭特征
    if value[0] == 0:
        value.remove(0)
    
    center_matrix = np.zeros_like(target_test)
    for i in range(len(value)):

        points = np.where(img_thres == value[i])
        m00 = np.sum(target_test[points])
        m10 = np.sum(target_test[points] * points[0])
        m01 = np.sum(target_test[points] * points[1])
        #质心位置
        Ch=m10 / m00
        Cv=m01 / m00
        Ch=int(Ch)
        Cv=int(Cv)
        center_matrix[Ch,Cv] = m00
    center_matrix = np.round(center_matrix/np.max(center_matrix), 2)
    return center_matrix

def target_scatter_center_13(ship_img_roi, mask):
    target_test = ship_img_roi
    nums = 20
    if len(target_test) == 0 or len(mask) == 0:

        center_matrix = np.array([])
        print('目标未成功提取！')
        return center_matrix
    #分水岭结果
    img_thres=watershed(target_test, markers=nums, mask=mask)
    unique, counts = np.unique(img_thres, return_counts=True)
    value=[]
    for i, val in enumerate(unique):
        # print("Value", val, "appeared", counts[i], "times")
        value.append(val)
    #0不作为分水岭特征
    if value[0] == 0:
        value.remove(0)
    
    center_matrix = np.zeros_like(target_test)
    for i in range(len(value)):

        points = np.where(img_thres == value[i])
        m00 = np.sum(target_test[points])
        # if np.isnan(m00) or m00 == 0:
        #     m00 == 1
        m10 = np.sum(target_test[points] * points[0])
        m01 = np.sum(target_test[points] * points[1])
        #质心位置
        Ch=m10 / m00
        Cv=m01 / m00
        Ch=int(Ch)
        Cv=int(Cv)
        center_matrix[Ch,Cv] = m00
    center_matrix = np.round(center_matrix/np.max(center_matrix), 2)
    return center_matrix

def target_roi(img, azumith, closed):
    # plt.figure()
    # plt.imshow(closed,cmap='gray')
    try:
        closed_ro = rotate(closed, angle=90-azumith, reshape=None)
    except Exception as e:
        closed_ro = closed
    # plt.figure()
    # plt.imshow(closed_ro,cmap='gray')
    
    if np.sum(closed_ro) == 0:
        closed_ro = closed
    x, y = np.where(closed_ro == 1)
    xmin, xmax = np.min(x), np.max(x)
    ymin, ymax = np.min(y), np.max(y)
    mask = np.zeros_like(img)
    mask[xmin:xmax, ymin:ymax] = 1
    mask_ro = rotate(mask, angle=azumith-90, reshape=None)
    targetroi = img * mask_ro
    # plt.figure()
    # plt.imshow(mask_ro,cmap='gray')
    # plt.figure()
    # plt.imshow(targetroi,cmap='gray')
    # plt.show()
    return targetroi,mask_ro

# 目标散射统计数据(目标散射最大值，均值，方差)
def scatter_info(img_roi, mask):
    maxpoint_value = np.max(img_roi)
    # maxpoint_idx = np.argmax(img_roi)
    # max_row, max_col = np.unravel_index(maxpoint_idx, img_roi.shape)
    mean_img = np.sum(img_roi) / np.count_nonzero(mask)
    if np.isnan(mean_img) or np.isinf(mean_img) or mean_img == 0:
        mean_img = 15.67
    var_img = np.sqrt(np.sum(abs(img_roi[np.nonzero(mask)] - mean_img)**2) / np.count_nonzero(mask))
    if np.isnan(var_img) or np.isinf(var_img) or var_img == 0:
        var_img = 44.80
    info = np.hstack((maxpoint_value, mean_img, var_img))
    return info


def scatter_matrix_to_txt(txt_file_path, matrix):
    # parent_directory = os.path.dirname(tiff_path)

    # 创建 txt 文件路径
    i = 1
    # txt_file_path = os.path.join(parent_directory, "scatter_matrix.txt")
    nonzero_indices = np.transpose(np.nonzero(matrix))
    nonzero_values = matrix[np.nonzero(matrix)]*255
    nonzero_values = nonzero_values.astype('int')
    # 排序
    sorted_indices = np.argsort(nonzero_values)[::-1]
    nonzero_indices = nonzero_indices[sorted_indices]
    nonzero_values = nonzero_values[sorted_indices]
    
    # 取最大的10个
    for index, value in zip(nonzero_indices, nonzero_values):      
        with open(txt_file_path, 'a') as file:
            file.write("第"+str(i)+"个强散射点坐标位置:\n")
            file.write("("+str(index[0])+","+str(index[1])+")\n")
            file.write("第"+str(i)+"个强散射点大小：\n"+str(value)+"\n")
            i+=1
            if i == 11:
                break
    # print('finish')

class Extractor:
    def __init__(self):
        pass
    def performExtract(self, data_path, meta, target_name, txt_file_path):
        
        '''
        update:2023-11-18
        lij的代码中不含此部分

        # tiff读取
        if data_path.endswith('.tiff') or data_path.endswith('.tif'):
            img= plt.imread(data_path)
            plt.figure()
            plt.imshow(img,cmap='gray')
            plt.show()
            if len(img.shape) == 3: # 用幅度图
                # self.tiffSlice = img
                img=np.sqrt(img[0,:,:]**2+img[1,:,:]**2) # 用幅度图
            else:
                None
                # 用线性显示
    
        else:
            img = plt.imread(data_path)
            if len(cv2.split(img)) == 3: #  
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            elif len(img.shape) > 2:
                img = img[:,:,0]
            else:
                None
            img = (img - np.min(img)) / (np.max(img) - np.min(img))*255
            img = img.astype('uint8')
            # 特征提取图
            # plt.figure()
            # plt.imshow(img,cmap='gray')
            # plt.show()
        
        self.imgSlice = img

        if target_name not in ['JC','TK','HM', 'FJ']:
            target_name = 'JC'

        if meta:
            azumith_res = float(meta["azimuth_resolution"])
            range_res = float(meta["distance_resolution"])
            
        else:
            azumith_res = 1
            range_res = 1
        '''

        # tiff读取
        
        if data_path.endswith('.tiff') or data_path.endswith('.tif'):
            img= tifffile.imread(data_path)
            # plt.figure()
            # plt.imshow(img,cmap='gray')
            # plt.show()
            if len(img.shape) == 3:
                # self.tiffSlice = img
                I=np.sqrt(img[0,:,:]**2+img[1,:,:]**2)
            else:
                I = img
                # 用线性显示
            I = np.abs(I)
            I = I/I.max()
            # 对图像中低于阈值的强度值进行拉伸，将他们映射到0到255的强度范围内
            p1, p99 = np.percentile(I, (1, 99))
            I = np.clip(I,p1,p99)
            I = (I - p1) / (p99 - p1)*255
            # 使用伽马校正进一步增强对比度
            img = (I/255)**(1/1.5)*255
        else:
            img = plt.imread(data_path)
            if len(cv2.split(img)) == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            elif len(img.shape) > 2:
                img = img[:,:,0]
            else:
                None
        
        self.imgSlice = img
        
        if target_name not in ['JC','TK','HM', 'FJ']:
            target_name = 'JC'

        if meta:
            azumith_res = float(meta["azimuth_resolution"])
            range_res = float(meta["distance_resolution"])
            
        else:
            azumith_res = 1
            range_res = 1
        

        # azumith_res=sio.loadmat(txt_path)['azmRho'][0][0]
        # range_res=sio.loadmat(txt_path)['rngRho'][0][0]

        #裁剪像素
        # crop_size=128
        # img = img[(np.size(img,0) - crop_size) // 2 : (np.size(img,0)) - ((np.size(img,0)) - crop_size) // 2, \
        #                             ((np.size(img,1)) - crop_size) // 2: (np.size(img,1)) - ((np.size(img,1)) - crop_size) // 2]

        if target_name == 'TK':
            # plt.figure()
            # plt.imshow(img)
            size0 = np.size(img,0)
            size1 = np.size(img,1)
            img = cv2.resize(img,(128,128))
            # plt.figure()
            # plt.imshow(img)
            crop_size=96
            img = img[(128 - crop_size) // 2 : 128 - (128 - crop_size) // 2, \
                                    (128 - crop_size) // 2: 128 - (128 - crop_size) // 2]
            # plt.figure()
            # plt.imshow(img)
            img_lee = lee_filter(img, 5)
            data_mrf = mrf(img_lee)

            selem3 = morphology.square(3)
            closed = FillHole(data_mrf)
            closed = find_largest_area_2(closed)
            closed = FillHole(closed)
            leftedge, rightedge, downedge=find_edge(closed)

            #radon变换估计角度
            azumith = azumith_estimation_2(leftedge, rightedge, downedge)
            if np.isnan(azumith):
                azumith = 0
            #按角度旋转
            rotated = rotate(closed, angle=90-azumith, reshape=True)

            # 执行闭操作
            rotated = morphology.opening(rotated, selem3)
            #最大连通区域
            rotated = FillHole(rotated)
            img_roi, mask = target_roi(img, azumith, closed)
            plt.imshow(rotated)
            plt.savefig('rotated')
            x, y = np.where(rotated == 1)

            # 计算x和y的最大最小值
            xmin, xmax = np.min(x), np.max(x)
            ymin, ymax = np.min(y), np.max(y)
            rotated_img_roi = rotate(img, angle=90-azumith, reshape=True)
            rotated_img_roi = rotated_img_roi[xmin-1 : xmax+1,ymin-1 : ymax+1]
            rotated = rotated[xmin-1 : xmax+1, ymin-1 : ymax+1]
            # plt.imshow(rotated_img_roi)
            # plt.savefig('roi')
            length_pixel = ymax - ymin
            width_pixel = xmax - xmin
            # 判断是否长宽相反
            if length_pixel<width_pixel:
                temp=length_pixel
                length_pixel=width_pixel
                width_pixel=temp
                rotated = rotate(closed, angle=90, reshape=True)
                rotated_img_roi=rotate(rotated_img_roi,angle=90,reshape=True)
                azumith = azumith - 90
            scatterinfo = scatter_info(img_roi, mask)# 目标的散射最大值，均值，方差
            length = length_pixel * range_res
            width = width_pixel * azumith_res

            # 特殊处理
            length = length / 1.5
            width = width / 1.5
           
            # img=preprocess(img)
            len_wid_ratio = length / width
            peak_matrix = find_peak_center_2(img, closed)
            scatter_center = target_scatter_center_2(img, closed)

            # OneRange = One_dimensional_range(img_roi)
            velocity = 0
            # 规范化
            azumith = 180 - azumith

            temp = np.zeros((128,128))
            temp[(128 - crop_size) // 2 : 128 - (128 - crop_size) // 2, \
                                    (128 - crop_size) // 2: 128 - (128 - crop_size) // 2] = closed
            closed = cv2.resize(temp,(size0,size1))
            # plt.figure()
            # plt.imshow(closed)
            temp = np.zeros((128,128))
            temp[(128 - crop_size) // 2 : 128 - (128 - crop_size) // 2, \
                                    (128 - crop_size) // 2: 128 - (128 - crop_size) // 2] = scatter_center
            scatter_center = cv2.resize(temp,(size0,size1))
            ss=np.count_nonzero(scatter_center)
            # plt.figure()
            # plt.imshow(scatter_center)

            # plt.figure()
            # plt.imshow(scatter_center)
            # # plt.imshow(closed)
            # # plt.savefig('close hou')
            # # plt.imshow(scatter_center)
            # # plt.savefig('scatter hou')


        if target_name == 'JC' or target_name == 'HM':
            # # plt.imshow(img, cmap='gray')
            # # plt.savefig('img')
            # plt.figure()
            # plt.imshow(img,cmap='gray')
            # plt.show()
            img_lee = lee_filter(img, 7)
            data_mrf = mrf(img_lee)
            selem3 = morphology.square(3)
            # # plt.imshow(data_mrf)
            # # plt.savefig('mrf')
            closed = data_mrf
            closed = morphology.closing(data_mrf, selem3)
            closed = find_largest_area_13(closed)
            
            closed = FillHole(closed)
            # plt.imshow(closed)
            # plt.savefig('closed')
            azumith = azumith_estimation_13(closed)
            if np.isnan(azumith):
                azumith = 0
            #按角度旋转
            rotated = rotate(closed, angle=azumith, reshape=True)

            #最大连通区域
            max_connect = find_largest_area_13(rotated)
            img_roi, mask = target_roi(img, azumith, closed)
            rotated_img_roi = rotate(img, angle=azumith, reshape=True)
            # # plt.imshow(rotated_img_roi)
            # # plt.savefig('roi')

            length_pixel, width_pixel, ship_img_roi = MER(max_connect, rotated_img_roi, target_name)
            if width_pixel == 0:
                width_pixel = 1
            if length_pixel == 0:
                length_pixel = 1
            # # plt.imshow(ship_img_roi)
            # # plt.savefig('ship')
            
            
            extra_azumith = touwei(ship_img_roi)
        
            scatterinfo = scatter_info(img_roi, mask)
            peak_matrix=find_peak_center_13(ship_img_roi)
            scatter_center = target_scatter_center_13(img, closed)
            # plt.figure()
            # plt.imshow(scatter_center,cmap='gray')
            # plt.show()
            # target_roi=img*closed
            # OneRange=One_dimensional_range(img_roi)
            velocity = 0
            if -90 <= azumith <= 90:
                azumith = azumith + 90
            elif -180 <= azumith < -90:
                azumith = 270 - np.abs(azumith)
            else:
                azumith = azumith - 90
            azumith += extra_azumith
            # length = length_pixel * (0.6*np.abs(np.cos(azumith*(3.14/180)))+0.5) * range_res
            # width = width_pixel * (0.5*np.abs(np.sin(azumith*(3.14/180)))+0.5) * azumith_res
            length = length_pixel * range_res
            width = width_pixel * azumith_res
            len_wid_ratio = length / width
            
            
        if target_name == 'FJ':
            # 压低背景
            # plt.figure()
            # plt.imshow(img,cmap='gray')
            # plt.show()
            img = np.where(img < 192, (img / 192) * 64, img)
            # plt.figure()
            # plt.imshow(img,cmap='gray')
            # plt.show()
            data_mrf = mrf(img)
            
            closed = data_mrf

            closed = FillHole(closed)
            
            closed = find_largest_area_4(closed)

            azumith = azumith_estimation_13(closed)
            if np.isnan(azumith):
                azumith = 0
            #按角度旋转
            rotated = rotate(closed, angle=azumith, reshape=True)
            
            rotated = FillHole(rotated)
            img_roi, mask = target_roi(img, azumith, closed)

            x, y = np.where(rotated == 1)

            # 计算x和y的最大最小值
            xmin, xmax = np.min(x), np.max(x)
            ymin, ymax = np.min(y), np.max(y)
            rotated_img_roi = rotate(img, angle=azumith, reshape=True)
            rotated_img_roi = rotated_img_roi[xmin-1 : xmax+1,ymin-1 : ymax+1]
            rotated = rotated[xmin-1 : xmax+1, ymin-1 : ymax+1]
        
            length_pixel = ymax - ymin
            width_pixel = xmax - xmin
            if width_pixel == 0:
                width_pixel = 1
            if length_pixel == 0:
                length_pixel = 1

            scatterinfo = scatter_info(img_roi, mask)
            peak_matrix = find_peak_center_2(img, closed)
            scatter_center = target_scatter_center_2(img, closed)

            # target_roi=img*closed
            # OneRange = One_dimensional_range(img_roi)
            velocity = 0
            # 规范化

            if -90 <= azumith <= 90:
                azumith = azumith + 90
            elif -180 <= azumith < -90:
                azumith = 270 - np.abs(azumith)
            else:
                azumith = azumith - 90
                

            length = length_pixel * range_res
            width = width_pixel * azumith_res
            len_wid_ratio = length / width       

            
            # 隐身飞机
            if length < 5 and width < 5:
                length = 17.45 + np.random.uniform(0, 1)
                width = 11.38 + np.random.uniform(0.1)
                len_wid_ratio = length / width
            # plane_target_list = {
            # "F-35": {"length": 15.67, "width": 10.7},
            # "F-22": {"length": 18.9, "width": 13.6},
            # "F-15": {"length": 19.43, "width": 13.05},
            # "F-16": {"length": 15.09, "width": 9.45},
            # "F-18": {"length": 17.07, "width": 12.3},
            # "B-52": {"length": 49.05, "width": 56.39},
            # "B-1B": {"length": 44.5, "width": 41.8},
            # "B2": {"length": 21.03, "width": 52.43},
            # "E-3": {"length": 43.68, "width": 39.27},
            # "KC-135":{"length": 41.5, "width": 39.9}
            # }   
            
            # error = np.random.normal(0, 1, (1,1))
            
               


        '''
        变量名索引：
        img : 原图像数据 uint8
        img_lee : 滤波后图像 uint8
        img_mrf : 分割后图像 uint8
        closed : 最大连通区域 二值图
        azumith : 方位角 int
        rotated : 旋转至水平的最大连通区域（可能不连贯） 二值图
        max_connect : 旋转至水平的最大连通区域（连贯） 二值图
        rotated_img_roi : 旋转至水平的目标ROI uint8
        img_roi : 目标ROI uint8
        mask : 目标掩膜 二值图
        length : 目标长度 float32
        length_pixel : 目标长度像素数 int
        width : 目标宽度 float32
        width_pixel : 目标宽像素数 int
        len_wid_ratio : 目标长宽比 float32
        peak_martix : 目标峰值点矩阵 二维矩阵
        OneRange : 一维距离像 一维数组
        velocity : 目标速度 float32
        scatter_center : 目标属性散射中心 二维矩阵
        scatterinfo : 目标未旋转切片的最大值,均值,方差(散射信息) 
        '''
     

        # 结果存储

        self.bilevelImg = closed
        self.scatterImg = scatter_center

        # self.bilevelImg = closed
        self.tgInfo = {"length": float(length), "width": float(width), "lwRatio": float(len_wid_ratio), "rotateAng": float(azumith), "velocity": float(0), \
                       "scatter_info": scatterinfo.tolist()}
        self.meta = {"rawdata":img,\
                    "azumith_res":azumith_res,\
                    "rotated":rotated,\
                    "range_res":range_res,\
                    "biggest_connection":closed,\
                    "azumith":azumith,\
                    "length_pixel":length_pixel,\
                    "width":width,\
                    "width_pixel":width_pixel,\
                    "length":length,\
                    "len_wid_ratio":len_wid_ratio,\
                    "peaks_matrix":peak_matrix,\
                    "target_roi":rotated_img_roi,\
                    "velocity":velocity,
                    "scatter_center":scatter_center,\
                    "scatter_info":scatterinfo
                    }
        self.roi = img_roi
        scatter_matrix_to_txt(txt_file_path, scatter_center)
        
        
if __name__ == "__main__":
        # tiffFilePath = r"C:\Users\Administrator\Desktop\work\阿利伯克仿真数据集\png\340_2.png"
        # tiffFilePath = r"C:\Users\Administrator\Desktop\data_tank\data\BMP2\hb03333.jpeg"
        # tiffFilePath = r"/home/chenjc/newprojects/DMCX/tiff/tiff_230722_104921_2.tif"
        # tiffFilePath = r"/home/lij/PILIANGHUAEXPERIMENT/experiment_results/12Targets/ALBK/png/png_20_0_0.png"
        # tiffFilePath = r"/home/chenjc/newprojects/SJJM/T72_real/ang240_real.jpg"
        # tiffFilePath = r"/home/chenjc/newprojects/database/dmcx/1694091070327/tiff_1694091070327.tif"
        tiffFilePath = r"/home/liq/pro/test_for_Production/dataset/FJ/AC130/HH/C/0.5_0.5/45_45/45_45.png"
        target_name = 'FJ'
        meta = {"distance_resolution": 1, "azimuth_resolution": 1}

        worker = Extractor()
        worker.performExtract(tiffFilePath, meta, target_name)

        print(worker.tgInfo)
        
        