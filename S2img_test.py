import os
from osgeo import gdal, osr
import rasterio
from rasterio.features import rasterize
import numpy as np
from scipy.ndimage import zoom, map_coordinates
import pandas as pd
import xlsxwriter
gdal.UseExceptions()
os.environ['GTIFF_SRS_SOURCE'] = 'GEOKEYS'
os.environ['PROJ_LIB'] = r'C:\share\proj'
def getSRSPair(dataset):
    '''
    获得给定数据的投影参考系和地理参考系
    :param dataset: GDAL地理数据
    :return: 投影参考系和地理参考系
    '''
    prosrs = osr.SpatialReference()
    prosrs.ImportFromWkt(dataset.GetProjection())
    geosrs = prosrs.CloneGeogCS()
    return prosrs, geosrs


def lonlat2geo(dataset, lon, lat):
    '''
    将经纬度坐标转为投影坐标（具体的投影坐标系由给定数据确定）
    :param dataset: GDAL地理数据
    :param lon: 地理坐标lon经度
    :param lat: 地理坐标lat纬度
    :return: 经纬度坐标(lon, lat)对应的投影坐标
    '''

    prosrs, geosrs = getSRSPair(dataset)
    ct = osr.CoordinateTransformation(geosrs, prosrs)
    coords = ct.TransformPoint(lon, lat)
    return coords[:2]

def lat_lon_to_pixel_coordinates(trans, x, y):
    '''
    根据GDAL的六 参数模型将给定的投影或地理坐标转为影像图上坐标（行列号）
    :param dataset: GDAL地理数据
    :param x: 投影或地理坐标x
    :param y: 投影或地理坐标y
    :return: 影坐标或地理坐标(x, y)对应的影像图上行列号(row, col)
    '''
    a = np.array([[trans[1], trans[2]], [trans[4], trans[5]]])
    b = np.array([x - trans[0], y - trans[3]])
    return np.linalg.solve(a, b)  # 使用numpy的linalg.solve进行二元一次方程的求解

def merge_jp2_to_multiband_tiff(band_paths, band_resize, output_tiff_path, pos, rect):
    """
    合并多个单波段.jp2文件为一个多波段的GeoTIFF文件。
    :param band_paths: 单波段.jp2文件的路径列表
    :param output_tiff_path: 输出多波段GeoTIFF文件的路径
    """
    # 打开第一个波段以获取地理信息和数据大小
    first_band = gdal.Open(band_paths[0], gdal.GA_ReadOnly)
    geo_transform = first_band.GetGeoTransform()
    num_id = len(pos)
    Pos_id = np.zeros([num_id, 2]) # 每个监测点的横纵坐标
    for id, pos in enumerate(pos):
        pos_x, pos_y = lonlat2geo(first_band, pos[1], pos[0])
        # print(f"pos[1]:{pos[1]},pos[0]:{pos[0]}")
        Pos_id[id, 1], Pos_id[id, 0] = lat_lon_to_pixel_coordinates(geo_transform, pos_x, pos_y)
        # print(f"Pos_id[id, 1]:{Pos_id[id, 1]},Pos_id[id, 0]:{Pos_id[id, 0]}")

    rect_cur = np.zeros([2, 2], dtype='int')
    pos_x1, pos_y1 = lonlat2geo(first_band, rect[0][1], rect[0][0])
    rect_cur[0, 1], rect_cur[0, 0] = lat_lon_to_pixel_coordinates(geo_transform, pos_x1, pos_y1)
    pos_x2, pos_y2 = lonlat2geo(first_band, rect[1][1], rect[1][0])
    rect_cur[1, 1], rect_cur[1, 0] = lat_lon_to_pixel_coordinates(geo_transform, pos_x2, pos_y2)
    Height = int(rect_cur[1, 0]-rect_cur[0, 0]+1)
    Width = int(rect_cur[1, 1]-rect_cur[0, 1]+1)

    projection = first_band.GetProjection()
    data_type = first_band.GetRasterBand(1).DataType
    bands=9
    # 初始化输出多波段数据集
    driver = gdal.GetDriverByName('GTiff')
    dataset = driver.Create(output_tiff_path,
                            Width,
                            Height,
                            bands,
                            data_type,
                            options=['COMPRESS=LZW', 'BIGTIFF=YES'])

    # 计算新的 GeoTransform 参数
    new_gt = (
        geo_transform[0] + rect_cur[0, 1] * geo_transform[1],  # 左上角 X 坐标更新
        geo_transform[1],  # 像素宽度不变
        geo_transform[2],  # X 方向旋转不变
        geo_transform[3] + rect_cur[0, 0] * geo_transform[5],  # 左上角 Y 坐标更新
        geo_transform[4],  # Y 方向旋转不变
        geo_transform[5]  # 像素高度不变
    )
    dataset.SetGeoTransform(new_gt)
    dataset.SetProjection(projection)
    first_band = None
    reflect_value = np.zeros([num_id, 9])
    # 逐个波段读取数据并写入多波段数据集
    for i, band_path in enumerate(band_paths):
        band = gdal.Open(band_path, gdal.GA_ReadOnly)
        band_data = band.GetRasterBand(1).ReadAsArray()
        reflect_value[:, i] = map_coordinates(band_data, Pos_id.T, order=1, mode='reflect')
        band_data = band_data[rect_cur[0, 0]:rect_cur[0, 0]+Height,
                    rect_cur[0, 1]:rect_cur[0, 1]+Width]
        dataset.GetRasterBand(i + 1).WriteArray(band_data)
        band = None  # 关闭波段

    for i, band_path in enumerate(band_resize):
        band = gdal.Open(band_path, gdal.GA_ReadOnly)
        band_data = band.GetRasterBand(1).ReadAsArray()
        interpolated_image = zoom(band_data, 2, order=1)  # order=1 表示双线性插值
        reflect_value[:, i+4] = map_coordinates(interpolated_image, Pos_id.T, order=1, mode='reflect')
        dataset.GetRasterBand(i + 5).WriteArray(interpolated_image[rect_cur[0, 0]:rect_cur[0, 0]+Height,
                                                rect_cur[0, 1]:rect_cur[0, 1]+Width])
        band = None  # 关闭波段

    # 刷新缓存并构建金字塔
    dataset.FlushCache()
    dataset.BuildOverviews("NEAREST", [2, 4, 8, 16, 32])
    dataset = None  # 关闭数据集
    data = {
        '波段2': reflect_value[:, 0],
        '波段3': reflect_value[:, 1],
        '波段4': reflect_value[:, 2],
        '波段8': reflect_value[:, 3],
        '波段5': reflect_value[:, 4],
        '波段6': reflect_value[:, 5],
        '波段7': reflect_value[:, 6],
        '波段11': reflect_value[:, 7],
        '波段12': reflect_value[:, 8],
    }
    df = pd.DataFrame(data)
    # 写入 Excel 文件
    print(output_tiff_path)
    # output_file = output_tiff_path.split('.')[0].split('\\')[1] + '.xlsx'  # 输出文件名
    output_file = output_tiff_path.split('\\')[-1].split('.')[0] + '.xlsx'
    print(output_file)
    df.to_excel(output_file, index=False)
    print(f"DataFrame 已成功写入 {output_file}")

def sentinel2_L2AtoGeotiff(input_folder, output_folder,pos, rect):
# 指定Sentinel-2 L2A数据的目录路径 input_folder = r"input_data"
# output_folder = r"output_Geotiff"
    # 确保输出目录存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历目录中的所有SAFE文件夹
    for root, dirs, files in os.walk(input_folder):
        for dir_name in dirs:
            if dir_name.endswith('.SAFE'):
                safe_folder = os.path.join(root, dir_name)
                # L2A产品中，数据通常位于GRANULE子目录下的IMG_DATA文件夹
                granule_folder = os.path.join(safe_folder, "GRANULE")
                img_data_folders = [f for f in os.listdir(granule_folder) if os.path.isdir(os.path.join(granule_folder, f))]

                if img_data_folders:
                    img_data_folder_10 = os.path.join(granule_folder, img_data_folders[0], "IMG_DATA\\R10m")
                    data_file = []
                    # 转换每个波段
                    for band_file in os.listdir(img_data_folder_10):
                        if band_file.endswith('.jp2') and band_file.split('_')[-2][0]=='B':
                            input_path = os.path.join(img_data_folder_10, band_file)
                            data_file.append(input_path)

                    img_data_folder_20 = os.path.join(granule_folder, img_data_folders[0], "IMG_DATA\\R20m")
                    data_file_resize = []
                    for band_file in os.listdir(img_data_folder_20):
                        if band_file.endswith('.jp2') and (band_file.split('_')[-2] == 'B05' or
                                                           band_file.split('_')[-2] == 'B06' or
                                                           band_file.split('_')[-2] == 'B07' or
                                                           band_file.split('_')[-2] == 'B11' or
                                                           band_file.split('_')[-2] == 'B12'):
                            input_path = os.path.join(img_data_folder_20, band_file)
                            data_file_resize.append(input_path)

                    output_path = os.path.join(output_folder,
                                               os.path.splitext(data_file[0])[0].split('\\')[-1][:-8] + '.tif')
                    merge_jp2_to_multiband_tiff(data_file, data_file_resize, output_tiff_path=output_path, pos=pos, rect=rect)
                    print("completed.")


if __name__=='__main__':
    #清河闸、土城沟（花园路）、长河白石桥和怀柔水库经纬度位置
    pos = [[116.3469555, 40.02560833],[116.36640556, 39.97468611],[116.320519444, 39.94045],
          [116.592811, 40.309662]]
    # pos = [[116.349049, 40.026901],[116.365688, 39.977668],[116.32528, 39.941767],
    #      [116.592811, 40.309662]]
    rect_pos = [[116.25196111, 40.082127777], [116.494569444, 40.007827777]]
    input_file = r'./Sentinel2Data'
    output_file = r'./Sentinel2Data/output_Geotiff'
    sentinel2_L2AtoGeotiff(input_file, output_file, pos, rect_pos)
    print('finished!')



