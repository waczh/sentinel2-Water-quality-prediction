from S2img_test import *
from preprocess import *

if __name__=='__main__':
    pos = [[116.3469555, 40.02560833] ,[116.36640556, 39.97468611] ,[116.320519444, 39.94045],
           [116.592811, 40.309662]]
    rect_pos = [[116.25196111, 40.082127777], [116.494569444, 40.007827777]]
    input_file_raw = r'./Sentinel2Data'
    output_file_r10m = r'./Sentinel2Data/output_Geotiff'
    sentinel2_L2AtoGeotiff(input_file_raw, output_file_r10m, pos, rect_pos)
    print('finished!')
    gt_data_folder = r'./Serial Data'
    match(gt_data_folder)




