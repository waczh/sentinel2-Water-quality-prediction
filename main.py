import os
from osgeo import gdal, osr
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
from torchviz import make_dot
import rasterio
from rasterio.features import rasterize
from sklearn.metrics import confusion_matrix
import numpy as np
from scipy.ndimage import zoom, map_coordinates
import pandas as pd
from torch.optim.lr_scheduler import CosineAnnealingLR
import xlsxwriter
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import rasterio
from rasterio.plot import show
from skimage.filters import threshold_otsu
from skimage import morphology
gdal.UseExceptions()
os.environ['GTIFF_SRS_SOURCE'] = 'GEOKEYS'
os.environ['PROJ_LIB'] = r'C:\share\proj'

torch.manual_seed(42)

def encoding(xlx_file):
    input_data = []
    metirc = []
    label = []
    for file_name in os.listdir(xlx_file):
        if file_name.endswith('.xlsx'):
            file_path = os.path.join(xlx_file, file_name)
            df = pd.read_excel(file_path)
            input_matrix = df.iloc[:, 0:9].values
            output_matrix = df.iloc[:, 10:19].values
            output_label = df.iloc[:,19].values
            input_data.append(input_matrix)
            metirc.append(output_matrix)
            label.append(output_label)
    input_data = np.vstack(input_data) if input_data else np.empty((0, 9))
    output_data = np.vstack(metirc) if metirc else np.empty((0, 10))
    labels = np.vstack(label) if label else np.empty((0, 10))
    return input_data, output_data, labels.reshape(-1)

def extract_water_body(file):
    # 使用MNDWI指数提取水体，计算公式为（B3-B11）/(B3+B11)
    data = gdal.Open(file, gdal.GA_ReadOnly)
    arr = data.ReadAsArray().astype(np.float32)  # (Channel, Height, Width)
    tresh = -0.05
    b3 = arr[1, :, :]  # B3波段
    # b8 = arr[3, :, :]  # B8波段
    b11 = arr[7, :, :]  # B11波段
    denominator = b3 + b11
    valid_mask = denominator != 0
    mndwi = np.zeros_like(denominator)
    mndwi[valid_mask] = (b3[valid_mask] - b11[valid_mask]) / denominator[valid_mask]
    mask = mndwi >= tresh
    binary_mask = mask.astype(np.uint8)
    plt.imshow(binary_mask)
    plt.show()
    return binary_mask.astype(int)

def extract_water_Otsu(file):
    with rasterio.open(file) as src:
        band2 = src.read(2) # green
        band4 = src.read(4) # NIR
        ndwi = (band2.astype(float) - band4.astype(float)) / (band2 + band4)
        thresh = threshold_otsu(ndwi)
        binary_water = ndwi > thresh
        cleaned_water = morphology.opening(binary_water, morphology.disk(3))
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 3, 1)
        show(band2, title='Red Band')
        plt.subplot(1, 3, 2)
        show(band4, title='NIR Band')
        plt.subplot(1, 3, 3)
        plt.imshow(cleaned_water, cmap='Blues')
        plt.title('Extracted Water Bodies')
        plt.show()
    return binary_water

def lossweight(train_set):

    train_label = train_set[:, -1].astype(int)
    unique_labels, label_counts = np.unique(train_label, return_counts=True)
    class_weights = 1 / (1 + label_counts)

    return class_weights

def datasplit(input_data, output_data, label):
    encoder = LabelEncoder()
    encoded_labels = encoder.fit_transform(label).astype(float)
    encoded_labels[encoded_labels==7] = np.nan
    sample_matrix = np.hstack((input_data, output_data, encoded_labels[:, np.newaxis]))
    sample_matrix = sample_matrix[~np.isnan(sample_matrix).any(axis=1)]
    sample_matrix[:,-1] = sample_matrix[:,-1].astype(int)
    np.random.seed(42)
    indices = np.random.permutation(sample_matrix.shape[0])
    total_samples = len(indices)
    train_size = int(total_samples * 5 / 10)
    val_size = int(total_samples * 2 / 10)
    test_size = total_samples - train_size - val_size
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    train_set = sample_matrix[train_indices]
    val_set = sample_matrix[val_indices]
    test_set = sample_matrix[test_indices]
    train_label_weight = lossweight(train_set)
    return train_set, val_set, test_set, train_label_weight

class Conv1DModel(nn.Module):
    def __init__(self):
        super(Conv1DModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3)
        self.batch_norm1 = nn.BatchNorm1d(32)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=1)
        self.batch_norm2 = nn.BatchNorm1d(64)
        self.batch_norm3 = nn.BatchNorm1d(5)
        self.fc1 = nn.Linear(64 * 3, 64)
        self.fc2 = nn.Linear(64, 9)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = nn.ReLU()(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = nn.ReLU()(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = nn.ReLU()(x)
        x = self.fc2(x)
        x = nn.ReLU()(x)
        return x


class SimpleMapper(nn.Module):
    def __init__(self, classcount):
        super(SimpleMapper, self).__init__()
        self.fc1 = nn.Linear(9, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, classcount)
        self._init_weights()
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.zeros_(m.bias)
    def forward(self, x):
        x = self.fc1(x)
        x = nn.ReLU()(x)
        x = self.fc2(x)
        x = nn.ReLU()(x)
        x = self.fc3(x)
        x = nn.ReLU()(x)
        x = self.fc4(x)
        return x
#=========================================================
class InverseModel(nn.Module):
    def __init__(self,inputchannel,metriccount):# 6 9
        super(InverseModel, self).__init__()
        self.fc1 = nn.Linear(inputchannel, 32)
        self.fc2 = nn.Linear(32,16)
        self.fc3 = nn.Linear(16,8)
        self.fc4 = nn.Linear(8,metriccount)
        self._init_weights()
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.zeros_(m.bias)
    def forward(self, x):
        x = self.fc1(x)
        x = nn.ReLU()(x)
        x = self.fc2(x)
        x = nn.ReLU()(x)
        x = self.fc3(x)
        x = nn.ReLU()(x)
        x = self.fc4(x)
        return x
#=========================================================

def compute_OA_AA(predictions, labels):

    predictions = torch.argmax(predictions, dim=1).numpy()
    labels = labels.cpu().numpy()
    cm = confusion_matrix(labels, predictions)
    TP = np.diagonal(cm)
    total = np.sum(cm)
    OA = np.sum(TP) / total

    return OA


def run(train_set, val_set, test_set, train_label_weight):
    forwardmodel = SimpleMapper(len(train_label_weight))
    inversemodel = InverseModel(inputchannel=6,metriccount=9)
    criterion_forward = nn.CrossEntropyLoss(weight=torch.tensor(train_label_weight, dtype=torch.float32))
    cirterion_inverse = nn.MSELoss()
    optimizer = optim.AdamW(forwardmodel.parameters(), lr=0.0001, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=3000, eta_min=0)
    num_epochs = 3000
    best_forward_loss = float('inf')
    best_inverse_loss = float('inf')
    t_input = torch.tensor(train_set[:, :9], dtype=torch.float32)
    t_target = torch.tensor(train_set[:, 18] - 1, dtype=torch.long)
    t_metric = torch.tensor(train_set[:, 9:18], dtype=torch.float32)
    v_input = torch.tensor(val_set[:, :9], dtype=torch.float32)
    v_target = torch.tensor(val_set[:, 18] - 1, dtype=torch.long)
    v_metirc = torch.tensor(val_set[:,9:18],dtype=torch.float32)
    s_input = torch.tensor(test_set[:, :9], dtype=torch.float32)
    s_target = torch.tensor(test_set[:, 18] - 1, dtype=torch.long)
    s_metirc = torch.tensor(test_set[:,9:18],dtype=torch.float32)
    train_losses_forward = []
    val_losses_forward = []
    train_losses_inverse = []
    val_losses_inverse = []
    best_val_OA = 0
    for epoch in tqdm(range(num_epochs)):
        forwardmodel.train()
        inversemodel.train()
        optimizer.zero_grad()

        t_output = forwardmodel(t_input)
        t_loss_forward = criterion_forward(t_output, t_target)
        t_loss_forward.backward()
        train_losses_forward.append(t_loss_forward.item() / t_input.shape[0])

        t_inverse = inversemodel(t_output.detach())
        t_inverse_loss = cirterion_inverse(t_inverse,t_metric)
        t_inverse_loss.backward()
        train_losses_inverse.append(t_inverse_loss.item()/t_input.shape[0])

        optimizer.step()

        if (epoch + 1) % 5 == 0:
            forwardmodel.eval()
            inversemodel.eval()
            with torch.no_grad():

                v_output = forwardmodel(v_input)
                v_loss_forward = criterion_forward(v_output, v_target)
                val_losses_forward.append(v_loss_forward.item() / v_input.shape[0])

                v_inverse = inversemodel(v_output.detach())
                v_inverse_loss = cirterion_inverse(v_inverse,v_metirc)
                val_losses_inverse.append(v_inverse_loss.item()/v_input.shape[0])


                if v_loss_forward <= best_forward_loss and v_inverse_loss <= best_inverse_loss:

                    best_forward_loss = v_loss_forward
                    best_inverse_loss = v_inverse_loss
                    print("=== model saving ===")
                    val_OA = compute_OA_AA(v_output, v_target)
                    torch.save(forwardmodel.state_dict(), r'./model/best_forward_model.pt')
                    torch.save(inversemodel.state_dict(), r'./model/best_inverse_model.pt')

                    print(f"val forward loss (per sample): {v_loss_forward.item() / v_input.shape[0]}")
                    print(f"val inverse loss (per sample): {v_inverse_loss.item() / v_input.shape[0]}")
                    print(f"val OA: {val_OA}")
        scheduler.step()
    print("testing..")
    test_model_forward = SimpleMapper(len(train_label_weight))
    test_model_inverse = InverseModel(inputchannel=6,metriccount=9)
    test_model_forward.load_state_dict(torch.load(r'./model/best_forward_model.pt'))
    test_model_inverse.load_state_dict(torch.load(r'./model/best_inverse_model.pt'))
    test_model_forward.eval()
    test_model_inverse.eval()
    with torch.no_grad():

        s_output = test_model_forward(s_input)
        s_loss_forward = criterion_forward(s_output, s_target)
        s_OA = compute_OA_AA(s_output, s_target)

        s_inverse = test_model_inverse(s_output.detach())
        s_inverse_loss = cirterion_inverse(s_inverse,s_metirc)

        print(f"test forward loss (per sample): {s_loss_forward.item() / s_input.shape[0]}")
        print(f"test inverse loss (per sample): {s_inverse_loss.item() / s_input.shape[0]}")
        print(f"test OA: {s_OA}")

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses_forward, label='Training Loss (forward)')
    plt.plot(np.arange(5, num_epochs + 1, 5), val_losses_forward, label='Validation Loss (forward)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Forward Loss over Epochs')
    plt.legend()
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.plot(train_losses_inverse, label='Training Loss (inverse)')
    plt.plot(np.arange(5, num_epochs + 1, 5), val_losses_inverse, label='Validation Loss (inverse)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Inverse Loss over Epochs')
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()

class metirc2class:

    def __init__(self,model_output,water_mask): # model_output (height,width,metric) torch.tensor
        """
        溶解氧D3, 高锰酸盐指数D6, 氨氮D7, 总磷（湖，库）D8, 总氮D9
        """
        self.model_output = model_output
        self.metric = model_output.shape[-1]
        self.mask = water_mask
        self.height = model_output.shape[0]
        self.width = model_output.shape[1]
        self.label = np.zeros([self.height,self.width])
        self.index = {"I":1,"II":2,"III":3,"IV":4,"V":5,"None":6}
    def D3(self,d3):
        if d3>=7.5:
            return "I"
        elif 7.5>d3>=6:
            return "II"
        elif 6>d3>=5:
            return "III"
        elif 5>d3>=3:
            return "IV"
        elif 3>d3>=2:
            return "V"
        else:
            return "None"
    def D6(self,d6):
        if d6<=2:
            return "I"
        elif 2<d6<=4:
            return "II"
        elif 4 < d6 <= 6:
            return "III"
        elif 6 < d6 <= 10:
            return "IV"
        elif 10 < d6 <= 15:
            return "V"
        else:
            return "None"
    def D7(self,d7):
        if d7<=0.15:
            return "I"
        elif 0.15<d7<=0.5:
            return "II"
        elif 0.5<d7<=1.0:
            return "III"
        elif 1.0<d7<=1.5:
            return "IV"
        elif 1.5<d7<=2.0:
            return "V"
        else:
            return "None"
    def D8(self,d8):
        if d8<= 0.01:
            return "I"
        elif 0.01<d8<=0.025:
            return "II"
        elif 0.025<d8<=0.05:
            return "III"
        elif 0.05<d8<=0.1:
            return "IV"
        elif 0.1<d8<=0.2:
            return "V"
        else:
            return "None"
    def D9(self,d9):
        if d9<=0.2:
            return "I"
        elif 0.2<d9<=0.5:
            return "II"
        elif 0.5<d9<=1.0:
            return "III"
        elif 1.0<d9<=1.5:
            return "IV"
        elif 1.5<d9<=2.0:
            return "V"
        else:
            return "None"
    def process(self):
        for row in range(self.height):
            for col in range(self.width):
                if self.mask[row,col]:
                    buffer = np.zeros(self.metric)
                    temp = self.model_output[row,col].squeeze()
                    buffer[2] = self.index[self.D3(temp[2])]
                    buffer[5] = self.index[self.D6(temp[5])]
                    buffer[6] = self.index[self.D7(temp[6])]
                    buffer[7] = self.index[self.D8(temp[7])]
                    buffer[8] = self.index[self.D8(temp[8])]
                    self.label[row,col] = np.max(buffer)
        return self.label

def forcast2(filepath, water_mask):
    model_save_path = r'./model/bestmodel.pt'
    data = gdal.Open(filepath, gdal.GA_ReadOnly)
    arr = data.ReadAsArray()
    arr = arr.astype(np.float32)
    arr = torch.tensor(arr, dtype=torch.float32).permute(1, 2, 0)
    height = arr.shape[0]
    width = arr.shape[1]
    output = torch.zeros([height, width, 6])
    test_model = SimpleMapper(classcount=6)
    test_model.load_state_dict(torch.load(model_save_path))
    test_model.eval()
    with torch.no_grad():
        for row in tqdm(range(height)):
            for col in tqdm(range(width)):
                if water_mask[row, col]:
                    pixel_input = arr[row, col].unsqueeze(0).unsqueeze(0)
                    pixel_output = test_model(pixel_input)  # (1, 6)
                    pixel_prob = torch.softmax(pixel_output, dim=-1)
                    output[row, col, :] = pixel_prob.squeeze(0).squeeze(0)
    print("Forecasting over! Visualize now..")
    max_prob, predicted_class = torch.max(output, dim=-1)
    arr_normalized = arr - arr.min()
    arr_normalized = arr_normalized / arr_normalized.max()
    vis_img = np.zeros((height, width, 3))
    category_colors = np.array([
        [255, 0, 0],
        [0, 255, 0],
        [0, 0, 255],
        [255, 255, 0],
        [255, 165, 0],
        [128, 0, 128],
    ])
    for row in range(height):
        for col in range(width):
            class_idx = predicted_class[row, col].item()
            prob = max_prob[row, col].item()
            color = category_colors[class_idx] * prob
            vis_img[row, col, :] = color
    overlay_img = np.where(vis_img == 0, arr_normalized[:,:,[0,1,2]], vis_img / 255.0 /2)
    plt.imshow(overlay_img)
    plt.colorbar()
    plt.title("Forecasted Image Overlay")
    plt.show()

if __name__=='__main__':

    # xlx_file = r'./'
    # mask_file = r'./cleandata4waterbody/T50TMK_20220810T030531.tif'
    # input_data, output_data,labels = encoding(xlx_file)
    # binary_water = extract_water_body(mask_file)
    # train_set, val_set, test_set, train_label_weight = datasplit(input_data, output_data, labels)
    # run(train_set, val_set, test_set, train_label_weight)
    # forcast2(r'./Sentinel2Data/output_Geotiff/T50TMK_20230830T030529.tif',binary_water)
    conv1d_model = Conv1DModel()
    simple_mapper_model = SimpleMapper(classcount=10)
    inverse_model = InverseModel(inputchannel=6, metriccount=9)

    input_shape_conv1d = (1, 1, 9)
    input_shape_simple_mapper = (1, 9)
    input_shape_inverse_model = (1, 6)

    input_data_conv1d = torch.randn(input_shape_conv1d)
    input_data_simple_mapper = torch.randn(input_shape_simple_mapper)
    input_data_inverse_model = torch.randn(input_shape_inverse_model)

    # output_conv1d = conv1d_model(input_data_conv1d)
    # graph_conv1d = make_dot(output_conv1d, params=dict(conv1d_model.named_parameters()))
    # graph_conv1d.view()
    # output_simple_mapper = simple_mapper_model(input_data_simple_mapper)
    # graph_simple_mapper = make_dot(output_simple_mapper, params=dict(simple_mapper_model.named_parameters()))
    # graph_simple_mapper.view()
    # #
    # output_inverse_model = inverse_model(input_data_inverse_model)
    # graph_inverse_model = make_dot(output_inverse_model, params=dict(inverse_model.named_parameters()))
    # graph_inverse_model.view()

