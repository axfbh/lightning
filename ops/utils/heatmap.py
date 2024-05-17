import cv2
import numpy as np
import torch
import torch.nn.functional as F


def fix_value(ipt):
    pix_max = np.max(ipt)  # 获取最大值和最小值 - 获取取值范围
    pix_min = np.min(ipt)
    base_value = np.abs(pix_min) + np.abs(pix_max)  # 获取两者距离
    base_rate = 255 / base_value  # 求每个单位之间的平均距离
    pix_left = base_rate * pix_min
    ipt = ipt * base_rate - pix_left  # 整体偏移使其最小值为0
    ipt[ipt < 0] = 0.  # 防止意外情况，增加程序健壮性
    ipt[ipt > 255] = 255.
    return ipt


def make_gaussian(width, height, sigma=3, center=None):
    '''
        generate 2d guassion heatmap
        反应该点是关键点的概率
    :return:
    '''

    x = np.arange(0, width, 1, float)
    y = np.arange(0, height, 1, float)[:, np.newaxis]

    # w,h 中心位置向外扩散减少概率
    if center is None:
        x0 = width // 2
        y0 = height // 2
    else:
        # 指定区域 扩散概率
        x0 = np.clip(center[0], a_min=0, a_max=width)
        y0 = np.clip(center[1], a_min=0, a_max=height)

    return np.exp(-4 * np.log(2) * ((x - x0) ** 2 + (y - y0) ** 2) / sigma ** 2)


class HeatMapFeatureShow:
    def __init__(self, alpha=0.4, beta=0.6, gamma=0):
        self.gamma = gamma
        self.beta = beta
        self.alpha = alpha

    def fit(self, x, y):
        """

        :param x: single features, numpy [C, N, M]
        :param y: heatmap size (S, S)
        :return:
        """
        gray_heatmap = self.concat_feature(x, y.shape[0])
        gray_heatmap = cv2.cvtColor(gray_heatmap, cv2.COLOR_BGR2GRAY)
        color_heatmap = cv2.applyColorMap(gray_heatmap, cv2.COLORMAP_RAINBOW)
        color_heatmap = cv2.cvtColor(color_heatmap, cv2.COLORMAP_PINK)
        color_heatmap = cv2.applyColorMap(color_heatmap, cv2.COLORMAP_JET)  # 将热力图应用于原始图像
        superimposed_img = cv2.addWeighted(y, self.alpha, color_heatmap, self.beta, self.gamma)
        return superimposed_img

    def concat_feature(self, feature, heatmap_size=None):
        result = np.sum(feature, axis=0)  # 将所有通道数据叠加
        im_list = fix_value(result).astype('uint8')
        im = cv2.cvtColor(im_list, cv2.COLOR_GRAY2BGR)
        if heatmap_size:
            im = cv2.resize(im, (heatmap_size, heatmap_size), cv2.INTER_LINEAR)  # Image.BILINEAR双线性插值
        return im


class HeatMapGaussPointGenerator:
    def __init__(self, sigma=3):
        """
        生成，高斯热力图 扩散点 mask
        :param sigma: 控制扩散的大小
        """
        self.sigma = sigma

    def fit(self, x, y, radius=80, center=None):
        """

        :param x:List  image size
        :param y: np.ndarray landmarks  [N, H, W] N : 同一张图生成 N 个扩散点
        :param radius: int 扩散半径, 控制可扩散范围
        :param center: List 扩散中心位置 (x,y)
        :return:
        """

        h, w = x
        gthmp = np.zeros((h, w))

        gaussMask = make_gaussian(radius, radius, self.sigma, center)

        for landmark in y:
            top_x, top_y = max(0, int(landmark[0] - radius / 2)), max(0,
                                                                      int(landmark[1] - radius / 2))  # 高斯图中心点和关键点重合，左上角

            bottom_x, bottom_y = min(w, int(landmark[0] + radius / 2)), min(h, int(landmark[1] + radius / 2))  # 右下脚坐标

            top_x_offset = top_x - int(landmark[0] - radius / 2)  # 在高斯图上的偏移
            top_y_offset = top_y - int(landmark[1] - radius / 2)  # 偏移

            gthmp[top_y:bottom_y, top_x:bottom_x] = gaussMask[top_y_offset:top_y_offset + bottom_y - top_y,
                                                    top_x_offset:top_x_offset + bottom_x - top_x]

        return gthmp


def get_activations_gradient(model):
    # 获取梯度
    return model.gradients


def get_activations(model, x):
    # 获取主干网络的输出
    x = model.feature_extractor(x)
    return x


class CAMHeatMapFeatureShow:
    def __init__(self, *, model, alpha=0.4, beta=0.6, gamma=0, COLOR_RAIN=False):
        """
        使用 grad 生成 热力图
        :param model: 模型的对象
        :param alpha:
        :param beta:
        :param gamma:
        :param COLOR_RAIN: bool 是否使用 RAINBOW 颜色

        """
        self.COLOR_RAIN = COLOR_RAIN
        self.model = model
        self.gamma = gamma
        self.beta = beta
        self.alpha = alpha

    def fit(self, x, y, image):
        """
        :param models: 模型
        :param x: Tensor 输入数据 单个 输入特性
        :param y: 分类结果
        :param image: Numpy 输入数据 单个 输入特性

        :return: heatmap + image
        """
        im_pre_prob = torch.softmax(y, 1)
        prob, prelab = torch.topk(im_pre_prob, 5)
        prelab = prelab.numpy().flatten()

        # 必须在 models 里面 使用 x.register_hook(self.activations_hook)，可使用在 多维特这个图，但必须是 2D 特征图
        y[:, prelab[0]].backward()  # 获取相对于模型参数的梯度

        gradients = get_activations_gradient(self.model)  # 获取模型的梯度
        mean_gradients = torch.mean(gradients, dim=[0, 2, 3])  # 计算梯度相应通道的均值
        activations = get_activations(self.model, x).detach()  # 获取输出的卷积特征
        for i in range(len(mean_gradients)):
            # 每个通道乘以相应的均值
            activations[:, i, :, :] *= mean_gradients[i]

        heatmap = torch.mean(activations, dim=1).squeeze()
        heatmap = F.relu(heatmap)
        heatmap /= torch.max(heatmap)
        heatmap = heatmap.numpy()

        heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
        heatmap = np.uint8(255 * heatmap)
        if self.COLOR_RAIN:
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_RAINBOW)
            heatmap = cv2.cvtColor(heatmap, cv2.COLORMAP_PINK)

        color_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        superimposed_img = cv2.addWeighted(image, self.alpha, color_heatmap, self.beta, self.gamma)
        return superimposed_img
