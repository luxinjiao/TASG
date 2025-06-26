import numpy as np

########################################################################
#                         Dataset Information                          #
########################################################################

# Toronto-3D官方下载地址（需手动下载）
DATASET_URL = "https://github.com/WeikaiTan/Toronto-3D"

# 原始数据文件名
RAW_TAR_NAME = "Toronto-3D.zip" 
RAW_UNTAR_NAME = "Toronto-3D"

########################################################################
#                              Data splits                             #
########################################################################

# 官方推荐的训练/验证/测试划分
TILES = {
    'train': ['L001', 'L003'],  # 🟢 移除 L004
    'val': ['L004'],            # 单独保留为验证集
    'test': ['L002']
}

########################################################################
#                                Labels                                #
########################################################################

# 类别体系配置（包含9个原始类别+1个忽略类）
TORONTO3D_NUM_CLASSES = 8  # 有效训练类别数（排除忽略类）
TORONTO3D_IGNORE_ID = 0    # 忽略类ID

# 原始标签到训练标签的映射（原始9类→训练8类+忽略）
ID2TRAINID = np.array([
    8,  # 0 -> Unclassified -> 忽略类（ID=8）
    0,  # 1 -> Ground
    1,  # 2 -> RoadMarkings
    2,  # 3 -> Natural
    3,  # 4 -> Building 
    4,  # 5 -> UtilityLine
    5,  # 6 -> Pole
    6,  # 7 -> Car
    7,  # 8 -> Fence
])

# 训练标签名称（包含忽略类）
CLASS_NAMES = [
    'Ground',          # 0
    'RoadMarkings',    # 1
    'Natural',         # 2  
    'Building',        # 3
    'UtilityLine',     # 4
    'Pole',            # 5
    'Car',             # 6
    'Fence',           # 7
    'Ignored'          # 8 (Unclassified)
]

# 可视化颜色配置（与CLASS_NAMES顺序一致）
CLASS_COLORS = np.array([
    [128, 128, 128],  # Ground - 灰色
    [32, 255, 255],   # RoadMarkings - 青色
    [16, 128, 1],     # Natural - 深绿
    [0, 0, 255],      # Building - 蓝色
    [33, 255, 6],     # UtilityLine - 亮绿
    [252, 2, 255],    # Pole - 品红
    [253, 128, 8],    # Car - 橙色
    [255, 255, 10],   # Fence - 黄色
    [0, 0, 0]         # Ignored - 黑色
], dtype=np.uint8)

# 实例分割配置
MIN_OBJECT_SIZE = 50  # 更小的最小对象尺寸（适应城市小物体）
THING_CLASSES = [3, 4, 5, 6, 7]  # 可实例化的类别（Building, UtilityLine, Pole, Car, Fence）
STUFF_CLASSES = [0, 1, 2]        # 非实例类别（Ground, RoadMarkings, Natural）