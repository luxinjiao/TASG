# src/datasets/semantic3d_config.py
import numpy as np

########################################################################
#                         Dataset Configuration                       #
########################################################################

# ======================== 数据集元信息 =========================
SEMANTIC3D_NUM_CLASSES = 9  # 包含未标注类别的总数
SEMANTIC3D_IGNORE_ID = 0     # 未标注点的标签ID

# ======================== 数据划分配置 =========================
ALL_BASE_CLOUD_IDS = {
    'train': [
        'bildstein_station1', 'bildstein_station3',
        'domfountain_station1', 'domfountain_station3',
        'neugasse_station1', 'sg27_station1',
        'sg27_station2', 'sg27_station5',
        'sg27_station9', 'sg28_station4',
        'untermaederbrunnen_station1'
    ],
    'val': [
        'bildstein_station5', 
        'domfountain_station2',
        'sg27_station4', 
        'untermaederbrunnen_station3'
    ],
    'test': [
        'MarketplaceFeldkirch_Station4',
        'StGallenCathedral_station6',
        'StGallenCathedral_station3'
    ]
}

# ======================== 类别体系配置 ========================
# ID到类别名称的映射
CLASS_NAMES = {
    0: "unlabeled",
    1: "man-made terrain",
    2: "natural terrain",
    3: "high vegetation",
    4: "low vegetation",
    5: "buildings",
    6: "hardscape",
    7: "scanning artefacts",
    8: "cars"
}

# 类别名称到ID的反向映射
NAME_TO_LABEL = {v.lower(): k for k, v in CLASS_NAMES.items()}

# 原始标签到训练标签的映射
RAW2TRAIN = np.array([
    0,  # 0 -> unlabeled (忽略)
    1,  # 1 -> man-made terrain
    2,  # 2 -> natural terrain
    3,  # 3 -> high vegetation
    4,  # 4 -> low vegetation
    5,  # 5 -> buildings
    6,  # 6 -> hardscape
    7,  # 7 -> scanning artefacts
    8   # 8 -> cars
], dtype=np.int64)

# ======================== 可视化配置 ========================
CLASS_COLORS = np.array([
    [0,   0,   0],    # 0: unlabeled
    [233, 229, 107],  # 1: man-made terrain
    [95,  156, 196],  # 2: natural terrain
    [77,  174, 84],   # 3: high vegetation
    [108, 135, 75],   # 4: low vegetation
    [179, 116, 81],   # 5: buildings
    [241, 149, 131],  # 6: hardscape
    [81,  163, 148],  # 7: scanning artefacts
    [223, 52,  52]    # 8: cars
], dtype=np.uint8)

# ====================== 实例分割配置 =======================
STUFF_CLASSES = [1, 2, 5, 6]  # 不可移动类别: 地形/建筑/硬景观
THING_CLASSES = [3, 4, 8]      # 可移动类别: 植被/车辆

# ====================== 实用函数 ========================
def semantic_name_to_label(name: str) -> int:
    """将语义名称转换为标签ID"""
    normalized = name.lower().strip()
    # 处理常见拼写变体
    normalized = normalized.replace("artifact", "artefact")
    return NAME_TO_LABEL.get(normalized, SEMANTIC3D_IGNORE_ID)