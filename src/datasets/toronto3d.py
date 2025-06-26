import os
import sys
import torch
import shutil
import logging
import os.path as osp
from plyfile import PlyData
from typing import List
from torch_geometric.nn.pool.consecutive import consecutive_cluster
from torch_geometric.data import extract_zip

from src.datasets import BaseDataset
from src.data import Data, InstanceData
from src.datasets.toronto3d_config import *

DIR = os.path.dirname(os.path.realpath(__file__))
log = logging.getLogger(__name__)

# 多进程共享策略
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

__all__ = ['Toronto3D', 'MiniToronto3D']

########################################################################
#                                Utils                                #
########################################################################


def read_toronto3d_ply(
        filepath, 
        xyz=True, 
        intensity=True, 
        semantic=True,
        instance=False,  
        remap=False,
        rgb=True  # 新增RGB处理开关
    ):
    data = Data()
    
    with open(filepath, "rb") as f:
        ply = PlyData.read(f)
        vertex = ply['vertex']
        
        # 统一使用torch.FloatTensor类型化处理
        def safe_tensor(arr, dtype=np.float32):
            return torch.FloatTensor(np.ascontiguousarray(arr.astype(dtype)))
        
        # 1. 坐标处理（保持原样）
        if xyz:
            pos = torch.stack([
                safe_tensor(vertex['x']),
                safe_tensor(vertex['y']),
                safe_tensor(vertex['z'])
            ], dim=-1)
            pos_offset = pos[0]
            data.pos = pos - pos_offset
            data.pos_offset = pos_offset
            
        # 2. RGB处理（关键新增）
        if rgb:
            # 确保PLY文件包含颜色属性[网页1]
            
            data.rgb = torch.stack([
                safe_tensor(vertex['red'], dtype=np.uint8),
                safe_tensor(vertex['green'], dtype=np.uint8),
                safe_tensor(vertex['blue'], dtype=np.uint8)
            ], dim=-1).float() / 255.0  # 归一化到[0,1]

        # 3. 语义标签处理（保持原有逻辑）
        if semantic:
            labels = vertex['scalar_Label']
            if labels.dtype != np.int_:
                labels = labels.astype(np.int64)
            data.y = torch.LongTensor(np.ascontiguousarray(labels))
        
        # 4. 过滤无效点（保持原逻辑）
        if semantic and remap:
            mask = (data.y != TORONTO3D_IGNORE_ID).squeeze()
            filtered_data = Data()
            for key in data.keys:
                tensor = data[key]
                if tensor.size(0) == data.num_nodes:  
                    filtered_data[key] = tensor[mask]
            data = filtered_data
    
    return data

########################################################################
#                              Toronto3D                              #
########################################################################

class Toronto3D(BaseDataset):
    """Toronto3D数据集 (基于DALES结构调整) 🌟 关键类结构保持一致"""

    # 数据集元信息
    @property
    def class_names(self) -> List[str]:
        return CLASS_NAMES

    @property
    def num_classes(self) -> int:
        return TORONTO3D_NUM_CLASSES

    @property 
    def stuff_classes(self) -> List[int]:
        return STUFF_CLASSES

    @property
    def class_colors(self) -> List[List[int]]:
        return CLASS_COLORS.tolist()

    # 文件路径配置
    _zip_name = "Toronto_3D.zip"
    _unzip_name = "Toronto_3D"  # 🌟 明确解压目录名

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, "raw", self._unzip_name)

    # 数据集划分
    @property
    def all_base_cloud_ids(self) -> dict:
        return {
            'train': ['L001', 'L003'],
            'val': ['L004'],
            'test': ['L002']
        }

    # 核心方法 🌟 保持与DALES相同的方法结构
    def download_dataset(self) -> None:
        zip_path = osp.join(self.root, self._zip_name)
        
        # 1. 检查原始ZIP存在性
        if not osp.exists(zip_path):
            raise FileNotFoundError(f"请将 {self._zip_name} 放入 {self.root}")

        # 2. 检查是否已完成解压
        required_files = {'L001.ply', 'L002.ply', 'L003.ply', 'L004.ply'}
        existing_files = set(os.listdir(self.raw_dir)) if osp.exists(self.raw_dir) else set()
        
        # 🌟 关键修改：如果文件已存在则跳过解压
        if required_files.issubset(existing_files):
            log.info(f"检测到 {self.raw_dir} 已包含全部文件，跳过解压")
            return

        # 3. 执行解压操作
        log.info(f"开始解压 {self._zip_name}...")
        tmp_dir = osp.join(self.root, "raw", "tmp_extract")
        os.makedirs(tmp_dir, exist_ok=True)
        
        # 🌟 安全解压到临时目录
        extract_zip(zip_path, tmp_dir)
        
        # 4. 处理嵌套目录结构
        actual_files = []
        for root, _, files in os.walk(tmp_dir):
            if any(f in required_files for f in files):
                actual_files.extend(osp.join(root, f) for f in files if f in required_files)
        
        # 5. 移动文件到目标位置
        os.makedirs(self.raw_dir, exist_ok=True)
        for src in actual_files:
            dst = osp.join(self.raw_dir, osp.basename(src))
            if not osp.exists(dst):  # 🌟 避免重复移动
                shutil.move(src, dst)
        
        # 6. 清理临时目录
        shutil.rmtree(tmp_dir)
        
        # 7. 最终验证
        existing = set(os.listdir(self.raw_dir))
        if not required_files.issubset(existing):
            missing = required_files - existing
            raise RuntimeError(f"解压失败！缺失文件: {missing}")

    def read_single_raw_cloud(self, raw_cloud_path: str) -> Data:  # 🌟 必须实现
        """读取单个原始文件"""
        return read_toronto3d_ply(
            raw_cloud_path,
            intensity=True,
            semantic=True,
            remap=True,
            rgb=True  # 强制启用RGB
        )

    # 路径转换方法
    def id_to_relative_raw_path(self, id: str) -> str:
        """ID转相对路径"""
        return osp.join(self._unzip_name, f"{id}.ply")

    def processed_to_raw_path(self, processed_path: str) -> str:
        """处理路径转原始路径"""
        parts = processed_path.split(os.sep)
        cloud_id = osp.splitext(parts[-1])[0].split('_')[0]
        return osp.join(self.raw_dir, f"{cloud_id}.ply")

    # 处理增强
    def _process_single_cloud(self, p: str) -> None:
        """添加RGB维度校验"""
        data = self.read_single_raw_cloud(self.processed_to_raw_path(p))
        assert hasattr(data, 'rgb'), "RGB通道未正确加载"
        super()._process_single_cloud(p)

########################################################################
#                            MiniToronto3D                            #
########################################################################

class MiniToronto3D(Toronto3D):
    """迷你版数据集 🌟 继承结构保持一致"""
    _NUM_MINI = 1

    @property
    def all_cloud_ids(self) -> dict:
        return {k: v[:self._NUM_MINI] for k, v in super().all_cloud_ids.items()}

    @property
    def data_subdir_name(self) -> str:
        return "toronto3d"

    # 🌟 保持与MiniDALES相同的重写方法
    def process(self) -> None:
        self._pre_filter = None
        self._pre_transform = None
        super().process()

    def download(self) -> None:
        super().download()

