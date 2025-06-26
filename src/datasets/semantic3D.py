import os
import sys
import torch
import shutil
import logging
import os.path as osp
import filecmp
import time
import numpy as np
from typing import List
from torch_geometric.data import extract_zip

from src.datasets import BaseDataset
from src.data import Data
from src.datasets.semantic3D_config import *

DIR = os.path.dirname(os.path.realpath(__file__))
log = logging.getLogger(__name__)

# 多进程共享策略
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

__all__ = ['Semantic3D', 'MiniSemantic3D']

########################################################################
#                                Utils                                #
########################################################################
def safe_tensor(arr, dtype=np.float32, device='cpu'):
    """确保数据连续性和类型安全"""
    return torch.as_tensor(
        np.ascontiguousarray(arr.astype(dtype)), 
        device=device
    )

def read_semantic3d(
        data_path: str,
        label_path: str = None,
        xyz: bool = True,
        rgb: bool = True,
        intensity: bool = True,
        semantic: bool = True,
        max_points: int = None,
        verbose: bool = False,
        remap_ignored: bool = True
) -> Data:
    """优化版Semantic3D数据读取"""
    data = Data()
    
    if verbose:
        print(f"Reading Semantic3D data from: {data_path}")
    
    # 1. 数据加载
    try:
        points = np.loadtxt(data_path, dtype=np.float32, ndmin=2)
    except ValueError as e:
        raise RuntimeError(f"Error loading {data_path}: {str(e)}") from e
    
    # 2. 降采样处理
    indices = None
    if max_points and (len(points) > max_points):
        rng = np.random.default_rng(seed=42)
        indices = rng.choice(len(points), max_points, replace=False)
        indices = np.sort(indices)
        points = points[indices]
        if verbose:
            print(f"Subsampled to {max_points} points")
    
    # 3. 坐标处理
    if xyz:
        pos = safe_tensor(points[:, :3], dtype=np.float32)
        data.pos_offset = pos[0].clone()
        data.pos = pos - data.pos_offset
        
    # 4. 强度值处理
    if intensity and points.shape[1] > 3:
        data.intensity = safe_tensor(points[:, 3], dtype=np.float32).unsqueeze(-1)
        
    # 5. RGB处理
    if rgb and points.shape[1] >= 6:
        rgb_data = points[:, 4:7].astype(np.uint8)
        data.rgb = safe_tensor(rgb_data, dtype=np.uint8).float() / 255.0
    
    # 6. 标签处理
    labels = None
    if semantic and label_path:
        try:
            labels = np.loadtxt(label_path, dtype=np.int64, ndmin=1)
            
            if indices is not None:
                labels = labels[indices]
                
            if remap_ignored:
                labels[labels == 0] = SEMANTIC3D_IGNORE_ID
                
            data.y = safe_tensor(labels, dtype=np.int64)
            
        except Exception as e:
            raise RuntimeError(f"Error loading labels {label_path}: {str(e)}") from e
    
    # 7. 无效点过滤
    if semantic and remap_ignored and (labels is not None):
        valid_mask = (data.y != SEMANTIC3D_IGNORE_ID).squeeze().numpy()
        if not np.all(valid_mask):
            filtered_data = Data()
            
            for key in data.keys:
                if key not in ['pos', 'rgb', 'intensity', 'y']:
                    filtered_data[key] = data[key]
            
            valid_indices = torch.from_numpy(valid_mask)
            for key in ['pos', 'rgb', 'intensity', 'y']:
                if hasattr(data, key):
                    tensor = getattr(data, key)
                    if tensor.size(0) == len(valid_mask):
                        setattr(filtered_data, key, tensor[valid_indices])
            
            data = filtered_data
            if verbose:
                print(f"Filtered {np.sum(~valid_mask)} invalid points")
    
    # # 元数据附加
    # data.file_path = data_path
    # if label_path:
    #     data.label_path = label_path
    
    return data

########################################################################
#                              Semantic3D                             #
########################################################################

class Semantic3D(BaseDataset):
    """官方Semantic3D数据集实现"""

    @property
    def class_names(self) -> List[str]:
        return CLASS_NAMES

    @property
    def num_classes(self) -> int:
        return SEMANTIC3D_NUM_CLASSES

    @property 
    def stuff_classes(self) -> List[int]:
        return STUFF_CLASSES

    @property
    def class_colors(self) -> List[List[int]]:
        return CLASS_COLORS.tolist()

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, "raw")

    @property
    def all_base_cloud_ids(self) -> dict:
        return ALL_BASE_CLOUD_IDS

    def download_dataset(self) -> None:
        """适配实际文件名格式的验证逻辑"""
        # 创建目录结构
        for split in ['train', 'val', 'test']:
            os.makedirs(osp.join(self.raw_dir, split), exist_ok=True)
            """添加预存在性检查"""
        # 定义各划分的必需文件模式
        required_patterns = {
            split: [f"{scene}_*" for scene in scenes]
            for split, scenes in ALL_BASE_CLOUD_IDS.items()
        }
        
        # 检查是否已完成解压
        all_exist = True
        for split, patterns in required_patterns.items():
            split_dir = osp.join(self.raw_dir, split)
            existing_files = os.listdir(split_dir) if osp.exists(split_dir) else []
            
            # 验证每个场景至少有一个匹配文件
            for pattern in patterns:
                if not any(fn.startswith(pattern.split('_')[0]) for fn in existing_files):
                    all_exist = False
                    break
        
        if all_exist:
            log.info("检测到完整数据集，跳过解压")
            return
        # 处理数据文件
        self._process_data_archives()
        self._process_label_archive()

        # 修改后的验证逻辑
        for split, scenes in ALL_BASE_CLOUD_IDS.items():
            for scene in scenes:
                # 检查数据文件存在性（允许不同后缀）
                data_files = [
                    f for f in os.listdir(osp.join(self.raw_dir, split))
                    if f.startswith(scene) and f.endswith('.txt')
                ]
                if not data_files:
                    raise FileNotFoundError(f"{split}划分缺失数据文件: {scene}")

                # 检查标签文件（仅train/val）
                if split != 'test':
                    label_files = [
                        f for f in os.listdir(osp.join(self.raw_dir, split))
                        if f.startswith(scene) and f.endswith('.labels')
                    ]
                    if not label_files:
                        raise FileNotFoundError(f"{split}划分缺失标签文件: {scene}")

    def _process_data_archives(self):
        """处理数据压缩包，适配新文件名"""
        # 生成预期的文件名模式（例如：bildstein_station1_*.7z）
        expected_patterns = [
            f"{scene}_*.7z" 
            for split_list in ALL_BASE_CLOUD_IDS.values()
            for scene in split_list
        ]
        
        # 遍历匹配的文件
        for f in os.listdir(self.raw_dir):
            if any(f.startswith(p.split('_')[0]) for p in expected_patterns) and f.endswith('.7z'):
                self._process_scene_archive(f)

    def _process_scene_archive(self, archive: str):
        
        
        """处理单个场景压缩包，添加文件存在性检查"""
        scene_id = '_'.join(archive.split('_')[:2])
        split = self._get_split_for_scene(scene_id)
        if not split:
            return

        target_dir = osp.join(self.raw_dir, split)
        tmp_dir = osp.join(self.raw_dir, f"tmp_{scene_id}")
        
        # 清理临时目录
        if osp.exists(tmp_dir):
            shutil.rmtree(tmp_dir)
        os.makedirs(tmp_dir, exist_ok=True)

        # 解压到临时目录
        os.system(f'7z x "{osp.join(self.raw_dir, archive)}" -o"{tmp_dir}" -y')

        # 增强文件移动逻辑
        for root, _, files in os.walk(tmp_dir):
            for f in files:
                if f.startswith(scene_id) and (f.endswith('.txt') or f.endswith('.labels')):
                    src = osp.join(root, f)
                    dst = osp.join(target_dir, f)
                    
                    # 1. 检查文件存在性
                    if osp.exists(dst):
                        # 2. 校验文件内容是否一致（可选）
                        if filecmp.cmp(src, dst, shallow=False):
                            continue  # 相同文件跳过
                        else:
                            # 3. 版本化备份
                            backup_name = f"{osp.splitext(f)[0]}_bak_{int(time.time())}{osp.splitext(f)[1]}"
                            shutil.move(dst, osp.join(target_dir, backup_name))
                    
                    # 4. 原子化移动操作
                    try:
                        shutil.move(src, dst)
                    except shutil.Error as e:
                        log.error(f"文件移动失败: {str(e)}")
                        continue

        # 清理临时目录
        shutil.rmtree(tmp_dir, ignore_errors=True)
            


    def _process_label_archive(self):
        """处理标签压缩包"""
        # 解压到临时目录
        tmp_dir = osp.join(self.raw_dir, "tmp_labels")
        os.makedirs(tmp_dir, exist_ok=True)
        os.system(f'7z x "{osp.join(self.raw_dir, "sem8_labels_training.7z")}" -o"{tmp_dir}" -y')
        
        # 移动标签到对应目录
        for root, _, files in os.walk(tmp_dir):
            for f in files:
                if f.endswith('.labels'):
                    scene_id = '_'.join(f.split('_')[:2])
                    split = self._get_split_for_scene(scene_id)
                    if split in ['train', 'val']:
                        dest_dir = osp.join(self.raw_dir, split)
                        dest_path = osp.join(dest_dir, f)
                        
                        # 存在性检查
                        if osp.exists(dest_path):
                            if filecmp.cmp(osp.join(root, f), dest_path, shallow=False):
                                continue
                            else:
                                os.replace(osp.join(root, f), dest_path)  # 原子替换
                        else:
                            shutil.move(osp.join(root, f), dest_dir)
        
        shutil.rmtree(tmp_dir)

    def _get_split_for_scene(self, scene_id: str) -> str:
        """根据配置获取场景划分"""
        for split, scenes in ALL_BASE_CLOUD_IDS.items():
            if scene_id in scenes:
                return split
        return None

    def processed_to_raw_path(self, processed_path: str) -> str:
        """适配带特征后缀的文件名"""
        # 输入示例: processed/semantic3d/train/bildstein_station1_0.pth
        parts = osp.normpath(processed_path).split(os.sep)
        split_dir = next(p for p in parts if p in ['train', 'val', 'test'])
        
        # 提取基础场景ID
        filename = osp.splitext(parts[-1])[0]  # bildstein_station1_0
        scene_id = '_'.join(filename.split('_')[:2])  # bildstein_station1

        # 查找匹配的.txt文件
        target_dir = osp.join(self.raw_dir, split_dir)
        matched_files = [f for f in os.listdir(target_dir) 
                        if f.startswith(scene_id) and f.endswith('.txt')]
        
        if not matched_files:
            raise FileNotFoundError(f"找不到{scene_id}的原始文件")
        
        return osp.join(target_dir, matched_files[0])

    def read_single_raw_cloud(self, raw_cloud_path: str) -> Data:
        base_name = osp.splitext(raw_cloud_path)[0]
        label_path = base_name + '.labels'
        return read_semantic3d(
            raw_cloud_path,
            label_path=label_path if osp.exists(label_path) else None,
            intensity=True,
            semantic=True,
            rgb=True
        )
        # 新增路径转换方法（与Toronto3D接口一致）
    def id_to_relative_raw_path(self, id: str) -> str:
        return osp.join(id.split('_')[0], f"{id}.txt")
    def _process_single_cloud(self, p: str) -> None:
        raw_path = self.processed_to_raw_path(p)
        if not osp.exists(raw_path):
            raise FileNotFoundError(f"原始文件缺失: {raw_path}")
            
        data = self.read_single_raw_cloud(raw_path)
        assert data.pos.size(0) > 1000, "点云数据过小"
        super()._process_single_cloud(p)

########################################################################
#                            MiniSemantic3D                           #
########################################################################

class MiniSemantic3D(Semantic3D):
    """轻量版数据集"""
    _NUM_MINI = 2

    @property
    def all_base_cloud_ids(self) -> dict:
        return {k: v[:self._NUM_MINI] for k, v in ALL_BASE_CLOUD_IDS.items()}

    @property
    def data_subdir_name(self) -> str:
        return "semantic3d-mini"

    def process(self) -> None:
        self._pre_filter = lambda data: data.pos.size(0) < 5e5
        self._pre_transform = None
        super().process()