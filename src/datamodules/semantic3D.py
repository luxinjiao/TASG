# src/datamodules/semantic3D.py
import logging
from src.datamodules.base import BaseDataModule
from src.datasets import Semantic3D, MiniSemantic3D  # 导入Semantic3D数据集类

log = logging.getLogger(__name__)

class Semantic3DDataModule(BaseDataModule):
    """LightningDataModule for Semantic3D dataset."""
    
    _DATASET_CLASS = Semantic3D          # 指定主数据集类
    _MINIDATASET_CLASS = MiniSemantic3D  # 指定迷你数据集类

if __name__ == "__main__":
    import hydra
    import omegaconf
    import pyrootutils

    root = pyrootutils.setup_root(__file__, pythonpath=True)
    cfg = omegaconf.OmegaConf.load(root / "configs/datamodule/semantic/semantic3D.yaml")  # 配置文件路径
    cfg.data_dir = str(root / "data")
    _ = hydra.utils.instantiate(cfg)