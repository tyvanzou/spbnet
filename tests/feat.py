import sys

sys.path.append("..")  # 相对路径或绝对路径

from spbnet.visualize.feat import feat
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

feat("./config.feat.yaml")
