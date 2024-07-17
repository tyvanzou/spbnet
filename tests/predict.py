import sys
sys.path.append("..") #相对路径或绝对路径

import spbnet
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "7"

spbnet.predict("./config.predict.yaml")
