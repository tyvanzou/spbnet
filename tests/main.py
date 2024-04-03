import spbnet
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

spbnet.finetune("./config.example.yaml")
