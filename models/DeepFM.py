import tensorflow as tf
import tensorflow.keras as keras
import argparse
from data import Data
from layers import FMLayer

parser = argparse.ArgumentParser()
parser.add_argument("--result",type=str,default='../results/result.txt')
parser.add_argument("--file",type=str,default='E:/project/rec_movielens/data/')
parser.add_argument("--embed_size",type=int,default=128)
parser.add_argument("--lr", type=float,default=1e-3)
parser.add_argument("--batch",type=int,default=1024)
parser.add_argument("--epochs",type=int,default=10)

arg = parser.parse_args()


class DeepFM(keras.Model):
    def __init__(self, feature_list):
        self.fm = FMLayer()