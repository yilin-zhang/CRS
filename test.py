from setup import setup_dataset
import tarfile
from configs import *
import os

setup_dataset()
#tf = tarfile.open('cache/dataset/mcgill-billboard.tar.gz', 'r:gz')
#tf.extractall('cache/dataset/')
#for file in os.listdir('cache/dataset/McGill-Billboard'):
#    print(file)

