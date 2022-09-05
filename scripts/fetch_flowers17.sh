#!/bin/bash


#curl -O http://www.robots.ox.ac.uk/~vgg/data/flowers/17/17flowers.tgz
#curl -O http://www.robots.ox.ac.uk/~vgg/data/flowers/17/datasplits.mat
#tar -xf 17flowers.tgz
python flowers_partition.py 1
rm 17flowers.tgz
rm datasplits.mat
rm -rf jpg
