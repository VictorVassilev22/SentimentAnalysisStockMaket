#!/bin/bash
cd /home/"${USER}" || exit 1
wget https://packages.confluent.io/archive/7.0/confluent-community-7.0.1.tar.gz
tar -xf confluent-community-7.0.1.tar.gz
sudo mv confluent-7.0.1 kafka
pip install pyspark
wget https://dlcdn.apache.org/hadoop/common/hadoop-3.3.1/hadoop-3.3.1.tar.gz
wget https://downloads.apache.org/hadoop/common/hadoop-3.3.1/hadoop-3.3.1.tar.gz.sha512
shasum -a 512 hadoop-3.3.1.tar.gz
tar -xzvf hadoop-3.3.1.tar.gz
sudo mv hadoop-3.3.1 /usr/local/hadoop
#Setups kafka and spark
#configure hadoops java home
#SET env variables of spark and kafka