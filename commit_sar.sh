sudo docker build -t segformer_sar .
sudo docker tag segformer_sar registry.cn-beijing.aliyuncs.com/glotwo/sar:segformer
sudo docker push registry.cn-beijing.aliyuncs.com/glotwo/sar:segformer