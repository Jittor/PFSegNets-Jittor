sudo docker build -t pfnet_sar .
# sudo docker run --rm -it --gpus all -v /home/gmh/project/yizhang/PFSegNets/gaofen/img/val/image:/input_path -v test_img:/output_path pfnet_sar
sudo docker tag pfnet_sar registry.cn-beijing.aliyuncs.com/glotwo/sar:pfnet_sar
sudo docker push registry.cn-beijing.aliyuncs.com/glotwo/sar:pfnet_sar