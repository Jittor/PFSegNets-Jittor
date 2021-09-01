sudo docker build -t pfnet_sar .
# sudo docker run --rm -it --network none  --gpus all -v gaofen/sar/val/image:/input_path -v test_img:/output_path pfnet_sar
sudo docker tag pfnet_sar registry.cn-beijing.aliyuncs.com/glotwo/sar:pfnet_sar
sudo docker push registry.cn-beijing.aliyuncs.com/glotwo/sar:pfnet_sar