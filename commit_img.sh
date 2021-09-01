sudo docker build -t pfnet_img .
# sudo docker run --rm -it --network none --gpus all -v gaofen/img/val/image:/input_path -v test_img:/output_path pfnet_img
sudo docker tag pfnet_img registry.cn-beijing.aliyuncs.com/glotwo/img:pfnet_img
sudo docker push registry.cn-beijing.aliyuncs.com/glotwo/img:pfnet_img