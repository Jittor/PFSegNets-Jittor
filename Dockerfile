FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04
RUN apt-get update && apt-get install g++ python3-pip python3.7-dev git  -y
RUN ln -s /usr/bin/python3.7 /usr/bin/python
ENV DEBIAN_FRONTEND=noninteractive
RUN python -m pip install torch==1.4.0 torchvision==0.5.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN python -m pip install tensorboardX -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN python -m pip install tqdm -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN python -m pip install scikit-build -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN apt install cmake -y
RUN python -m pip install opencv-python -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN python -m pip install scipy -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN python -m pip install scikit-image -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN python -m pip install ipython -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN python -m pip install attr -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN python -m pip install jittor -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN python -m jittor.test.test_cudnn_op
RUN export DISABLE_MULTIPROCESSING=1
ENV DISABLE_MULTIPROCESSING=1

COPY . /workspace
WORKDIR /workspace
RUN rm /usr/bin/python3 && ln -s /usr/bin/python3.7 /usr/bin/python3
# 用来在build的时候，把模型下载到本地
RUN python -c "from jittor.models.res2net import res2net101_26w_4s;resnet = res2net101_26w_4s(pretrained=True)"

CMD ["python", "run.py", "/input_path", "/output_path"]
