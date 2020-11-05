# Use Caffe2 image as parent image
FROM caffe2/caffe2:snapshot-py2-cuda9.0-cudnn7-ubuntu16.04

RUN mv /usr/local/caffe2 /usr/local/caffe2_build
ENV Caffe2_DIR /usr/local/caffe2_build

ENV PYTHONPATH /usr/local/caffe2_build:${PYTHONPATH}
ENV LD_LIBRARY_PATH /usr/local/caffe2_build/lib:${LD_LIBRARY_PATH}

# Clone the Detectron repository
RUN git clone https://github.com/facebookresearch/densepose /densepose

# correct pyyaml version
COPY ./requirements.txt /densepose/
# update pip
RUN pip install --upgrade pip==10.0.1 -i http://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com

# Install Python dependencies
RUN pip install -i http://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com -r /densepose/requirements.txt

# Install the COCO API
RUN git clone https://github.com/cocodataset/cocoapi.git /cocoapi
WORKDIR /cocoapi/PythonAPI
RUN make install

# Go to Densepose root
WORKDIR /densepose

# Set up Python modules
RUN make

# [Optional] Build custom ops
RUN make ops

# update sources
# RUN echo "deb http://mirror.lzu.edu.cn/debian stable main contrib non-free" > /etc/apt/sources.list &&\
#     echo "deb-src http://mirror.lzu.edu.cn/debian stable main contrib non-free" >> /etc/apt/sources.list &&\
#     echo "deb http://mirror.lzu.edu.cn/debian stable-updates main contrib non-free" >> /etc/apt/sources.list &&\
#     echo "deb-src http://mirror.lzu.edu.cn/debian stable-updates main contrib non-free" >> /etc/apt/sources.list &&\
#     echo "deb http://mirror.lzu.edu.cn/debian/ stretch-backports main contrib non-free" >> /etc/apt/sources.list &&\
#     echo "deb-src http://mirror.lzu.edu.cn/debian/ stretch-backports main contrib non-free" >> /etc/apt/sources.list &&\
#     echo "deb http://mirror.lzu.edu.cn/debian-security/ stretch/updates main contrib non-free" >> /etc/apt/sources.list &&\
#     echo "deb-src http://mirror.lzu.edu.cn/debian-security/ stretch/updates main contrib non-free" >> /etc/apt/sources.list &&\
# 	apt-get update &&\
#     apt-get install -y ffmpeg

# RUN apt-get install -y vim

