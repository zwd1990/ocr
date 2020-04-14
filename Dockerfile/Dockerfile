FROM python:3.6
LABEL author="Lvjj"

#安装需要的python库
RUN pip3 install -i http://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com requests tqdm opencv-python numpy Shapely matplotlib Cython ipython tensorflow-gpu==1.14.0 pillow
WORKDIR /opts/python-project/ocr
#print()时在控制台正常显示中文
ENV PYTHONIOENCODING=utf-8