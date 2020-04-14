FROM python:3.6
LABEL author="Lvjj"

#安装需要的python库
RUN pip install -i http://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com tqdm==4.11.2 opencv-python==3.4.0.12 Shapely==1.6.4.post1 matplotlib==1.5.3 numpy==1.14.2 tensorflow==1.4.0 Cython==0.24.1 ipython==5.1.0

#print()时在控制台正常显示中文
ENV PYTHONIOENCODING=utf-8