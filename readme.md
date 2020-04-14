# OCR

**\*\*\*\*\* 一个基于CTPN+CRNN的完整OCR项目\*\*\*\*\***

## 目录结构：

```
ocr--|--crnn_ctc(crnn模型)--|--train.py(模型训练)
     |                      |--predict.py(测试集预测)
     |                      |--inference_crnn_ctc.py(前向传播)
     |                      |--save_model.py(模型ckpt结果转换为tfserving所需格式)
     |                      |--output(结果)
     |                      |--logs(训练可视化日志)
     |--dataset(数据集)
     |--Dockerfile
     |--text-detection
     |--readme.md
     |--run.py(执行主程序)
```
## 训练及训练可视化过程
### 训练
进入crnn_ctc目录，运行train.py程序即可
```
cd crnn_ctc/
python3 train.py
```
### 训练可视化
进入crnn_ctc目录，启动TensorBoard服务(host后面替换上自己的主机IP)
```
cd crnn_ctc/
tensorboard --logdir=./logs --host ***.***.***.***
```
运行正常即可看到可视化界面URL，默认端口为6006

## ocr服务部署
其中source后面为tfserving格式文件路径
```
docker run --name tfserving-ocr \
        --hostname tfserving-ocr \
        -tid \
        --restart=on-failure:10 \
        -v  /etc/timezone:/etc/timezone \
        -v  /etc/localtime:/etc/localtime \
        -p 6501:8501 \
        -p 6500:8502 \
        --mount type=bind,source=/home/python-project/ocr/tfserving/crnn_ctc,target=/models/crnn_ctc \
        -e MODEL_NAME=crnn_ctc \
        -t tensorflow/serving &
```
## ctpn服务部署
其中source后面为tfserving格式文件路径
```
docker run --name tfserving-ctpn \
        --hostname tfserving-ctpn \
        -tid \
        --restart=on-failure:10 \
        -v  /etc/timezone:/etc/timezone \
        -v  /etc/localtime:/etc/localtime \
        -p 7501:8501 \
        -p 7500:8502 \
        --mount type=bind,source=/home/python-project/ocr/tfserving/ctpn,target=/models/ctpn \
        -e MODEL_NAME=ctpn \
        -t tensorflow/serving &
```