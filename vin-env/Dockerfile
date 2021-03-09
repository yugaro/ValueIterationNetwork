FROM nvidia/cuda:10.1-cudnn7-runtime-ubuntu18.04
RUN apt-get update && apt-get install -y \
	sudo\
	wget\
	vim
WORKDIR /opt
RUN wget https://repo.anaconda.com/archive/Anaconda3-2020.07-Linux-x86_64.sh && \
	sh Anaconda3-2020.07-Linux-x86_64.sh -b -p /opt/anaconda3 && \
	rm -f Anaconda3-2020.07-Linux-x86_64.sh
ENV PATH /opt/anaconda3/bin:$PATH

RUN pip install --upgrade pip
RUN pip install chainer
RUN pip install progressbar
RUN pip install numpy
RUN pip install opencv-python
RUN pip install matplotlib
RUN pip install opencv-contrib-python
RUN apt-get install -y libgl1-mesa-dev
RUN pip install cupy-cuda80
WORKDIR /
CMD ["jupyter","lab","--ip=0.0.0.0","--allow-root","--LabApp.token=''"]
