FROM tensorflow/tensorflow:2.10.0rc3-gpu-jupyter
MAINTAINER Tom <tmbdev@gmail.com>
ENV DEBIAN_FRONTEND noninteractive
ENV PATH /usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:.
RUN pip install -U git+git://github.com/tmbdev/webdataset.git#egg=webdataset
RUN pip install -U typer
