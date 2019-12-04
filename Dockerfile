FROM pytorch/pytorch:latest
# FROM python:3.6

RUN conda install -c conda-forge opencv
RUN conda install -c conda-forge tensorflow mpi4py gensim pandas joblib scikit-learn
RUN conda install -c conda-forge tensorboard


RUN export PATH=/usr/local/cuda/bin:$PATH
RUN export CPATH=/usr/local/cuda/include:$CPATH
RUN export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
RUN export DYLD_LIBRARY_PATH=/usr/local/cuda/lib:$DYLD_LIBRARY_PATH

RUN mkdir -p /code
WORKDIR /code
RUN git clone https://github.com/stanfordnlp/miniwob-plusplus-demos.git ./miniwob-plusplus-demos

COPY ./requirements.txt /code
RUN pip install -r requirements.txt
COPY ./setup.py /code
RUN python setup.py install


COPY ./fuzzdom /code/fuzzdom
