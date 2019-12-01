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
COPY ./requirements.txt /code
WORKDIR /code
RUN pip install -r requirements.txt
COPY ./setup.py /code
RUN python setup.py install


COPY ./fuzzdom /code/fuzzdom
