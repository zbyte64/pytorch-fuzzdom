FROM pytorch/pytorch:1.4-cuda10.1-cudnn7-runtime
# FROM python:3.6

RUN conda install -c conda-forge opencv
RUN conda install -c conda-forge tensorflow mpi4py gensim pandas joblib scikit-learn
RUN conda install -c conda-forge tensorboard


ENV PATH=/usr/local/cuda/bin:$PATH
ENV CPATH=/usr/local/cuda/include:$CPATH
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
ENV DYLD_LIBRARY_PATH=/usr/local/cuda/lib:$DYLD_LIBRARY_PATH
ENV CUDA cu101

RUN mkdir -p /code
WORKDIR /code
RUN git clone https://github.com/stanfordnlp/miniwob-plusplus-demos.git ./miniwob-plusplus-demos

RUN pip install torch-scatter==latest+$CUDA -f https://pytorch-geometric.com/whl/torch-1.4.0.html
RUN pip install torch-sparse==latest+$CUDA -f https://pytorch-geometric.com/whl/torch-1.4.0.html
RUN pip install torch-cluster==latest+$CUDA -f https://pytorch-geometric.com/whl/torch-1.4.0.html
RUN pip install torch-spline-conv==latest+$CUDA -f https://pytorch-geometric.com/whl/torch-1.4.0.html

COPY ./requirements.txt /code
RUN pip install -r requirements.txt
COPY ./setup.py /code
RUN python setup.py install


COPY ./fuzzdom /code/fuzzdom
