## Dockerfile to build VAE-Gumbel-Softmax container image

FROM python:2.7.14
MAINTAINER Vithursan Thangarasa

# dependencies
RUN \
    apt-get -qq -y update \
    && \
    pip install -U \
    numpy \
    holoviews \
    jupyter \
    pandas \
    matplotlib \
    seaborn \
    tqdm \
    packaging \
    appdirs \
    tensorflow

COPY ./ /root/vae_gumbel_softmax

WORKDIR /root/vae_gumbel_softmax
RUN mkdir /root/vae_gumbel_softmax/results

CMD python vae_gumbel_softmax.py
