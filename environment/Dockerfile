# hash:sha256:a41559312bd1be8597bf2181d2f409253f3a2073c5c1071b6e2f98860d191888
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/mambaforge3:23.1.0-4-python3.10.12-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

ENV PIPELINE_URL="https://codeocean.allenneuraldynamics.org/capsule/6519101"
ENV PIPELINE_VERSION="1.0"
ENV VERSION="1.0"
ENV DFF_EXTRACTION_URL=https://github.com/AllenNeuralDynamics/aind-fip-dff

RUN pip3 install -U --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu

RUN pip3 install -U --no-cache-dir \
    aind-data-schema==0.38.0 \
    aind-ophys-utils \
    hdmf-zarr \
    matplotlib \
    pynwb \
    scikit-learn \
    statsmodels \
    aind-metadata-upgrader
