# hash:sha256:a41559312bd1be8597bf2181d2f409253f3a2073c5c1071b6e2f98860d191888
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/mambaforge3:23.1.0-4-python3.10.12-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

ENV PIPELINE_URL="https://github.com/AllenNeuralDynamics/aind-fiber-photometry-pipeline"
ENV PIPELINE_VERSION="7"
ENV VERSION="10"
ENV DFF_EXTRACTION_URL=https://github.com/AllenNeuralDynamics/aind-fip-dff

RUN pip3 install -U --no-cache-dir \
    aind-data-schema>=1.2.0 \
    aind-log-utils \
    aind-metadata-upgrader \
    aind-qcportal-schema \
    hdmf-zarr \
    matplotlib \
    pynwb \
    scikit-learn \
    statsmodels
