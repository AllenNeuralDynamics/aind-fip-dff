# hash:sha256:c364b6a81399d9af2c2eb45a236a1cc3d76ededea3a3e6d411d57f320ffe64c3
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/mambaforge3:23.1.0-4-python3.10.12-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG GIT_ASKPASS
COPY git-askpass /

ENV VERSION=14.0
ENV DFF_EXTRACTION_URL=https://github.com/AllenNeuralDynamics/aind-fip-dff

RUN pip3 install -U --no-cache-dir \
    aind-data-schema>=2.4.0\
    aind-log-utils \
    aind-qcportal-schema \
    hdmf-zarr \
    matplotlib \
    pynwb \
    scikit-learn \
    statsmodels

COPY postInstall /
RUN /postInstall

