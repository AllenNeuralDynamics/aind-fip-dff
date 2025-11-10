# hash:sha256:c364b6a81399d9af2c2eb45a236a1cc3d76ededea3a3e6d411d57f320ffe64c3
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/mambaforge3:23.1.0-4-python3.10.12-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG GIT_ASKPASS
COPY git-askpass /

ENV VERSION=14.0
ENV DFF_EXTRACTION_URL=https://github.com/AllenNeuralDynamics/aind-fip-dff

RUN pip3 install -U --no-cache-dir \
    aind-data-schema==1.4.0 \
    aind-log-utils==0.2.4 \
    aind-metadata-upgrader==0.0.26 \
    aind-qcportal-schema==0.4.0 \
    hdmf-zarr==0.11.3 \
    matplotlib==3.10.6 \
    pynwb==3.1.2 \
    scikit-learn==1.7.2 \
    statsmodels==0.14.5
