# hash:sha256:1059524086d5e3b2d9e2216e266ae536ec885dc14bcb4a468eae3437ce8aba40
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/mambaforge3:23.1.0-4-python3.10.12-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG GIT_ASKPASS
COPY git-ask-pass /

ENV VERSION=14.0
ENV DFF_EXTRACTION_URL=https://github.com/AllenNeuralDynamics/aind-fip-dff

RUN pip3 install -U --no-cache-dir \
    aind-data-schema==1.4.0 \
    aind-metadata-upgrader==0.0.26 \
    aind-qcportal-schema==0.4.0 \
    hdmf-zarr==0.11.3 \
    matplotlib==3.10.6 \
    pynwb==3.1.2 \
    scikit-learn==1.7.2 \
    statsmodels==0.14.5
