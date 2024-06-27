# hash:sha256:6e4cb66fab482c3c883f7cdbe832ec9f67e1945b336e2c35601bb392d8d367ca
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/cuda-miniconda-jupyterlab:latest

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN pip3 install -U --no-cache-dir \
    aind-codeocean-api==0.4.2 \
    boto3==1.34.96 \
    hdmf-zarr==0.8.0 \
    pandas==2.0.3 \
    pathlib2==2.3.7.post1 \
    pynwb==2.6.0 \
    scikit-learn==1.4.0 \
    seaborn==0.13.2
