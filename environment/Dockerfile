# hash:sha256:17a391399f2ce0319072ea53256c5eacf8a17def6e54c270951c50d846b2357a
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/mambaforge3:23.1.0-4-python3.10.12-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG GIT_ASKPASS
COPY git-ask-pass /

ENV VERSION=16.0
ENV DFF_EXTRACTION_URL=https://github.com/AllenNeuralDynamics/aind-fip-dff
ENV PROCESS_NAME=aind-fip-dff

RUN pip3 install -U --no-cache-dir \
    aind-data-schema==1.4.0 \
    aind-metadata-upgrader==0.0.26 \
    aind-qcportal-schema==0.4.0 \
    hdmf-zarr==0.11.3 \
    matplotlib==3.10.6 \
    pynwb==3.1.2 \
    pyyaml==6.0.3 \
    scikit-learn==1.7.2 \
    statsmodels==0.14.5 \
    watchtower==3.4.0

ADD "https://github.com/coder/code-server/releases/download/v4.100.3/code-server-4.100.3-linux-amd64.tar.gz" /.code-server/code-server.tar.gz

RUN cd /.code-server \
    && tar -xvf code-server.tar.gz \
    && rm code-server.tar.gz \
    && ln -s /.code-server/code-server-4.100.3-linux-amd64/bin/code-server /usr/bin/code-server \
    && mkdir -p /.vscode/extensions

RUN code-server --extensions-dir="/.vscode/extensions" --install-extension ms-python.python@2025.14.0 \
    && code-server --extensions-dir="/.vscode/extensions" --install-extension ms-toolsai.jupyter@2025.4.1 \
    && code-server --extensions-dir="/.vscode/extensions" --install-extension reditorsupport.r@2.8.6 \
    && code-server --extensions-dir="/.vscode/extensions" --install-extension saoudrizwan.claude-dev@3.35.1
