# hash:sha256:760f72a519a14c7d96e2f548ef89dcf4b4dcb6501365fe47c3dcaa122a38e937
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/cuda-miniconda-jupyterlab:latest

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN pip3 install -U --no-cache-dir \
    aind-codeocean-api==0.4.2 \
    boto3==1.34.96 \
    pandas==2.0.3 \
    pathlib2==2.3.7.post1 \
    pynwb==2.6.0 \
    scikit-learn==1.4.0 \
    seaborn==0.13.2

ADD "https://github.com/coder/code-server/releases/download/v4.9.0/code-server-4.9.0-linux-amd64.tar.gz" /.code-server/code-server.tar.gz
	
RUN cd /.code-server \
	&& tar -xvf code-server.tar.gz \
	&& rm code-server.tar.gz \
	&& ln -s /.code-server/code-server-4.9.0-linux-amd64/bin/code-server  /usr/bin/code-server
