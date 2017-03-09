FROM nvidia/cuda:8.0-cudnn5-runtime
RUN ln -sf /usr/local/cuda/lib64/libcudnn.so.5 /usr/local/cuda/lib64/libcudnn.so && \
    ldconfig
RUN apt-get update && apt-get install --no-install-recommends -y \
        gcc \
        libopenblas-base \
        libzookeeper-mt-dev \
        ca-certificates \
        python-dev \
        git-core && \
    apt-get autoremove --purge -y && \
    apt-get clean && \
    rm -rf /var/cache/apt /var/lib/apt/lists
RUN python -c 'import urllib2;exec(urllib2.urlopen("https://bootstrap.pypa.io/get-pip.py").read())' --no-cache-dir --timeout 1000 && \
    pip install --no-cache-dir --timeout 1000 -r "https://raw.githubusercontent.com/douban/tfmesos/master/requirements.txt" && \
    pip install --no-cache-dir --timeout 1000 "git+https://github.com/douban/tfmesos.git@master#egg=tfmesos"
ENV DOCKER_IMAGE tfmesos/tfmesos
