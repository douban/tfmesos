FROM nvidia/cuda:7.5-cudnn5-runtime
RUN ln -s /usr/lib/x86_64-linux-gnu/libcudnn.so.5 /usr/lib/x86_64-linux-gnu/libcudnn.so
RUN apt-get update && apt-get install --no-install-recommends -y \
        g++ \
        libopenblas-dev \
        libzookeeper-mt-dev \
        python-dev \
        python-pip \
        git-core && \
    apt-get autoremove --purge -y && \
    apt-get clean && \
    rm -rf /var/cache/apt /var/lib/apt/lists
RUN pip install --download-cache="/tmp/pip" -r "https://raw.githubusercontent.com/douban/tfmesos/master/requirements.txt" && \
    pip install --download-cache="/tmp/pip" "git+https://github.com/douban/tfmesos.git@master#egg=tfmesos" && \
    rm -rf /tmp/*
