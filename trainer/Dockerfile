FROM rockylinux:9

RUN dnf -y update && dnf -y groupinstall "Development Tools" && \
    dnf -y install git wget python3.11 python3.11-pip python3.11-devel \
                   cmake make gcc-c++ openssl-devel && \
    # この「1」は alternatives グループ内での優先順位（優先度＝priority）です
    # 数字が大きいほど優先して選択されます
    alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 && \
    alternatives --set python3 /usr/bin/python3.11 && \
    python3 -m pip install --upgrade pip

COPY requirements.txt /tmp/requirements.txt
RUN python3 -m pip install -r /tmp/requirements.txt

# llama.cpp ビルド (量子化/変換に使用)
# RUN git clone --depth=1 https://github.com/ggerganov/llama.cpp /opt/llama.cpp && \
#     cd /opt/llama.cpp && \
#     make -j && \
#     ln -s /opt/llama.cpp/build/bin/llama-quantize /usr/local/bin/llama-quantize

# 作業ディレクトリ
WORKDIR /trainer
