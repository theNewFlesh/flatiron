FROM tensorflow/tensorflow:nightly-gpu AS base

USER root

# coloring syntax for headers
ENV CYAN='\033[0;36m'
ENV CLEAR='\033[0m'
ENV DEBIAN_FRONTEND='noninteractive'

# setup ubuntu user
ARG UID_='1000'
ARG GID_='1000'
RUN echo "\n${CYAN}SETUP UBUNTU USER${CLEAR}"; \
    addgroup --gid $GID_ ubuntu && \
    adduser \
        --disabled-password \
        --gecos '' \
        --uid $UID_ \
        --gid $GID_ ubuntu && \
    usermod -aG root ubuntu

# setup sudo
RUN echo "\n${CYAN}SETUP SUDO${CLEAR}"; \
    apt update && \
    apt install -y sudo && \
    usermod -aG sudo ubuntu && \
    echo '%ubuntu    ALL = (ALL) NOPASSWD: ALL' >> /etc/sudoers && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /home/ubuntu

# update ubuntu and install basic dependencies
RUN echo "\n${CYAN}INSTALL GENERIC DEPENDENCIES${CLEAR}"; \
    apt update && \
    apt install -y \
        bat \
        curl \
        git \
        graphviz \
        npm \
        parallel \
        software-properties-common \
        unzip \
        vim \
        wget && \
    rm -rf /var/lib/apt/lists/* && \
    curl -fsSL \
        "https://github.com/ogham/exa/releases/latest/download/exa-linux-x86_64-v0.10.1.zip" \
        -o exa.zip && \
    unzip -q exa.zip bin/exa -d /usr/local && \
    rm -rf exa.zip && \
    curl -fsSL \
        "https://github.com/BurntSushi/ripgrep/releases/latest/download/ripgrep_13.0.0_amd64.deb" \
        -o ripgrep.deb && \
    apt install -y ./ripgrep.deb && \
    rm -rf ripgrep.deb

# install nvidia drivers
# RUN echo "\n${CYAN}INSTALL NVIDIA DRIVERS${CLEAR}"; \
#     apt update && \
#     apt install -y \
#         nvidia-utils-525 \
#         nvidia-driver-525 && \
#     rm -rf /var/lib/apt/lists/*

# install all python versions
RUN echo "\n${CYAN}INSTALL PYTHON${CLEAR}"; \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt update && \
    apt install -y \
        python3-pydot \
        python3.10-dev \
        python3.10-venv \
        python3.10-distutils \
        python3.9-dev \
        python3.9-venv \
        python3.9-distutils \
        python3.8-dev \
        python3.8-venv \
        python3.8-distutils && \
    rm -rf /var/lib/apt/lists/*

# install pip
RUN echo "\n${CYAN}INSTALL PIP${CLEAR}"; \
    wget https://bootstrap.pypa.io/get-pip.py && \
    python3.10 get-pip.py && \
    pip3.10 install --upgrade pip && \
    rm -rf get-pip.py

# install and setup zsh
RUN echo "\n${CYAN}SETUP ZSH${CLEAR}"; \
    apt update && \
    apt install -y zsh && \
    rm -rf /var/lib/apt/lists/* && \
    curl -fsSL https://raw.github.com/ohmyzsh/ohmyzsh/master/tools/install.sh \
        -o install-oh-my-zsh.sh && \
    echo y | sh install-oh-my-zsh.sh && \
    mkdir -p /root/.oh-my-zsh/custom/plugins && \
    cd /root/.oh-my-zsh/custom/plugins && \
    git clone https://github.com/zdharma-continuum/fast-syntax-highlighting && \
    git clone https://github.com/zsh-users/zsh-autosuggestions && \
    npm i -g zsh-history-enquirer --unsafe-perm && \
    cd /home/ubuntu && \
    cp -r /root/.oh-my-zsh /home/ubuntu/ && \
    chown -R ubuntu:ubuntu .oh-my-zsh && \
    rm -rf install-oh-my-zsh.sh && \
    echo 'UTC' > /etc/timezone

USER ubuntu
ENV PATH="/home/ubuntu/.local/bin:$PATH"
COPY ./config/henanigans.zsh-theme .oh-my-zsh/custom/themes/henanigans.zsh-theme

ENV LANG "C.UTF-8"
ENV LANGUAGE "C.UTF-8"
ENV LC_ALL "C.UTF-8"
# ------------------------------------------------------------------------------

FROM base AS dev
USER root

# install OpenEXR
ENV CC=gcc
ENV CXX=g++
# ENV LD_LIBRARY_PATH='/usr/include/python3.10m/dist-packages'
RUN echo "\n${CYAN}INSTALL OPENEXR${CLEAR}"; \
    apt update && \
    apt install -y \
        build-essential \
        g++ \
        gcc \
        libopenexr-dev \
        openexr \
        zlib1g-dev && \
    rm -rf /var/lib/apt/lists/*

USER ubuntu
WORKDIR /home/ubuntu

# insetll dev dependencies
RUN echo "\n${CYAN}INSTALL DEV DEPENDENCIES${CLEAR}"; \
    curl -sSL \
        https://raw.githubusercontent.com/pdm-project/pdm/main/install-pdm.py \
        | python3.10 - && \
    pip3.10 install --upgrade --user \
        pdm \
        pdm-bump \
        'rolling-pin>=0.9.2' && \
    mkdir -p /home/ubuntu/.oh-my-zsh/custom/completions && \
    pdm self update && \
    pdm completion zsh > /home/ubuntu/.oh-my-zsh/custom/completions/_pdm

# setup pdm
COPY --chown=ubuntu:ubuntu config/* /home/ubuntu/config/
COPY --chown=ubuntu:ubuntu scripts/* /home/ubuntu/scripts/
RUN echo "\n${CYAN}SETUP DIRECTORIES${CLEAR}"; \
    mkdir pdm

# create dev env
WORKDIR /home/ubuntu/pdm
RUN echo "\n${CYAN}INSTALL DEV ENVIRONMENT${CLEAR}"; \
    . /home/ubuntu/scripts/x_tools.sh && \
    export CONFIG_DIR=/home/ubuntu/config && \
    export SCRIPT_DIR=/home/ubuntu/scripts && \
    x_env_init dev 3.10 && \
    cd /home/ubuntu && \
    ln -s `_x_env_get_path dev 3.10` .dev-env && \
    ln -s `_x_env_get_path dev 3.10`/lib/python3.10/site-packages .dev-packages

# create prod envs
RUN echo "\n${CYAN}INSTALL PROD ENVIRONMENTS${CLEAR}"; \
    . /home/ubuntu/scripts/x_tools.sh && \
    export CONFIG_DIR=/home/ubuntu/config && \
    export SCRIPT_DIR=/home/ubuntu/scripts && \
    x_env_init prod 3.10 && \
    x_env_init prod 3.9 && \
    x_env_init prod 3.8

# cleanup dirs
WORKDIR /home/ubuntu
RUN echo "\n${CYAN}REMOVE DIRECTORIES${CLEAR}"; \
    rm -rf config scripts

ENV REPO='flatiron'
ENV PYTHONPATH ":/home/ubuntu/$REPO/python:/home/ubuntu/.local/lib"
ENV PYTHONPYCACHEPREFIX "/home/ubuntu/.python_cache"
