FROM nvcr.io/nvidia/pytorch:25.02-py3

MAINTAINER Jaekyung Cho, GenAIIC-APJ-CMO <jackcho@amazon.com>

# SageMaker setting
ARG NB_USER="sagemaker-user"

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/ml/code:${PATH}" \
    SAGEMAKER_SUBMIT_DIRECTORY=/opt/ml/code

# SageMaker specific environment
ENV TOKENIZERS_PARALLELISM=false

# Install system dependencies
USER root
RUN apt-get update && \
    apt-get install -y \
    wget \
    gcc \
    linux-headers-generic \
    software-properties-common \
    curl \
    git && \
    rm -rf /var/lib/apt/lists/* cuda-keyring_1.1-1_all.deb

# CUDA setting
ENV PATH="${CUDA_HOME}/bin:${PATH}" \
    LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"

# Install dependencies
WORKDIR /opt/ml/code
COPY requirements.txt .
RUN pip3 install ninja cmake && \
    pip3 install torch==2.7.1+cu128 --index-url https://download.pytorch.org/whl/cu128 && \
    pip3 install numpy==1.26.4 psutil==7.1.1 && \
    pip3 install flash-attn==2.8.1 --no-build-isolation && \
    pip3 install --no-build-isolation "transformer-engine[pytorch]==2.11.0"
RUN pip3 install -r requirements.txt


# Install Kernel for SageMaker Studio
RUN apt-get remove -y python3-blinker python3-cryptography || true && \
    pip3 install sagemaker sagemaker-training boto3

# Install ms-swift in editable mode
COPY ms-swift/ ./ms-swift/
RUN pip3 install -e ./ms-swift

# Copy project files
COPY model_patch/ ./model_patch/
COPY scripts/ ./scripts/
COPY utils.py ./utils.py
COPY sagemaker_entrypoint.py ./sagemaker_entrypoint.py
COPY __init__.py ./

# Execution permission for scripts
RUN chmod +x /opt/ml/code/scripts/*.sh && \
    chmod +x /opt/ml/code/scripts/expert/*.sh

# Create SageMaker required directories
RUN mkdir -p /opt/ml/input/data/training \
             /opt/ml/input/config \
             /opt/ml/output/data \
             /opt/ml/model \
             outputs/expert_scores \
             outputs/logs \
             results/expert_configs \
             results/expert_scores

RUN mkdir -p /home/${NB_USER} && \
    ln -s /opt/ml/code /home/${NB_USER}/ESFT-ms-swift
ENV PATH="/opt/ml/code:${PATH}"
ENV PYTHONPATH="/opt/ml/code:${PYTHONPATH}"

ENV HOME=/home/${NB_USER}
RUN chmod -R 777 /opt/ml /tmp
# USER ${NB_UID}:${NB_GID}

# Move working SageMaker directory (EFS will be mounted)
WORKDIR /home/${NB_USER}

# Set entrypoint for SageMaker training
# ENTRYPOINT ["python", "/opt/ml/code/sagemaker_entrypoint.py"]