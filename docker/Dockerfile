# Use a non-interactive frontend to suppress prompts during builds
ARG DEBIAN_FRONTEND=noninteractive

# Base image with CUDA and Python
FROM nvidia/cuda:12.1.0-base-ubuntu22.04

# Install apt dependencies
RUN apt-get update && \
    apt-get install -y \
    curl \
    git \
    parallel \
    wget \
    libtiff5 \
    gcc \
    g++ \
    openssl \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN curl -fsSL -v -o ~/miniforge.sh -O https://github.com/conda-forge/miniforge/releases/download/24.11.2-1/Miniforge3-24.11.2-1-Linux-x86_64.sh

RUN chmod +x ~/miniforge.sh && \
    bash ~/miniforge.sh -b -p /opt/conda && \
    rm ~/miniforge.sh && \
    /opt/conda/bin/conda install -y python=3.11 && \
    /opt/conda/bin/conda update --all -y -n base && \
    /opt/conda/bin/conda clean -ya

# Set the PATH for Conda environment
ENV PATH /opt/conda/bin:$PATH

RUN /bin/bash -c "source activate base"

# Install Conda dependencies
RUN conda install -y \
    ipython \
    jupyter \
    jupyterlab \
    opencv \
    && conda clean -qya

# Install Python packages with pip
RUN python -m pip install --no-cache-dir \
    fire \    
    zarr \
    mahotas \
    pillow \
    pandas \
    typer \
    inquirerpy \
    snakemake \
    snakemake-executor-plugin-slurm \
    numpy \
    wandb \
    accelerate \
    "imagecodecs>=2023.8.12" \
    rasterio \
    ipykernel \
    seaborn \
    matplotlib \
    umap-learn \
    hdbscan \
    transformers \
    diffusers \
    datasets \
    cleanlab \
    ruff \
    mypy \
    black \
    isort \
    pytest \
    anndata \
    scanpy \
    scikit-misc \
    tifffile \
    scikit-image \
    scikit-learn \
    dask \
    timm \
    einops \
    polars \
    dask-image \
    dask-ml \
    harmonypy \
    squidpy \
    annoy \
    spatialdata \
    sopa["baysor"]==2.0.0 \
    spatialdata_io \
    spatialdata_plot \
    sopa==2.0.0 \
    geopandas \
    shapely \
    readfcs \
    torch_geometric \
    scvi-tools \
    novae \
    pyg-lib -f https://data.pyg.org/whl/torch-2.3.0+cu121.html \
    'rapids-singlecell[rapids12]' --extra-index-url=https://pypi.nvidia.com \
    torch==2.3 \
    torchvision \
    torchaudio \
    --extra-index-url https://download.pytorch.org/whl/cu121


RUN apt-get update && \
    apt-get install unzip

# Install Baysor
RUN curl -O -L https://github.com/kharchenkolab/Baysor/releases/download/v0.7.1/baysor-x86_x64-linux-v0.7.1_build.zip && \
    unzip baysor-x86_x64-linux-v0.7.1_build.zip && \
    mv bin/baysor /opt/ && \
    rm baysor-x86_x64-linux-v0.7.1_build.zip

ENV PATH="${PATH}:/opt/baysor/bin"

# Set environment variables for NVIDIA compatibility
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV LD_LIBRARY_PATH \
$LD_LIBRARY_PATH:\
/opt/conda/lib/python3.11/site-packages/nvidia/cuda_nvrtc/lib:\
/opt/conda/lib/python3.11/site-packages/nvidia/cusolver/lib:\
/opt/conda/lib/python3.11/site-packages/nvidia/cublas/lib:\
/opt/conda/lib/python3.11/site-packages/nvidia/cusparse/lib:\
/opt/conda/lib/python3.11/site-packages/nvidia/curand/lib:\
/opt/conda/lib/python3.11/site-packages/nvidia/nvjitlink/lib

# Set the working directory
WORKDIR /workspace
