# Use an official Ubuntu runtime as a parent image
FROM ubuntu:22.04

# Set the working directory
WORKDIR /P-BIGE

# Install necessary dependencies
RUN apt-get update && apt-get install -y wget git htop xvfb

# Install Miniconda
RUN MINICONDA_INSTALLER_SCRIPT=Miniconda3-py38_23.1.0-1-Linux-x86_64.sh && \
    MINICONDA_PREFIX=/usr/local && \
    wget https://repo.continuum.io/miniconda/$MINICONDA_INSTALLER_SCRIPT && \
    chmod +x $MINICONDA_INSTALLER_SCRIPT && \
    ./$MINICONDA_INSTALLER_SCRIPT -b -f -p $MINICONDA_PREFIX && \
    rm $MINICONDA_INSTALLER_SCRIPT

# Update PATH to include conda
ENV PATH=/usr/local/bin:$PATH

# Copy the entire repo (including environment.yml) into the image
COPY . /P-BIGE

# Create the conda environment from environment.yml
RUN conda env create -f environment.yml

# Activate the conda environment for subsequent RUN commands
SHELL ["conda", "run", "-n", "P-BIGE", "/bin/bash", "-c"]

# Download the model and extractor
RUN bash dataset/prepare/download_model.sh && \
    bash dataset/prepare/download_extractor.sh

# Install additional Python packages (if needed)
RUN pip install --user ipykernel polyscope easydict trimesh
RUN pip install --user --force-reinstall numpy==1.22.0

# Set up Xvfb for Polyscope
ENV DISPLAY=:99.0
RUN Xvfb :99 -screen 0 1024x768x24 > /dev/null 2>&1 &

# Set the entrypoint to use the conda environment
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "P-BIGE", "python"]