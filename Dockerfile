# based upon this example
# https://blog.ceshine.net/post/replicate-conda-environment-in-docker/
FROM nvidia/cuda:11.0-cudnn8-runtime-ubuntu18.04

LABEL maintainer="Ben Ely"

ARG CONDA_PYTHON_VERSION=3
ARG CONDA_DIR=/opt/conda
ARG USERNAME=docker
ARG USERID=1000
ARG PROJECT_DIR=ProductRecommendation

# Install basic utilities
RUN apt-get update && \
    apt-get install -y --no-install-recommends git wget unzip bzip2 sudo build-essential ca-certificates && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install miniconda
ENV PATH $CONDA_DIR/bin:$PATH
RUN wget --quiet https://repo.continuum.io/miniconda/Miniconda$CONDA_PYTHON_VERSION-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    echo 'export PATH=$CONDA_DIR/bin:$PATH' > /etc/profile.d/conda.sh && \
    /bin/bash /tmp/miniconda.sh -b -p $CONDA_DIR && \
    rm -rf /tmp/*

# Create the user
RUN useradd --create-home -s /bin/bash --no-user-group -u $USERID $USERNAME && \
    chown $USERNAME $CONDA_DIR -R && \
    adduser $USERNAME sudo && \
    echo "$USERNAME ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers

USER $USERNAME
WORKDIR /home/$USERNAME

# copy the YAML file into the Docker image
ADD ./environment.yml .
# update the base/root Conda environment
RUN conda env update --name base --file ./environment.yml

# add my files to the docker container
ADD ./$PROJECT_DIR /home/$USERNAME/$PROJECT_DIR

## add my src code to the conda environment
# move to the project directory
WORKDIR /home/$USERNAME/$PROJECT_DIR
# run setup.py (not in editable mode)
RUN pip install .
# return to the work directory
WORKDIR /home/$USERNAME

# Exposing ports
EXPOSE 8888

# Running jupyter notebook
# 'demo' is the password
CMD ["jupyter", "notebook", "--no-browser", "--ip=0.0.0.0", "--allow-root", "--NotebookApp.token='demo'"]