# Installation

- Clone this repo:

```bash
git clone https://github.com/emadRad/VideoModelBenchmark.git
cd VideoModelBenchmark
```

## Docker 
### Installing Docker
To use the docker for running the code you need the following packages to be installed on ubuntu:
- Docker

You can follow the instructions on the [Docker](https://docs.docker.com/engine/install/ubuntu/) to install the packages.

After installation don't forget to do the post-installation steps by following the instructions 
at [Linux-Post-install](https://docs.docker.com/engine/install/linux-postinstall/).

Then to use gpu capabilities you need to install the following package:
- nvidia-docker

Follow the instructions on the [nvidia-docker](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) to install the package.

### Setting up Docker Image

Run the following commands to set up the docker image:

```bash
cd docker/
docker build -t video_benchmark .
```

<!--
## Conda Environment
### Create Environment
If environment is not created, you can create it by running:
```bash
conda env create --name vml --file docker/environment.yml
```

If environment is created, you can update it by running:
```bash
conda env update --name vml --file docker/environment.yml
```

-->
