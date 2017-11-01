# av-feature-generator
Feature generator for Authorship Verification problems with pair-wise data

See utils/ds_models.py for how data is loaded.

It is recommended that you use [Docker](http://www.docker.com)

For gpu support, get [nvidia-docker](https://github.com/NVIDIA/nvidia-docker)

To build, go to the directory with the dockerfile and do (don't miss the dot in the end):

`docker build -f Dockerfile.av-feature-generator.txt -t av/tf-keras-ssh .`

Run, forwarding ssh to 2222. Jupyter notebook is on 8888:

`nvidia-docker run -ti -d -p 8888:8888 -p 2222:22 --name tf-ssh av/tf-keras-ssh`

Stop:

`docker stop tf-ssh`

Remove:

`docker rm tf-ssh`


[Instructions on how to debug dockerized code in host IDE](https://gist.github.com/dainis-boumber/fd580659db368d372afe4c1dd2af0521) 


More documentation can be found on [Project Wiki](https://github.com/dainis-boumber/av-feature-generator/wiki)
