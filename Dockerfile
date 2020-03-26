FROM nvidia/cuda:10.2-cudnn7-devel


## FROM curlimages/curl:7.68.0 as curl

## WORKDIR /home/nlp

## RUN curl -L#O https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json && \
##     curl -L#O https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json

#FROM pytorch/pytorch:1.4-cuda10.1-cudnn7-devel

#WORKDIR /home/nlp

## COPY --from=curl https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json /home/nlp/train-v2.0.json
## COPY --from=curl https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json /home/nlp/dev-v2.0.json

## COPY https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json /home/nlp/train-v2.0.json
## COPY https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json /home/nlp/dev-v2.0.json

#RUN apt-get update && \
#    apt-get install -y ffmpeg git cmake

#EXPOSE 8123

##RUN free -h
##RUN conda install -y hypothesis jupyterlab

##ENTRYPOINT jupyter notebook --notebook-dir=/home/nlp --ip 0.0.0.0 --no-browser --allow-root --port=8123
