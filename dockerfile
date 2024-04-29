FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime

RUN ['apt',"update"]
RUN ['apt',"install","-y","vim"]
RUN ['pip',"install","-r","requirements.txt"]
RUN ['python3',"train.py"]