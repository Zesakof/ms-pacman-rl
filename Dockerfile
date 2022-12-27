FROM stablebaselines/rl-baselines3-zoo@sha256:46976041aaa4a5252eae1aebb4533ab2e2bbe449512ff290a37f8eddc1f1c56f

RUN mkdir /project
ADD . /project

RUN pip install -r /project/requirements.txt
