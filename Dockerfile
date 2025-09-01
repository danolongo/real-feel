# my py version as the initial image
FROM python:3.11.13-slim

# working directory in container
WORKDIR /real-feel

# copy the current directory contents into the container at /real-feel
COPY . .

# install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# install poetry
RUN curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python -

# use poetry to install python dependencies
RUN /root/.poetry/bin/poetry config virtualenvs.create false \
    && /root/.poetry/bin/poetry install --no-interaction --no-ansi

# specify the command to run on container start
CMD ["python", "real-feel/train.py"]