# start from python base image
FROM python:3.10-slim

# change working directory
WORKDIR /code

EXPOSE 80

# add requirements file to image
COPY ./requirements.txt /code/requirements.txt

RUN apt-get update && \
    apt-get install -y libgomp1 && \
    rm -rf /var/lib/apt/lists/*

# install python libraries
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Install NLTK
RUN pip install nltk

# Download the stopwords data during image build
RUN python -c "import nltk; nltk.download('stopwords')"

# add python code
COPY ./app/ /code/app/

# specify default commands
CMD ["fastapi", "run", "app/main.py", "--port", "80"]