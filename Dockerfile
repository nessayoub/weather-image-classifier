FROM python:bullseye




WORKDIR /app

COPY . /app/
RUN apt-get update && apt-get upgrade -y
RUN pip install --upgrade setuptools
RUN pip install --upgrade pip

RUN pip3 install --no-cache-dir -r requirements.txt

EXPOSE 5000
# Run the specified command
CMD ["python3","app.py"]
