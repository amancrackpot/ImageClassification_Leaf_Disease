FROM python:3.7-slim-stretch

RUN apt-get update && apt-get install -y git python3-dev gcc \
    && rm -rf /var/lib/apt/lists/*
RUN pip install torch_nightly -f https://download.pytorch.org/whl/nightly/cpu/torch_nightly.html

COPY requirements.txt .
RUN pip install --upgrade -r requirements.txt
#pip install --no-cache-dir -r

COPY src src/

# Run it once to trigger densenet download
RUN python src/app.py 

EXPOSE 8008

# Start the server
CMD ["python", "src/app.py", "serve"]


  







