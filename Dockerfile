# Set the Docker OS version
#TODO Add FROM to set base image

# Set working directory
WORKDIR /election-predictor

# Set the image content
COPY requirements_prod.txt requirements.txt
RUN pip install -r requirements.txt

COPY election-predictor election-predictor
COPY setup.py setup.py
RUN pip install .

COPY Makefile Makefile

# Hanndle file reset
RUN make reset_local_files

# Listen to all network connections inside the container for api.fast:app
CMD uvicorn taxifare.api.fast:app --host 0.0.0.0 --port 8080
