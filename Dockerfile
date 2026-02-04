# Use official lightweight Python image
FROM python:3.10-slim

# Set working directory inside container
WORKDIR /app

# Copy requirements first (for caching)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy entire project into container
COPY . .

# Expose port (FastAPI runs on 8000)
EXPOSE 8000

# Run FastAPI app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]


# in the working directory run the following command to build the docker image
# docker build -t fraud-api . // . menas everthing in the current directory, when run Docker did - this dockerfile, download base python image, copy your files, install libraries, pack everything in image, name it fuaud-api

# and to run the container
# docker run -d -p 8000:8000 fraud-api  // when run, create a container from image, started linux inside it, started python, ran uvicorn, load your ml model, started fastapi server
# 8000:8000 means map port 8000 of your local machine to port 8000 of the container
# -d means run in detached mode (in background)
# Now you can access the API at http://localhost:8000

# when using POST/predict, json goes in fastapi validate schema, numpy array is built, scaler transform, model predicts, json response returned

