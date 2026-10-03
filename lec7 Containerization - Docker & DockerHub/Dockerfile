#python:3.10-slim debian variant of linux

# Using slim version to keep the image size small. python:3.10-slim includes OS libraries and dependencies required to run Python applications, but it does not include unnecessary packages that are not needed for most applications. This helps to reduce the attack surface and improve security.
FROM python:3.10-slim 


# Set the working directory in the container
WORKDIR /flask-loan-app

# Copy the requirements file into the container at /flask-loan-app
COPY artefacts/requirements.txt /flask-loan-app/requirements.txt

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container at /flask-loan-app
COPY . /flask-loan-app

# Make port 8000 available to the world outside this container
CMD ["python", "-m", "flask", "--app", "hello.py", "run", "--host=0.0.0.0", "--port=8000"]


#docker build -t oct3 .

#docker run -p 9000:8000 oct3 first 8000 is the port on the host machine and second 8000 is the port inside the container.

#docker image tag oct3:latest shivam13juna/oct3-live-demo:latest


#docker pull shivam13juna/oct3-live-demo

#docker run -p 9000:8000 oct3

