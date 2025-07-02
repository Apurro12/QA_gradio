FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update

# Install uv package manager
RUN pip install uv
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen

COPY src/ ./src/
COPY config/ ./config/
COPY main.py ./
EXPOSE 7860
ENV CONFIG_NAME=default_docker
# To add different configs see add --config-name or --offline
CMD ["sh", "-c", "uv run main.py --config-name ${CONFIG_NAME}"]

#docker build -t gradio-app .  
#docker run -p 7860:7860 --env-file .env gradio-app