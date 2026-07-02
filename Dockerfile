FROM python:3.11-slim

RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade -r requirements.txt

COPY --chown=user . .

# CRITICAL: Must listen on port 7860 (HF Spaces requirement)
CMD ["python", "run.py", "--host", "0.0.0.0", "--port", "7860"]