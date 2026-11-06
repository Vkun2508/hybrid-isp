# One image: builds the React site, then serves it and the API with FastAPI on port 7860.
#   docker build -t hybrid-isp . && docker run -p 7860:7860 hybrid-isp
# Works as-is on Hugging Face Spaces (Docker SDK, non-root uid 1000, port 7860).

FROM node:20-slim AS web
WORKDIR /web
COPY web/package.json web/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY web/ ./
RUN npm run build

FROM python:3.13-slim
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    ISP_MAX_MP=2.0
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/* \
 && useradd -m -u 1000 user
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY --chown=user isp/ isp/
COPY --chown=user server/ server/
COPY --chown=user models/ models/
COPY --chown=user samples/ samples/
COPY --chown=user results/ results/
COPY --from=web --chown=user /web/dist web/dist
USER user
EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/api/health')"
CMD ["uvicorn", "server.main:app", "--host", "0.0.0.0", "--port", "7860", "--proxy-headers"]
