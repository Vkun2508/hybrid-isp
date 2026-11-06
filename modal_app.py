"""Host the Hybrid ISP web app on Modal (the public live URL).

    cd web && npm run build && cd ..          # the site is served from web/dist
    modal deploy modal_app.py                 # prints https://<workspace>--hybrid-isp-web.modal.run

One always-warm CPU container runs the same FastAPI app as the Dockerfile (server/main.py: API + site).
A single container keeps the in-memory result cache coherent; requests within it run concurrently.
Stop it after the hackathon with `modal app stop hybrid-isp`.
"""
import modal

IGNORE = ["**/__pycache__/**", "**/*.pyc", "**/_eval_*"]

image = (
    modal.Image.debian_slim(python_version="3.13")
    .apt_install("libgomp1")
    .pip_install_from_requirements("requirements.txt")
    .env({"ISP_MAX_MP": "2.0", "PYTHONUNBUFFERED": "1"})
    .add_local_dir("isp", "/app/isp", ignore=IGNORE)
    .add_local_dir("server", "/app/server", ignore=IGNORE)
    .add_local_dir("models", "/app/models", ignore=IGNORE)
    .add_local_dir("samples", "/app/samples")
    .add_local_dir("results", "/app/results")
    .add_local_dir("web/dist", "/app/web/dist")
)

app = modal.App("hybrid-isp", image=image)


@app.function(cpu=4.0, memory=4096, min_containers=1, max_containers=1, timeout=120, scaledown_window=1200)
@modal.concurrent(max_inputs=16)
@modal.asgi_app()
def web():
    import os
    import sys

    sys.path.insert(0, "/app")
    os.chdir("/app")
    from server.main import app as fastapi_app

    return fastapi_app
