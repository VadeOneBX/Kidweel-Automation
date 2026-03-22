from fastapi import FastAPI
import subprocess

app = FastAPI()

@app.get("/")
def home():
    return {"status": "qops paper trader running"}

@app.get("/run")
def run_strategy():
    subprocess.Popen(["qops-select-candidates"])
    return {"msg": "candidate selector started"}

@app.get("/status")
def status():
    return {"ok": True}