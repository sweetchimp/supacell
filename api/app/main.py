from fastapi import FastAPI

app = FastAPI(title="supacell API")


@app.get("/health")
def health():
    return {"status": "ok"}
