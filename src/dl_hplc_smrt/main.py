from fastapi import FastAPI

app = FastAPI(title="HPLC RT Prediction API", version="0.1.0")


@app.get("/")
def read_root() -> dict[str, str]:
    return {"message": "HPLC RT prediction API"}


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("dl_hplc_smrt.main:app", host="0.0.0.0", port=8000, reload=True)
