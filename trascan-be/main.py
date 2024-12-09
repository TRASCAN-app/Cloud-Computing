from fastapi import FastAPI
from routes import article, predict

app = FastAPI()

# Include routers
app.include_router(article.router)
app.include_router(predict.router)

@app.get("/")
def home():
    return {
        "message": "Welcome to the Trascan!",
        "endpoints": {
            "/predict": "POST an image file to classify waste.",
            "/articles": "GET to fetch waste-related articles."
        }
    }

@app.get("/favicon.ico")
def favicon():
    return "", 204

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
