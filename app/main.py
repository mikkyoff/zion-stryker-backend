
from fastapi import FastAPI
from routers import signals, auth, users

app = FastAPI()

app.include_router(signals.router, prefix="/signals")
app.include_router(auth.router, prefix="/auth")
app.include_router(users.router, prefix="/users")

@app.get("/")
def read_root():
    return {"message": "Zion Stryker backend is running"}
