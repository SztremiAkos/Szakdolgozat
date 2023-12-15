from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from environments import dataLength
from sklearn.metrics import r2_score
from predictions import InvertedPendulumPredict, MSDPredict

class SystemRequest(BaseModel):
    name: str
    modelParams: list[float] = []


class SystemResponse(BaseModel):
    expected: list[list[float]] = []
    predicted: list[list[float]] = []
    r2Score: float

origins = ["*"]

app = FastAPI(title='szakdoga')

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get('/')
async def home() -> SystemRequest:
    return SystemRequest(name='Api works fine')

@app.post('/predict/')
async def predict(request: SystemRequest) -> SystemResponse:
    if(request.name == "Mass Spring Damper"):
        expected, predicted = MSDPredict(request.modelParams)
        r2 = (r2_score(expected[0][:dataLength-1], predicted[0]) + r2_score(expected[1][:dataLength-1], predicted[1]))/2
    elif(request.name == "Inverted Pendulum"):
        expected, predicted = InvertedPendulumPredict(request.modelParams)
        r2 = (r2_score(expected[0][:dataLength-1], predicted[0]) + r2_score(expected[1][:dataLength-1], predicted[1]))/2
    else:
        expected, predicted = []
        r2 = 0
    return SystemResponse(expected=expected, predicted=predicted, r2Score=r2)

