from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from scipy.integrate import solve_ivp
from stable_baselines3 import TD3
from environments import MassSpringDamper, tEnd, tStart, MultipleDataEnv, MultipleDataY1Env, x0, t, NormalizeData, dataLength
from sklearn.metrics import r2_score

class SystemRequest(BaseModel):
    name: str
    modelParams: list[float] = []


class SystemResponse(BaseModel):
    expected: list[list[float]] = []
    predicted: list[list[float]] = []
    r2Score: float

origins = [
    "http://localhost",
    "http://localhost:5173",
]

app = FastAPI(title='szakdoga')

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
)

@app.get('/')
async def home() -> SystemRequest:
    return SystemRequest(name='asd')

@app.post('/predict/')
async def predict(request: SystemRequest) -> SystemResponse:
    if(request.name == "Mass Spring Damper"):
        expected, predicted = MSDPredict(request.modelParams)
    else:
        expected, predicted = []
    r2 = (r2_score(expected[0][:dataLength-1], predicted[0]) + r2_score(expected[1][:dataLength-1], predicted[1]))/2  
    return SystemResponse(expected=expected, predicted=predicted, r2Score=r2)


def MSDPredict(params):
    predicted0 = []
    predicted1 = []

    y0td3 = TD3.load('./Models/MassSpringDamperY0TD3.zip')
    y1td3 = TD3.load('./Models/MassSpringDamperY1TD3.zip')

    y0env = MultipleDataEnv(params[0],params[1],params[2])
    y1env = MultipleDataY1Env(params[0],params[1],params[2])

    obs = y0env.reset()[0]
    done = False
    while done is False:
        action, _states = y0td3.predict(obs)
        obs, reward, truncated, done, info = y0env.step(action)
        predicted0.append(action[0])

    obs = y1env.reset()[0]
    done = False
    while done is False:
        action, _states = y1td3.predict(obs)
        obs, reward, truncated, done, info = y1env.step(action)
        predicted1.append(action[0])

    expected = solve_ivp(MassSpringDamper, [tStart, tEnd], x0, t_eval=t, args=params).y
    return (expected, [predicted0, predicted1])
