from environments import MassSpringDamper, NormalizeData, tEnd, tStart, MultipleDataEnv, MultipleDataY1Env, x0, t
from scipy.integrate import solve_ivp
from stable_baselines3 import TD3
from invertedPendulum import InvertedPendulum, InvertedPendulumEnvY0, InvertedPendulumEnvY1

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

def InvertedPendulumPredict(params):
    predicted0 = []
    predicted1 = []

    y0td3 = TD3.load('./Models/InvertedPendulum2.zip')

    y1td3 = TD3.load('./Models/InvertedPendulum3.zip')

    y0env = InvertedPendulumEnvY0(9.8,params[0],params[1])
    y1env = InvertedPendulumEnvY1(9.8,params[0],params[1])
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

    expected = solve_ivp(InvertedPendulum, [tStart, tEnd], x0, t_eval=t, args=([9.8,params[0],params[1]])).y
    
    return ([NormalizeData(expected[0]),NormalizeData(expected[1])], [predicted0, predicted1])
