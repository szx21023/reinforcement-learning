import gymnasium as gym
import cv2
import time
import torch

from main import ACTIONS, DQNAgent, preprocess

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
agent = DQNAgent(num_actions=len(ACTIONS), device=device, pretrain_model_path='./car_dqn_episode120.pth')
agent.epsilon = 0.05  # 設定 epsilon 為 0.05，這樣 agent 就不會隨機選擇行動了

# 建立環境，使用 rgb_array 模式
env = gym.make("CarRacing-v3", render_mode="rgb_array")
obs, info = env.reset()

for _ in range(30000):
    # 取得畫面 frame（rgb）
    frame = env.render()

    # OpenCV 顯示畫面（轉為 BGR）
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    cv2.imshow("CarRacing", frame_bgr)

    # 1ms 延遲 + 鍵盤中斷偵測
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    state = preprocess(obs)
    action_idx = agent.act(state)
    action = ACTIONS[action_idx]
    obs, reward, terminated, truncated, info = env.step(action)

    if terminated:# or truncated:
        break

env.close()
cv2.destroyAllWindows()
