import gymnasium as gym
import cv2
import time

# 建立環境，使用 rgb_array 模式
env = gym.make("CarRacing-v3", render_mode="rgb_array")
obs, info = env.reset()

for _ in range(300):
    # 取得畫面 frame（rgb）
    frame = env.render()

    # OpenCV 顯示畫面（轉為 BGR）
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    cv2.imshow("CarRacing", frame_bgr)

    # 1ms 延遲 + 鍵盤中斷偵測
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # 隨機動作
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        break

env.close()
cv2.destroyAllWindows()
