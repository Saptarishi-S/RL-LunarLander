# RL-LunarLander

This project trains a Deep Q-Network (DQN) agent to successfully land a spacecraft in OpenAI Gym's [LunarLander-v2](https://www.gymlibrary.dev/environments/box2d/lunar_lander/) environment.  
The model aims to solve the environment by reaching an average score of **250** across the last 100 episodes.

---

## 📂 Project Structure
- `LunarLander_250_w.ipynb` — Main notebook with:
  - Environment setup
  - DQN model (`QNet` and `DQN` classes)
  - Replay Buffer
  - Training loop (`train`)
  - Testing loop (`testLander`)
  - Score plotting (`plotScore`)

---

## 🛠️ Setup Instructions

1. **Install required packages:**
    ```bash
    pip install gym pygame box2d-py numpy==1.23.0 torch matplotlib tqdm
    ```

2. **Ensure GPU is available** (optional but recommended for faster training):
    - Check with:
      ```python
      import torch
      print(torch.cuda.is_available())
      ```

3. **Run the Notebook:**
    Open `LunarLander_250_w.ipynb` in Jupyter Notebook, JupyterLab, or Google Colab and execute the cells sequentially.

---

## ⚙️ Training Configuration

| Parameter            | Value  |
|----------------------|--------|
| Batch Size           | 128    |
| Learning Rate        | 3e-4   |
| Episodes             | 1500   |
| Gamma (Discount)     | 0.98   |
| Memory Size          | 50,000 |
| Learn Step Interval  | 2      |
| Tau (Soft Update)    | 0.01   |
| Target Score         | 250    |

- **Epsilon-Greedy Policy**: 
  - Starts with epsilon = 1.0 (full exploration)
  - Decays to epsilon = 0.1 over time

- **Early Stopping**: 
  - Stops training if average score over last 100 episodes exceeds 250.

---

## 📈 Results
- Training progress is displayed via a live progress bar (`tqdm`) showing:
  - Current episode score
  - Average score over last 100 episodes
- A final plot shows the full training score history.

---

## 💾 Checkpoints
- If `SAVE_CHKPT = True`, the trained model weights are saved to `checkpoint.pth` automatically when training completes.

---

## 🧠 Key Components
- **QNet**: A simple feedforward neural network with two hidden layers.
- **DQN Agent**:
  - Epsilon-greedy action selection
  - Experience replay
  - Soft target updates
- **ReplayBuffer**: Random experience sampling for stable training.

---

## 🖥️ Testing the Agent
Use the `testLander` function to watch the trained agent land the Lunar Lander successfully.

```python
testLander(env, agent, loop=3)
```
*(Runs 3 episodes with rendering)*

---

## 📜 Notes
- Works with OpenAI Gym's classic control environments.
- Designed for fast convergence on GPUs.
- Uses PyTorch for model definition and optimization.
