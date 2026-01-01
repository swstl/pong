import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim

# Simple MLP Policy
class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return torch.softmax(self.fc2(x), dim=-1)

# Initialize
env = gym.make('CartPole-v1', render_mode='human')
policy = Policy()
optimizer = optim.Adam(policy.parameters(), lr=0.01)

def train_episode():
    state, _ = env.reset()
    log_probs = []
    rewards = []

    for _ in range(500):  # Max 500 steps
        # Get action
        state_tensor = torch.FloatTensor(state)
        probs = policy(state_tensor)
        action = torch.multinomial(probs, 1).item()
        log_probs.append(torch.log(probs[action]))

        # Take step
        state, reward, terminated, truncated, _ = env.step(action)
        rewards.append(reward)

        if terminated or truncated:
            break

    # Calculate returns (discounted rewards)
    returns = []
    G = 0
    for r in reversed(rewards):
        G = r + 0.99 * G
        returns.insert(0, G)

    # Normalize returns
    returns = torch.FloatTensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)

    # Policy gradient update
    loss = []
    for log_prob, G in zip(log_probs, returns):
        loss.append(-log_prob * G)

    optimizer.zero_grad()
    torch.stack(loss).sum().backward()
    optimizer.step()

    return len(rewards)

# Train!
print("Training CartPole with REINFORCE...")
for episode in range(50):
    score = train_episode()
    if episode % 20 == 0:
        print(f"Episode {episode:3d} | Score: {score:3d}")

env.close()
print("\nDone! The agent should be balancing for longer now.")
