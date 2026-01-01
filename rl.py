import torch
import torch.nn as nn
from pong import PongGame

class Policy(nn.Module):
    def __init__(self, inputs, outputs):
        super().__init__()
        self.fc1 = nn.Linear(inputs, 128)
        self.fc2 = nn.Linear(128, outputs)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return torch.softmax(self.fc2(x), dim=-1)


class PongEnv:
    def __init__(self, player, render=False):
        self.render = render
        self.game = PongGame(render=self.render)
        self.player = player
        self.opponent = "left" if player == "right" else "right"
        self.actions = self.game.get_possible_actions()
        self.observations = self.game.get_state()

    def reset(self):
        self.game = PongGame(self.render)
        state = self.game.get_state()
        return self._get_state(state)

    def step(self, action):
        # perform action
        self.game.move_player(self.player, self.actions[action])

        # update game with action and get new state
        game_state = self.game.update()

        # calculate reward based on state
        reward = self._calculate_reward(game_state)

        # check if the game is done
        done = game_state["side_lost"] is not None

        state = self._get_state(game_state)

        return state, reward, done, {}

    def _get_state(self, game_state):
        return [
            game_state["ball_x"] / 1000,
            game_state["ball_y"] / 600,
            game_state["ball_vel_x"] / 500,
            game_state["ball_vel_y"] / 500,
            game_state["player_" + self.player + "_y"] / 800,
        ]

    def _calculate_reward(self, state):
        reward = 0.0
        if state["side_lost"] == self.player:
            reward += -10.0
        elif state["side_lost"] == self.opponent: 
            reward += 10.0

        colliding_with_player = "ball_colliding_" + self.player
        if state[colliding_with_player]:
            reward += 1.0

        return reward

    def _get_nr_actions(self):
        return len(self.actions)

    def _get_nr_observations(self):
        return len(self._get_state(self.game.get_state()))


def train_episode():
    log_probs = []
    rewards = []
    state = env.reset()
    done = False

    while not done:
        state_tensor = torch.FloatTensor(state)
        probs = policy(state_tensor)
        action = torch.multinomial(probs, 1).item()
        log_probs.append(torch.log(probs[action]))

        state, reward, done, _ = env.step(action)
        rewards.append(reward)
        if reward != 0:
            print(f"Action: {action}, Reward: {reward}, Done: {done}")

    returns = []
    G = 0
    for r in reversed(rewards):
        G = r + 0.99 * G
        returns.insert(0, G)

    print("Episode finished. Total reward:", sum(rewards))

    # normalize returns
    returns = torch.FloatTensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)

    # policy gradient update
    loss = []
    for log_prob, G in zip(log_probs, returns):
        loss.append(-log_prob * G)

    optimizer.zero_grad()
    torch.stack(loss).sum().backward()
    optimizer.step()
    print("Training step completed.")

if __name__ == "__main__":
    env = PongEnv(player="right", render=True)
    policy = Policy(inputs=env._get_nr_observations(), outputs=env._get_nr_actions())  # 8 state variables, 3 actions (up, down, idle)
    policy.load_state_dict(torch.load("pong_policy.pth"))
    optimizer = torch.optim.Adam(policy.parameters(), lr=0.01)

    for episode in range(1000):
        train_episode()

