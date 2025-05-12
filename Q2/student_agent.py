import gymnasium
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
import random
from torch.distributions import MultivariateNormal
class ReplayBuffer():
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.advantages = []
        self.values = []
        self.log_probs = []

    def push(self, state, action, reward, advantage, value, log_prob):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.advantages.append(advantage)
        self.values.append(value)  
        self.log_probs.append(log_prob)

    def sample(self):

        return (torch.tensor(self.states), 
                torch.tensor(self.actions), 
                torch.tensor(self.rewards),
                torch.tensor(self.advantages),
                torch.tensor(self.values),
                torch.tensor(self.log_probs))
    
    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.advantages = []
        self.values = []
        self.log_probs = []

class ValueNet(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=64):
        super(ValueNet, self).__init__()
        self.seq = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
    def forward(self, x):
        return self.seq(x)
class PolicyNet(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=64):
        super(PolicyNet, self).__init__()
        self.seq = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.mu = nn.Linear(hidden_size, output_size)

        # Global log_std as a learnable parameter (not state-dependent)
        self.log_std = nn.Parameter(torch.zeros(output_size))

    def forward(self, x):
        x = self.seq(x)
        mu = torch.tanh(self.mu(x))  # If env expects bounded actions like [-1, 1]
        std = torch.exp(self.log_std)  # Broadcasts over batch
        return mu, std
class PPO_agent:
    def __init__(self, input_dim, action_dim, gamma=0.99, learning_rate = 3e-4):

        self.policy_net = PolicyNet(input_dim, action_dim)
        self.critic_net = ValueNet(input_dim,1)
        self.optimizer = torch.optim.Adam([  # Update both models together
            {'params': self.policy_net.parameters(), 'lr': learning_rate},
            {'params': self.critic_net.parameters(), 'lr': learning_rate}
                    ])

        self.memory = ReplayBuffer()
        self.gamma = gamma
        self.lambda_ = 1
        self.vf_coef = 0.5 # c1
        self.entropy_coef = 0.01  # c2
        # use fixed std
        #self.std = torch.diag(torch.full(size=(1,)))
        self.eps = 0.2
    def get_advantages(self,rewards, values, gamma=0.99, lambda_=1):
        advantages = torch.zeros_like(torch.as_tensor(rewards))
        sum = 0
        for t in reversed(range(len(rewards)-1)):
            delta = rewards[t] + gamma * values[t + 1] - values[t]
            sum = delta + gamma * lambda_ * sum
            advantages[t] = sum
        return advantages
    def generate_trajectory(self):
        current_state, info = env.reset(seed=np.random.randint(0, 1000000))
        states = []
        actions = []
        rewards = []
        log_probs = []
                  
        for t in range(num_timesteps):
            with torch.no_grad():
                mean, std = self.policy_net(torch.as_tensor(np.array(current_state, dtype=np.float32)))
                normal = MultivariateNormal(mean, torch.diag_embed(std))
                action = normal.sample().detach()
                log_prob = normal.log_prob(action).detach()

            next_state, reward, terminated, truncated, info = env.step(action)

            states.append(np.array(current_state, dtype=np.float32))
            actions.append(action)
            rewards.append(reward)
            log_probs.append(log_prob)
        
            current_state = next_state
        
        # calculate values
        values = self.critic_net(torch.as_tensor(np.array(states, dtype=np.float32))).squeeze()
        # calculate advantages
        advantages = self.get_advantages(rewards, values.detach(), self.gamma, self.lambda_)
        # save the transitions in replay memory
        for t in range(len(advantages)):
            self.memory.push(states[t], actions[t], rewards[t], advantages[t], values[t], log_probs[t])
    def train(self):
        
        for iter_num in range(num_iterations): # k

            # collect a number of trajectories and save the transitions in replay memory
            for _ in range(num_trajectories):
                self.generate_trajectory()

            # sample from replay memory
            states, actions, rewards, advantages, values, log_probs = self.memory.sample()
            print("mean_advantages: ", torch.mean(advantages), end = ' | ')
            print("mean_values: ", torch.mean(values), end = ' | ')
            print("mean_reward: ", torch.mean(rewards))

            actor_losses = []
            critic_losses = []
            total_losses = []
            #reward_list = []
            for e in range(epochs):

                # calculate the new log prob
                mean, std = self.policy_net(states)
                normal = MultivariateNormal(mean, torch.diag_embed(std))
                new_log_probs = normal.log_prob(actions.unsqueeze(-1))

                r = torch.exp(new_log_probs - log_probs)

                clipped_r = torch.clamp(r, 1 - self.eps, 1 + self.eps)

                new_values = self.critic_net(states).squeeze()
                returns = (advantages + values).detach()

                actor_loss = (-torch.min(r * advantages, clipped_r * advantages)).mean()
                critic_loss = nn.MSELoss()(new_values.float(), returns.float())

                # Calcualte total loss
                total_loss = actor_loss + (self.vf_coef * critic_loss) - (self.entropy_coef * normal.entropy().mean())

                # update policy and critic network
                self.optimizer.zero_grad()
                total_loss.backward(retain_graph=True)
                self.optimizer.step()

                actor_losses.append(actor_loss.item())
                critic_losses.append(critic_loss.item())
                total_losses.append(total_loss.item())
                #reward_list.append(sum(rewards))

            # clear replay memory
            self.memory.clear()



            print("iteration = ",iter_num, end= ' : ')
            print('Actor loss = ', np.mean(actor_losses), end= ' | ')
            print('Critic loss = ', np.mean(critic_losses), end= ' | ')
            print('Total Loss = ', np.mean(total_losses), end= ' | ')
            print("")
    def saved(self):
        torch.save(self.policy_net.state_dict(), f'policy_net.pt')
        torch.save(self.critic_net.state_dict(), f'critic_net.pt')
    
    def load(self):
        self.policy_net.load_state_dict(torch.load(f'policy_net.pt'))
        self.critic_net.load_state_dict(torch.load(f'critic_net.pt'))
    def get_action(self, state):
        state_tensor = torch.as_tensor(state, dtype=torch.float32)
        with torch.no_grad():
            mean, std = self.policy_net(state_tensor)
            normal = MultivariateNormal(mean, torch.diag_embed(std))
            action = normal.sample().cpu().numpy()
        return action

    def test(self, load = False, vedio = False):
        if load:
            self.policy_net.load_state_dict(torch.load(f'policy_net.pt'))
        current_state, info = env.reset(seed=np.random.randint(0, 1000000))
        total_reward = 0
        frames = []
        while True:
            action = self.get_action(current_state)
            next_state, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
            total_reward += reward
            frame = env.render()
            frames.append(frame)
            current_state = next_state

        print(total_reward)
        imageio.mimsave("demo.gif", frames, fps=30)
        env.close()

# Do not modify the input of the 'act' function and the '__init__' function. 
class Agent(object):
    """Agent that acts randomly."""
    def __init__(self):
        #self.action_space = gymnasium.spaces.Box(-1.0, 1.0, (1,), np.float64)
        self.agent = PPO_agent(5, 1)
        self.agent.load()

    def act(self, observation):
        return self.agent.get_action(observation)
