# Importing the core files for pytorch
import torch
import torch.nn as nn
import torch.optim as optim

# Setup the Q Network

class Q_Network(nn.Module):

    def __init__(self , input_dimensions , hidden_dimensions , output_dimensions):
        super(Q_Network , self).__init__()
        self.net = nn.Sequential(
        nn.Linear(input_dimensions , hidden_dimensions),
        nn.ReLU(),
        nn.Linear(hidden_dimensions, output_dimensions)
        )
    def forward( self , x):
        return self.net(x)


# Setting up the Regret Agent 

class Regret_Agent:
    def __init__(self , state_dimensions , action_dimensions , hidden_dimensions = 64 , lr = 0.001 , device = None):
        self.device = device or torch.device( "cuda" if torch.cuda.is_available() else "cpu" )
        self.q_net = Q_Network(state_dimensions , action_dimensions , hidden_dimensions).to(self.device)
        self.optimizer = optim.Adam(self.q_net.parameters() , lr = lr)
        self.loss_fn = nn.MSELoss()
        self.action_dimensions = action_dimensions

    # Selcting an action using the Epsilon greddy policy
    def select_actions(self , states , epsilon):
        

        states = states.to(self.device)
        batch_size = states.shape[0]
        
        with torch.no_grad():
            q_values = self.q_net(states)
            greedy_actions =q_values.argmax(dim=1)
            return torch.argmax(q_values).item()


        random_actions = torch.randint(0, self.action_dimensions, (batch_size,) , device = self.device)
        probs = torch.rand(batch_size , device = self.device)

        actions =torch.where(probs < epsilon , random_actions , greedy_actions)
        assert actions.max() < self.actions and actions.min() >= 0, "Action index out of bounds!"
        
        return actions
        
    #Calculating Loss and Regret
    def compute_loss_and_regret (self, states , actions , next_states, rewards, dones , gamma = 0.99):
        # Transferring stuff to the right device
        states =     states.to(self.device)
        actions=     actions.to(self.device)
        next_states= next_states.to(self.device)
        rewards=     rewards.to(self.device)
        dones =      dones.to(self.device)

        q_values = self.q_net(states)
        q_sa = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad:
            next_q_values = self.q_net(next_states)
            v_targets = next_q_values.max(dim = 1)[0]
            targets = rewards + gamma * v_targets *(~dones)

        regrets = targets - q_sa
        loss = self.loss_fn(q_sa , targets)
        return loss , regrets.mean().item()
    
    def train_step (self, states , actions , next_states , rewards , dones):
        self.q_net.train()
        self.optimizer.zero_grad()
        loss , average_regret = self.compute_loss_and_regret(states , actions , next_states, rewards, dones )
        loss.backward()
        self.optimizer.step()
        return loss.item(), average_regret


            













