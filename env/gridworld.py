import torch

class Grid_World:
    # Init file will run only once since it is a constructor
    def __init__( self , batch_size = 128, size= 5, goal = (4,4) , Stochastic_probabilities = 0.05 , device = None  ):
        self.batch_size = batch_size
        self.size = size 
        self.goal = goal
        self.actions = 4
        self.device =  device or torch.device("cuda" if torch.cuda.is_available else "cpu")
        self.Stochastic_probabilities = Stochastic_probabilities
        

        #Setting up the Actions

        self.move_table = torch.tensor([
            [ 0,-1],
            [ 0, 1],
            [-1, 0],
            [ 1, 0]
            ], dtype = torch.int64 , device = self.device)
       
        
        self.reset()
    # Reset function to reset the states to zeroes after the completion of the episode
    def reset(self):
        self.positions = torch.zeros( (self.batch_size , 2), dtype= torch.int64 , device = self.device)
        self.done = torch.zeros((self.batch_size , 2) , dtype = bool , device = self.device)
        return self.positions.clone().float()
    

    def step( self, actions):
        
        rand = torch.rand(self.batch_size , device = self.device)
        real_stochastic = torch.tensor( self.Stochastic_probabilities , device = actions.device)
        expanded_probs = real_stochastic.expand(rand.shape)
        random_actions = torch.randint(0,4,(self.actions, ) , device = self.device)
        actual_actions = torch.where(rand < real_stochastic , random_actions , actions)

        # Looking up movement delatas

        deltas = self.move_table[actual_actions]  # shape = (batch_size , 2 )

        # Updating the positions

        new_pos = self.positions + deltas
        new_pos = torch.clamp(new_pos , 0 , self.size -1)

        self.positions = new_pos 

        # CHecking for goal

        reached_goal = (self.positions == self.goal).all(dim = 1)
        self.done = reached_goal 

        # Rewards 

        rewards = torch.where(reached_goal , 20 , -1 , device= self.device)

        return self.positions.clone().float() , rewards , self.done.clone()
    





    








