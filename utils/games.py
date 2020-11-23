import autograd.numpy as np
import autograd.scipy as sp
from scipy import stats
import pandas as pd


class WetChicken2D:

    """
    Benchmark reinforcement learning problem where 
    a canoist is paddling in a river upstream of a waterfall
    and is rewarded for approaching it without being sent over.
    The canoeist can paddle upstream or side to side.
    The river has current (deterministic) and turbulence (stochastic),
    which vary along the width of the river (i.e. there is greater
    turbulence but slower flow along the river's left bank).

    This implementation is a simplified 2D version of the problem
    in discrete space. The rivers is represented as a grid,
    and current and flow are encoded in the transition
    probabilities between cells.

    The original Wet Chicken problem (in 1 dimension)
    is attributed to Volker Tresp in a 1994 Technical Report.
    Hans & Udluf (2009) state a 2D version (in continuous space):
    https://www.tu-ilmenau.de/fileadmin/media/neurob/publications/conferences_int/2009/Hans-ICANN-2009.pdf 
    """

    def __init__(self, L=5, W=3, max_steps=10, seed=None):

        # Initialize a random state:
        self.seed = seed
        self.random = np.random.RandomState(seed)

        # Define simulation parameters:
        self.max_steps = max_steps  # Maximum number of steps per epsiode.
        
        # Define grid size (in cells):
        self.L = L  # Length of river (the waterfall is at the boundary between 0 and 1).
        self.W = W  # Width of river (x=0 and x=W are in the river, x=-1 and x=W+1 are not).

        # Define valid actions:
        #   Canoeist can fight current, drift, or move side to side.
        #   (All actions include at least one unit upstream padding, to combat current).
        self.valid_actions = [(0,+1),(-1,+1),(+1,+1),(0,+2)]

        # Create placeholders for state values:
        self.state_history = None
        self.noise_history = None
        self.action_history = None

        # Initialize game history:
        self.new_game()

    def __repr__(self):
        return f"WetChicken2D(L={self.L}, W={self.W}, seed={self.seed})"

    def __str__(self):
        s = self.__repr__()
        s += " [{episodes} episodes, {steps} total steps, canoeist at {pos}]".format(
            episodes = self.episode_count,
            steps = self.total_step_count,
            pos = self.state,
        )
        return s

    def new_game(self):
        """
        Clears the game history.
        """

        # States are stored in a list of episodes; each episode is a list of steps.
        self.state_history = []  # List of lists.
        self.noise_history = []  # List of lists.
        self.action_history = []  # List of lists.

        # Begin new episode:
        self.new_episode()

    def new_episode(self):
        """
        Starts a new episode with a new initial state.
        """

        # Add new episode to history:
        self.state_history.append([])
        self.noise_history.append([])
        self.action_history.append([])
        
        # Start far upstream of the waterfall, at a random lateral position:
        x = self.random.choice(range(1,1+self.W))
        self.state = (x,self.L)

    @property
    def episode_count(self):
        return len(self.action_history)

    @property
    def step_count(self):
        return len(self.action_history[-1])

    @property
    def total_step_count(self):
        return sum([len(episode) for episode in self.action_history])

    @property
    def state(self):
        """
        Get latest (x,y) position (i.e. latest position in latest episode).
        """
        return self.state_history[-1][-1]
    
    @state.setter
    def state(self, state):
        x,y = state
        assert (x>0) and (x<=self.W), f"Invalid x: {x}"
        assert (y>=0) and (y<=self.L), f"Invalid y: {y}"
        self.state_history[-1].append(state)

    @property
    def noise(self):
        """
        Get noise associated with latest position.
        """
        len_noise = len(self.noise_history[-1])
        len_state = len(self.state_history[-1])
        if len_noise == len_state:
            return self.noise_history[-1][-1]
        elif len_noise == len_state-1:
            return None  # Latest noise not set yet.
        else:
            raise RuntimeError(f"Noise history ({len_noise}) should only be 0 or 1 entry shorter than position history ({len_state}).")
    
    @noise.setter
    def noise(self, noise):
        self.noise_history[-1].append(noise)

    @property
    def action(self):
        """
        Get noise associated with latest position.
        """
        len_action = len(self.action_history[-1])
        len_state = len(self.state_history[-1])
        if len_action == len_state:
            return self.action_history[-1][-1]
        elif len_action == len_state-1:
            return None  # Latest action not set yet.
        else:
            raise RuntimeError(f"Action history ({len_action}) should only be 0 or 1 entry shorter than position history ({len_state}).")
    
    @action.setter
    def action(self, action):
        assert action in self.valid_actions
        self.action_history[-1].append(action)        

    def simulate_noise(self):
        """
        Simulates turbulence (which depends on x position).
        Turbulence is highest on left bank, and decreases
        linearly from (W-1) to 0 as we move toward the right bank.
        Example if W==3:
            x=1  -->  Turbulence uniformly selected in [-2,+2].
            x=2  -->  Turbulence uniformly selected in [-1,+1].
            x=3  -->  Turbulence always 0.
        """

        (x,y) = self.state
        y_noise_max = x-1
        y_noise = self.random.choice(range(-y_noise_max,y_noise_max+1))
        return (0, y_noise)

    def simulate_transition(self):
        """
        Simuates movement based on noise, action, and current (which depends on x position).
        Turbulence is highest on right bank, and decreases
        linearly from (W-1) to 0 as we move toward the left bank.
        Example if W==3:
            x=1  -->  Current is 0.
            x=2  -->  Current is -1.
            x=3  -->  Current is -2.
        """
        pos = self.state
        turb = self.noise
        paddle = self.action
        current = (0,-(pos[0]-1))
        new_x = pos[0] + turb[0] + paddle[0] + current[0]
        new_y = pos[1] + turb[1] + paddle[1] + current[1]
        new_x = max(1,min(new_x,self.W))
        new_y = max(0,min(new_y,self.L))  # Episode ends if y==0.
        return (new_x, new_y)

    def select_action(self, policy=None):
        """
        Get action according to random or custom policy.
        If `policy` is specified, it should be a function
        that takes this environment as a parameter
        and returns a valid action.
        """

        if policy is None:
            i = self.random.choice(range(len(self.valid_actions)))
            action = self.valid_actions[i]
            return action
        else:
            action = policy(self)
            assert action in self.valid_actions, f"Action {action} is not a valid action: {self.valid_actions}"
            return action

    def update(self, policy=None):
        """
        Simulate one step and returns True if the episode will continue or False if it has ended.
        """
        assert self.state[1]>=0, f"Episode already over -- canoeist has gone over the waterfall (pos={self.state})."
        assert self.step_count<self.max_steps, f"Episode already over -- step count ({self.step_count}) reached max ({self.max_steps})."

        self.noise = self.simulate_noise()
        self.action = self.select_action(policy=policy)
        self.state = self.simulate_transition()

        if ( self.state[1] == 0 ) or ( self.step_count >= self.max_steps ):
            return False
        else:
            return True

    def run(self, episodes=1_000, policy=None, progress=None, max_total_steps=1e6):
        total_steps = 0
        for ep in range(episodes):
            while total_steps<max_total_steps and self.update(policy=policy):
                total_steps += 1
            if progress and (ep+1)%progress == 0:
                print(f"Episode {ep+1}/{episodes} took {self.step_count} steps.")
            self.new_episode()

    def extract_transition_dataset(self):

        transitions = []

        for ep in range(self.episode_count):

            ep_state_history = self.state_history[ep]
            ep_noise_history = self.noise_history[ep]
            ep_action_history = self.action_history[ep]

            for t in range(len(ep_action_history)):

                s = ep_state_history[t]
                a = ep_action_history[t]
                s_ = ep_state_history[t+1]

                observation = {
                    'start_x' : s[0],
                    'start_y' : s[1],
                    'action_x' : a[0],
                    'action_y' : a[1],
                    'result_x' : s_[0],
                    'result_y' : s_[1],
                }
                transitions.append(observation)

        transitions = pd.DataFrame(transitions)

        return transitions

