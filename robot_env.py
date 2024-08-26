from TabularQLearner import TabularQLearner
from DoubleQLearner import DoubleQLearner
import argparse
import matplotlib.pyplot as plt
import numpy as np
import sys, time

TRIPS_WITHOUT_DYNA = 500
TRIPS_WITH_DYNA = 50
FAILURE_RATE = 0.2

MOVE_OFFSET = ( (-1,0), (0,1), (1,0), (0,-1) )
TIME_PENALTY = -1

def find_unique_location (world, item):
  """
  Returns a tuple of the x,y coordinate of an item in the world.
  The item is identified by a one-character string representation
  as it appears in the world text file, e.g. 'g' for the goal.
  
  If the item appears more than once, one is arbitrarily chosen.
  If the item does not appear, None is returned.
  """

  result = np.where(world == item)
  return (result[0][0], result[1][0]) if len(result[0]) > 0 else None


def calc_state (loc):
  """ Quantizes the state to a single number: row*10 + column. """
  return loc[0]*10 + loc[1]


def query_world (world, s, a, fail = FAILURE_RATE):
  """
  Apply the action to the state and return the new state and reward.
  The default failure rate can be overridden, for example, to compute
  the learned optimal path from start to goal.
  """

  # See if the attempted action succeeded.
  if np.random.random() < fail:
    a = np.random.randint(4)

  # Determine the result of the action.
  s_prime = s[0] + MOVE_OFFSET[a][0], s[1] + MOVE_OFFSET[a][1]
  row, col = s_prime

  # Ensure the action was possible, roll back if needed, and set the reward.
  if row < 0 or row >= world.shape[0] or col < 0 or col >= world.shape[1]:
    # Ran into exterior wall.
    s_prime, r = s, TIME_PENALTY
  elif world[s_prime] == 'X':
    # Ran into interior obstacle.
    s_prime, r = s, TIME_PENALTY
  elif world[s_prime] == 'q':
    # Walked into quicksand.
    r = -100
  elif world[s_prime] == 'g':
    # Walked into goal.
    r = 1
  else:
    # Legal move to empty space with default reward.
    r = TIME_PENALTY

  return s_prime, r


def test_world (world, dyna, trips, trial, learner_class, show_path = True):
  """
  Run an experiment with a persistent learning taking many trips through a robot world,
  possibly with dyna iterations of the Dyna-Q algorithm after each experience,
  and return the median trip reward.
  """

  # Create a learner to interact with the world.
  eps = 0.98 if dyna == 0 else 0.5
  eps_decay = 0.999 if dyna == 0 else 0.99

  if dyna > 0:
    Q = learner_class(states = world.size, actions = 4, epsilon = eps, epsilon_decay = eps_decay, dyna = dyna)
  else:
    Q = learner_class(states = world.size, actions = 4, epsilon = eps, epsilon_decay = eps_decay)

  # Initialize the robot and goal position.
  start = find_unique_location(world, 's')
  goal = find_unique_location(world, 'g')

  # Remember the total rewards of each trip individually.
  trip_rewards = []
  trip_state_values = []

  # Each loop is one complete trip through the maze.
  for trip in range(trips):

    # A new trip starts with the robot at the start with no rewards.
    # Get the initial action.
    s = start
    trip_reward = 0
    a = Q.test(calc_state(s), allow_random=True)

    # There is a step limit per trip to prevent infinite execution.
    steps_remaining = 10000

    if show_path: path = world.copy()

    # Each loop is one step in the maze.
    while s != goal and steps_remaining > 0:

      # Apply the most recent action and determine its reward.
      s, r = query_world(world, s, a)

      if show_path: path[s] = '+'

      # Allow the learner to experience what happened.
      a = Q.train(calc_state(s), r)

      # Accumulate the total rewards for this trip.
      trip_reward += r

      # Elapse time.
      steps_remaining -= 1

    # Remember the total reward of each trip.
    trip_rewards.append(trip_reward)
    trip_state_values.append(Q.getStateValues())

    # Print a line of what happened this trip.
    if show_path:
      found_goal = "     GOAL NOT REACHED" if s != goal else ""
      print (f"\nTrial {trial}, trip {trip}, reward: {trip_reward:.04f}{found_goal}:")
      print(pretty_print(path))

      # Comment next line to run faster!
      time.sleep(0.25)

  # Always print the converged path.
  print_converged_path(world, trial, start, goal, Q)

  # Turn the state values after each trip into a proper 2-D numpy array.
  trip_state_values = np.array(trip_state_values)

  return np.median(np.array(trip_rewards)), trip_state_values


def print_converged_path(world, trial, start, goal, Q):
  """ Prints the Q-learner's learned path (no exploration, no action failure). """
  s = start
  a = Q.test(calc_state(s))

  # Still use a step limit in case the learner has a circular policy
  # that does not lead to the goal.
  steps_remaining = 1000

  path = world.copy()

  while s != goal and steps_remaining > 0:
    s, r = query_world(world, s, a, fail = 0.0)
    if s != goal: path[s] = '+'
    a = Q.test(calc_state(s))
    steps_remaining -= 1

  if steps_remaining <= 0: print("\nConverged path does not reach goal!\n" + \
                                 "(It is okay for this to occasionally happen.)")

  print (f"\nConverged path: trial {trial}\n{pretty_print(path)}\n")


def pretty_print (path):
  """ Returns a pretty-printed view of the world and the robot's path through it. """

  p = np.array2string(path, separator='', formatter={'str_kind': lambda x: x})
  return p.replace("[["," [").replace("]]","] ").translate(str.maketrans('.+[]X', ' .||#'))


if __name__ == '__main__':
  # Load the world, instantiate a Q-Learning agent, and help it navigate.

  # The program requires the path to a world file.  No default is applied.
  # Optional arguments allow Double Q learning (default: don't) or the
  # hallucination of N Dyna-Q experiences after each real experience (default: 0).

  parser = argparse.ArgumentParser(description='Robot environment for Q-Learning.')
  parser.add_argument('--world', metavar='WORLD', required=True, help='Path to robot world file.')
  parser.add_argument('--dyna', metavar='DYNA_PER_EXP', default=0, type=int, help='Dyna iterations per experience.')
  parser.add_argument('--trips', metavar='WORLD_TRIPS', default=0, type=int, help='Number of trips through the world.')
  parser.add_argument('--double', action='store_true', help='If supplied, use Double Q.')
  parser.add_argument('--trials', metavar='NUM_TRIALS', default=1, type=int,
                      help='Number of complete experiemental trials.  Use only when all code works.')

  args = parser.parse_args()

  world = np.genfromtxt(args.world, dtype=np.dtype('U'), delimiter=1)
  trips = args.trips

  if trips == 0: trips = TRIPS_WITH_DYNA if args.dyna > 0 else TRIPS_WITHOUT_DYNA

  learner = DoubleQLearner if args.double else TabularQLearner

  scores = []
  tsvs = []

  # Run many experiments.
  for trial in range(args.trials):

    # Want to see each trip as a visual map?  Use this line.
    # score, trip_state_values = test_world(world, args.dyna, trips, trial, learner)

    # Tired of all the printing?  Use this line instead of the above.
    score, trip_state_values = test_world(world, args.dyna, trips, trial, learner, show_path = False)

    # Uncomment this to see the score for each trial individually.
    #print (f"After {trips} trips, median trip reward was: {score:.4f}")

    scores.append(score)
    tsvs.append(trip_state_values)

  print (f"After {args.trials} trials of {trips} trips, median trip reward was: {score:.4f}\n")

  # Original array is (trials, trips, states).  Want (trials, states, trips).
  trip_state_values = np.transpose(np.array(tsvs), (0, 2, 1))

  # For every state, for every trial, express the state value at the end of each trip
  # relative to the same state value after the final trip of that trial.
  # This shows the process of converging to the final Q values.
  trip_state_values = (trip_state_values[:,:,:-1] / trip_state_values[:,:,-1:]) - 1

  # Within each trial, what was the median state error after each trip?
  trip_state_values = np.median(trip_state_values, axis=1)

  # Plot the progression of each trial's median state error over its trips.
  plt.plot(trip_state_values.T)
  plt.show()

  # Plot the median of the above across all trials.
  trip_state_values = np.median(trip_state_values, axis=0)
  plt.plot(trip_state_values)
  plt.show()

  # Print a few intermediate trip values since the plot doesn't show fine detail.
  print (f"Median difference between estimated and final Q as of trip:")
  for trip, diff in [ (trip, trip_state_values[trip]) for trip in range(0,trips,int(trips/10)) ]:
    print (f"{trip:6d}: {diff:.6f}")

