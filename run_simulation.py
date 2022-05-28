import numpy as np

from tqdm import tqdm

class GaussianDistribution:
    """
    A wrapper class with gaussian parameters for gaussian sampling.
    Call to the object gives a NumPy vector of size sampled using internal parameters.
    Default sample size is 1.
    """
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma
    
    def __call__(self, sample_size=1):
        return np.random.normal(self.mu, self.sigma, sample_size)


MASTERS_DURATION = 730


def max_reward(monte_carlo_iter, schools):
    # Max reward
    print()
    bar = tqdm(range(monte_carlo_iter), "MAX_REWARD", total=monte_carlo_iter, leave=True)
    total_rewards = []
    for _ in bar:
        total_reward = 0
        for _ in range(MASTERS_DURATION):        
            rewards = [s()[0] for s in schools]
            total_reward += max(rewards)
        total_rewards.append(total_reward)
    print(f"Expected max reward: {int(np.mean(total_rewards))}")


def explore_only(monte_carlo_iter, schools):
    # Explore only
    print()
    bar = tqdm(range(monte_carlo_iter), "EXPLORE ONLY", total=monte_carlo_iter, leave=True)
    total_rewards = []
    total_regrets = []
    for _ in bar:
        total_reward = 0
        total_regret = 0
        for _ in range(MASTERS_DURATION):
            # Choose school to explore
            school_id = np.random.randint(0, len(schools))
            
            rewards = [s()[0] for s in schools]

            total_reward += rewards[school_id]
            total_regret += max(rewards) - rewards[school_id]
        
        total_rewards.append(total_reward)
        total_regrets.append(total_regret)
        
    print(f"Expected total reward: {int(np.mean(total_rewards))}")
    print(f"Expected total regret: {int(np.mean(total_regrets))}")


def exploit_only(monte_carlo_iter, schools):
    # Exploit only
    print()
    bar = tqdm(range(monte_carlo_iter), "EXPLOIT ONLY", total=monte_carlo_iter, leave=True)
    total_rewards = []
    total_regrets = []
    for _ in bar:
        total_reward = 0
        total_regret = 0
        
        exploit_school_id = -1
        exploit_school_reward = 0

        # Explore phase
        for i in range(len(schools)):
            rewards = [s()[0] for s in schools]
            
            if rewards[i] > exploit_school_reward:
                exploit_school_id = i

            total_reward += rewards[i]
            total_regret += max(rewards) - rewards[i]

        # Exploit phase
        for _ in range(MASTERS_DURATION-len(schools)):
            rewards = [s()[0] for s in schools]

            total_reward += rewards[exploit_school_id]
            total_regret += max(rewards) - rewards[exploit_school_id]
        
        total_rewards.append(total_reward)
        total_regrets.append(total_regret)
        
    print(f"Expected total reward: {int(np.mean(total_rewards))}")
    print(f"Expected total regret: {int(np.mean(total_regrets))}")


def epsilon_greedy(monte_carlo_iter, schools, epsilon):
    # Epsilon greedy
    print()
    bar = tqdm(
        range(monte_carlo_iter), "EPSILON GREEDY {epsilon}", total=monte_carlo_iter, leave=True
    )

    # Use distribution means until first exploration
    first_explore = False
    max_expected_school = np.argmax([school.mu for school in schools])

    total_rewards = []
    total_regrets = []
    for _ in bar:
        # We need to keep track of most rewarding action
        total_reward = [0] * 3
        total_regret = [0] * 3
        reward_counts = [1] * 3

        for _ in range(MASTERS_DURATION):
            should_explore = np.random.rand() <= epsilon
            
            # Explore case
            if should_explore:
                if not first_explore:
                    first_explore = True

                # Choose school to explore
                school_id = np.random.randint(0, len(schools))

                rewards = [s()[0] for s in schools]

                total_reward[school_id] += rewards[school_id]
                reward_counts[school_id] += 1
                
                total_regret[school_id] += max(rewards) - rewards[school_id]

            # Exploit case
            else:
                # Exploit the school with highest expected reward
                if first_explore:
                    school_id = np.argmax(np.divide(total_reward, reward_counts))
                else:
                    school_id = max_expected_school
                
                rewards = [s()[0] for s in schools]

                total_reward[school_id] += rewards[school_id]
                reward_counts[school_id] += 1

                total_regret[school_id] += max(rewards) - rewards[school_id]
        
        total_rewards.append(np.sum(total_reward))
        total_regrets.append(np.sum(total_regret))
    
    print(f"Expected total reward: {int(np.mean(total_rewards))}")
    print(f"Expected total regret: {int(np.mean(total_regrets))}")


if __name__ == "__main__":

    # Determinism
    np.random.seed(42)

    # Number of sample of masters degree experience
    monte_carlo_iter = 1000

    # Yes, masters is not a daily event/experience so changing schools everyday
    # is not realistic but whatever...

    # Reward distribution of graduate students for each school
    # The parameters are carefully crafted with the process of making things up
    bilkent = GaussianDistribution(9, 3)
    odtu = GaussianDistribution(6, 10)
    koc = GaussianDistribution(4, 4)

    schools = [bilkent, odtu, koc]

    max_reward(monte_carlo_iter, schools)

    explore_only(monte_carlo_iter, schools)

    exploit_only(monte_carlo_iter, schools)

    epsilon_greedy(monte_carlo_iter, schools, 0.1)
