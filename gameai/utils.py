def mask_policy(policy, valid_actions):
    '''
    Mask a policy with a list of valid actions and renormalize

    Args:
        policy (list): The policy returned from a network
        valid_actions (list): List of valid actions returned from the 
            game action space

    Returns:
        list: The normalized and masked actions
    '''
    valid_policy = [
        policy[i] if i in valid_actions else 0 for i in range(len(policy))]

    if sum(valid_policy) == 0:
        print('No valid actions from policy, using uniform distribution')
        return [1 / len(valid_actions) if a in valid_actions else 0 for a in range(len(policy))]

    return [a / sum(valid_policy) for a in valid_policy]
