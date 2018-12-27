'''
Collection of utility functions that can be shared between trainers
'''


def assign_rewards(examples, winner):
    ''' Assign rewards to the examples after the outcome is known '''
    return [[p, s, 1 if p == winner else 0] for [p, s, _] in examples]
