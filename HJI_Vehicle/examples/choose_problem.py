##### Change this line for different a different problem. #####

system = 'vehicle'
# system = 'burgers'

# time_dependent = False  # True
time_dependent = True  # True

if system == 'vehicle':
    from HJI_Vehicle.examples.vehicle.problem_def import setup_problem, config_NN
# elif system == 'burgers':
#     from examples.burgers.problem_def import setup_problem, config_NN

problem = setup_problem()
config = config_NN(problem.N_states, time_dependent)

# if system == 'burgers':
#     system += '/D' + str(problem.N_states)
