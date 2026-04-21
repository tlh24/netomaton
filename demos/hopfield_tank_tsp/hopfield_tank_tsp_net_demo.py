import netomaton as ntm
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
	successes = np.zeros(10)
	for j in range(10): 
		N = 10
		# points = [(0, 1), (0.23, 0.5), (0.6, 0.77), (0.33, 0.88), (0.25, 0.99),
		#           (0.55, 0.25), (0.67, 0.78), (0.12, 0.35), (0.19, 0.89), (0.40, 0.23)]
		points = np.random.rand(N, 2)

		# A, B, C, D, n, dt, timesteps = 500, 500, 200, 500, 15, 0.001, 1000  # avg. 2.67325154944775, 20% convergence
		A, B, C, D, n, dt, timesteps = 300, 300, 100, 300, 12, 1e-05, 1000  # avg. 2.9605078829924527, 70% convergence
		# A, B, C, D, n, dt, timesteps = 400, 400, 150, 400, 12, 1e-05, 1000  # avg. 3.1818680173024543, 80% convergence
		# A, B, C, D, n, dt, timesteps = 500, 500, 150, 300, 12, 1e-05, 1000  # avg. 3.4807220504123797, 100% convergence

		tsp_net = ntm.HopfieldTankTSPNet(points, dt=dt, A=A, B=B, C=C, D=D, n=n)

		adjacency_matrix = tsp_net.adjacency_matrix

		# -0.022 was chosen so that the sum of V for all nodes is 10; some noise is added to break the symmetry
		initial_conditions = [-0.022 + np.random.uniform(-0.1*0.02, 0.1*0.02) for _ in range(len(adjacency_matrix))]

		trajectory = ntm.evolve(initial_conditions=initial_conditions, activity_rule=tsp_net.activity_rule,
										network=ntm.topology.from_adjacency_matrix(adjacency_matrix), timesteps=timesteps)

		# ntm.animate_activities(trajectory, shape=(N, N))

		activities = ntm.get_activities_over_time_as_list(trajectory)
		permutation_matrix = tsp_net.get_permutation_matrix(activities)
		print(permutation_matrix)

		try: 
			G, pos, length = tsp_net.get_tour_graph(points, permutation_matrix)
			print(length)
			tsp_net.plot_tour(G, pos)
			plt.savefig(f'tour_{j}.png')
			plt.close()
			successes[j] = 1
		except Exception as e:
			print(f"{e}")

		print(f'done with {j}')
	print(f'successes: {np.sum(successes)}')
