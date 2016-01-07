#!/usr/bin/python

import numpy as np

d = 2
N = 10 # N rand vectors
M = 5  # M hash functions

def sample_sphere(r = 1.0):
	theta = np.random.uniform(0, np.pi)
	phi = 2 * np.arcsin(np.sqrt(np.random.uniform(0, 1)))

	x = np.sin(phi) * np.cos(theta)
	y = np.sin(phi) * np.sin(theta)
	z = np.cos(phi)

	return np.array([x, y, z])

def cosine_dist(x, y):
	"""
	print cosine_dist(V[1], V[2])
	"""
	if x.size != 2 or y.size != 2:
		raise Exception("Invalid dimension")
	a = np.dot(x, y) / np.linalg.norm(x) / np.linalg.norm(y)
	return np.arccos(a)

# vectors
V = np.random.uniform(-1, 1, (N, d))
# hashes
H = np.vstack([sample_sphere() for i in range(M)])
