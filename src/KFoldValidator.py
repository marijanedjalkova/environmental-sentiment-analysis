def get_data_chunks(data, data_size, n):
	""" Produces equaly sized chunks of size n. Throws away the remainder. """
	if n == 0:
		n = data_size
	for i in range(0, data_size, n):
		if len(data[i:i + n]) == n:
			yield data[i:i + n]