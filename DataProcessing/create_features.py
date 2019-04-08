# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np


# Time threshold (in ms) for n-grams (if too much time is elapsed between two events we can not count them as one event)
time_threshold = 1000

# List of common keys (ignore special characters, digits an special keys)
common_keys = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k",
			   "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w",
			   "x", "y", "z", "0", "1", "2", "3", "4", "5", "6", "7", "8",
			   "9", "Shift", "Tab"]

capital_keys = ["ShiftA", "ShiftB", "ShiftC", "ShiftD", "ShiftE",
				"ShiftF", "ShiftG", "ShiftH", "ShiftI", "ShiftJ",
				"ShiftK", "ShiftL", "ShiftM", "ShiftN", "ShiftO",
				"ShiftP", "ShiftQ", "ShiftR", "ShiftS", "ShiftT",
				"ShiftU", "ShiftV", "ShiftW", "ShiftX", "ShiftY",
				"ShiftZ"]

# All possible bi_numbers
bi_numbers = ["10", "11", "12", "13", "14", "15", "16", "17", "18", "19",
			  "20", "21", "22", "23", "24", "25", "26", "27", "28", "29",
			  "30", "31", "32", "33", "34", "35", "36", "37", "38", "39",
			  "40", "41", "42", "43", "44", "45", "46", "47", "48", "49",
			  "50", "51", "52", "53", "54", "55", "56", "57", "58", "59",
			  "60", "61", "62", "63", "64", "65", "66", "67", "68", "69",
			  "70", "71", "72", "73", "74", "75", "76", "77", "78", "79",
			  "80", "81", "82", "83", "84", "85", "86", "87", "88", "89",
			  "90", "91", "92", "93", "94", "95", "96", "97", "98", "99",
			  "00"]

# Exactly the tri_numbers that are asked for on Seite 1
tri_numbers = ["133", "337", "489", "893", "935", "355", "554", "541",
			   "201", "018", "014", "015", "016", "017", "813", "447",
			   "472", "721", "218", "180", "390", "900", "582", "827",
			   "163", "633"]

# 18 most common German bigrams + 10 most common same key bigrams + other bigrams
bigrams = ["er", "en", "ch", "de", "ei", "nd", "te", "in",
		   "ie", "ge", "es", "ne", "un", "st", "re", "he",
		   "an", "be",
		   "ss", "nn", "ll", "ee", "mm", "tt", "rr", "dd",
		   "ff", "aa",
		   "co", "ha", "no", "pa", "so", "to", "ca", "la",
		   "ia", "na", "ra", "ta", "ma", "sa", "da", "ce",
		   "ve", "me", "se", "ue", "le"]

# Most common German trigrams
trigrams = ["ich", "ein", "und", "der", "nde", "sch", "die",
			"den", "end", "cht", "das", "che", "gen", "ine",
			"nge", "nun", "ung", "hen", "ind", "enw", "ens",
			"ies", "ste", "ten", "ere", "lic", "ach", "ndi",
			"sse", "aus", "ers", "ebe", "erd", "enu", "nen",
			"rau", "ist", "nic", "sen", "ene", "nda", "ter",
			"ass", "ena", "ver", "wir", "wie", "ede", "ese",
			"auf", "ben", "ber", "eit", "ent", "est", "sei",
			"and", "ess", "ann", "esi", "ges", "nsc", "nwi",
			"tei", "eni", "ige", "aen", "era", "ern", "rde",
			"ren", "tun", "ing", "sta", "sie", "uer", "ege",
			"eck", "eru", "mme", "ner", "nds", "nst", "run",
			"sic", "enn", "ins", "mer", "rei", "eig", "eng",
			"erg", "ert", "erz", "fra", "hre", "hei", "lei",
			"nei", "nau", "sge", "tte", "wei", "abe", "chd",
			"des", "nte", "rge", "tes", "uns", "vor", "dem"]

def getFeatureByNumber(featNr):

	switcher = {
		0: "min",
		1: "max",
		2: "mean",
		3: "median",
		4: "var",
		5: "std"
	}

	switcher_2 = {
		0: "min value in x-direction of ",
		1: "min value in y-direction of ",
		2: "min value of np.linalg.norm() of ",
		3: "max value in x-direction of ",
		4: "max value in y-direction of ",
		5: "max value of np.linalg.norm() of ",
		6: "mean value in x-direction of ",
		7: "mean value in y-direction of ",
		8: "mean value of np.linalg.norm() of ",
		9: "median value in x-direction of ",
		10: "median value in y-direction of ",
		11: "median value of np.linalg.norm() of ",
		12: "var value in x-direction of ",
		13: "var value in y-direction of ",
		14: "var value of np.linalg.norm() of ",
		15: "std value in x-direction of ",
		16: "std value in y-direction of ",
		17: "std value of np.linalg.norm() of "
	}

	org = featNr
	featNr = featNr / 6
	operation = switcher.get(org % 6, "invalid argument ")

	f1 = len(bigrams) + len(bi_numbers) + len(capital_keys)
	f2 = f1 + f1
	f3 = f2 + 3 # mouse features velo
	f4 = f3 + 3 # mouse features acc
	f5 = f4 + len(common_keys)
	f6 = f5 + len(trigrams) + len(tri_numbers)
	f7 = f6 + len(trigrams) + len(tri_numbers)
	f8 = f7 + 1 # mouse scroll length
	f9 = f8 + 1 # mouse scroll time
	f10 = f9 + 1 # mouse scroll velo


	if featNr < f1:
		ngrams = bigrams + bi_numbers + capital_keys
		ngram = ngrams[featNr]
		return operation + " " + ngram + " (key_down)"

	elif f1 <= featNr < f2:
		ngrams = bigrams + bi_numbers + capital_keys
		ngram = ngrams[featNr - f1]
		return operation + " " + ngram + " (key_up)"

	elif f2 <= featNr < f3:
		n = org - (f2 * 6)
		return switcher_2.get(n, "invalid argument ") + "mouse_velocity"

	elif f3 <= featNr < f4:
		n = org - (f2 * 6 + 18)
		return switcher_2.get(n, "invalid argument ") + "mouse_acceleration"

	elif f4 <= featNr < f5:
		ngram = common_keys[featNr - f4]
		return operation + " " + ngram + " (key_up_down)"

	elif f5 <= featNr < f6:
		ngrams = trigrams + tri_numbers
		ngram = ngrams[featNr - f5]
		return operation + " " + ngram + " (key_up)"

	elif f6 <= featNr < f7:
		ngrams = trigrams + tri_numbers
		ngram = ngrams[featNr - f6]
		return operation + " " + ngram + " (key_down)"

	elif f7 <= featNr < f8:
		return operation + " (mouse_scroll_lenghts)"

	elif f8 <= featNr < f9:
		return operation + " (mouse_scroll_time)"

	elif f9 <= featNr < f10:
		return operation + " (mouse_scroll_velocity)"

	else:
		return "featNr too high"


def build_features_name_dict(num_features):
	return {i: getFeatureByNumber(i) for i in range(num_features)}



def create_keyboard_features(data, time_col_id='time'):
	# Collecting data
	ngrams = bigrams + bi_numbers + capital_keys
	stat = {k: [] for k in ngrams}

	last_time = None
	cur_bigram = ""
	for _, row in data.iterrows():
		if last_time is None:
			last_time = row[time_col_id]
			cur_bigram += row['key']
			continue

		time_passed = row[time_col_id] - last_time
		if time_passed <= time_threshold:
			cur_bigram += row['key']

			if cur_bigram in ngrams:
				stat[cur_bigram].append(time_passed)
		else:
			cur_bigram = ""

		last_time = row[time_col_id]
		cur_bigram = row['key']

	# Compute min, max, mean, avg, var, std
	stat_feat = {}
	for k, v in stat.iteritems():
		if v == []:
			v = [0]
		stat_feat[k] = [np.min(v), np.max(v), np.mean(v), np.median(v), np.var(v), np.std(v)]

	return np.array([v for _, v in stat_feat.iteritems()]).flatten()


def create_trigram_features(data, time_col_id='time'):
	ngrams = trigrams + tri_numbers
	stat = {k: [] for k in ngrams}

	first_time = None
	second_time = None
	second_key = None
	cur_trigram = ""
	count = None
	for _, row in data.iterrows():
		if first_time is None:
			first_time = row[time_col_id]
			cur_trigram += row['key']
			count = 1
			continue

		if count == 2:
			time_passed = row[time_col_id] - first_time
			count += 1
			if time_passed <= time_threshold:
				cur_trigram += row['key']

				if cur_trigram in ngrams:
					stat[cur_trigram].append(time_passed)
		else:
			cur_trigram += row['key']
			second_time = row[time_col_id]
			second_key = row['key']
			count += 1

		if count == 3:
			first_time = second_time
			second_time = None
			cur_trigram = second_key
			count = 1

	# Compute min, max, mean, avg, var, std
	stat_feat = {}
	for k, v in stat.iteritems():
		if v == []:
			v = [0]
		stat_feat[k] = [np.min(v), np.max(v), np.mean(v), np.median(v), np.var(v), np.std(v)]

	return np.array([v for _, v in stat_feat.iteritems()]).flatten()



def create_keyupdown_features(data_up, data_down, time_col_id='time'):
	# Collecting data
	stat = {k: [] for k in common_keys}

	for _, row in data_down.iterrows():
		# Key is pressed down
		key = row['key'].lower()
		dt = None

		if key not in common_keys:
			continue

		# When was the key released?
		for _, r in data_up[data_up[time_col_id] > row[time_col_id]].iterrows():  # TODO: Speed up!
			if r['key'].lower() == key:
				dt = r[time_col_id] - row[time_col_id]
				break
		if dt is None:
			continue    # Not released in this frame!

		# Store the current value
		stat[key].append(dt)

	# Compute min, max, mean, avg, var, std over the data
	# TODO: Look at some particular (common or uncommon) keys and do not throw everything into the same pot!
	stat_feat = []
	for _, v in stat.iteritems():
		if v == []:
			v = [0]
		stat_feat.append([np.min(v), np.max(v), np.mean(v), np.median(v), np.var(v), np.std(v)])

	return np.array(stat_feat).flatten()


def create_mouse_features(data, time_col_id='time'): 
	# Collecting data
	stat_velocities = []
	stat_accelerations = []

	old_pos = None
	old_v = None
	last_time = None
	for _, row in data.iterrows():
		if last_time is None:
			last_time = row[time_col_id]
			old_pos = np.array([row['x'], row['y']], dtype=np.float)
			continue

		new_pos = np.array([row['x'], row['y']], dtype=np.float)
		cur_time = row[time_col_id]

		dt = (cur_time - last_time) + 1e-5
		new_v = (new_pos - old_pos) / dt  # Compute current velocity

		if old_v is not None:   # If possible, compute current acceleration
			a = (new_v - old_v) / dt
			stat_accelerations.append([a[0], a[1], np.linalg.norm(a)])  # Include absolute/length value

		stat_velocities.append([new_v[0], new_v[1], np.linalg.norm(new_v)]) # Include absolute/length value

		old_pos = new_pos
		last_time = cur_time
		old_v = new_v


	# Compute min, max, mean, avg, var, std over velocities and accelerations
	if len(stat_velocities) == 0:
		stat_velocities = [[0, 0, 0]]
	stat_velocities_feat = (lambda v: np.array([np.min(v, axis=0), np.max(v, axis=0), np.mean(v, axis=0), np.median(v, axis=0), np.var(v, axis=0), np.std(v, axis=0)]))(stat_velocities)

	if len(stat_accelerations) == 0:
		stat_accelerations = [[0, 0, 0]]
	stat_accelerations_feat = (lambda a: np.array([np.min(a, axis=0), np.max(a, axis=0), np.mean(a, axis=0), np.median(a, axis=0), np.var(a, axis=0), np.std(a, axis=0)]))(stat_accelerations)

	return stat_velocities_feat.flatten(), stat_accelerations_feat.flatten()


def create_scroll_featues(data, time_col_id="time"):

	last_pos = None
	sub = None
	subsets = []
	for _, row in data.iterrows():
		if last_pos is None:
			last_pos = np.array([row['x'], row['y']], dtype=np.float)
			sub = [row]
			continue

		if last_pos[0] == row['x'] or last_pos[1] == row['y']:
			sub.append(row)
			last_pos = np.array([row['x'], row['y']], dtype=np.float)

		else:
			if len(sub) > 1:
				subsets.append(sub)
			sub = []
			last_pos = np.array([row['x'], row['y']], dtype=np.float)

	if len(subsets) is not 0:
		scroll_lengths = []
		scroll_times = []
		velocities = []
		for i in range(len(subsets)):
			sub = subsets[i]
			scroll_length = np.sum([row['dy'] for row in sub])
			scroll_lengths.append(scroll_length)
			dt = (sub[-1][time_col_id] - sub[0][time_col_id]) + 1e-5
			scroll_times.append(dt)
			velocities.append(scroll_length / dt)

		stats_len = [np.min(scroll_lengths), np.max(scroll_lengths), np.mean(scroll_lengths), np.median(scroll_lengths), np.var(scroll_lengths), np.std(scroll_lengths)]
		stats_tim = [np.min(scroll_times), np.max(scroll_times), np.mean(scroll_times), np.median(scroll_times), np.var(scroll_times), np.std(scroll_times)]
		stats_vel = [np.min(velocities), np.max(velocities), np.mean(velocities), np.median(velocities), np.var(velocities), np.std(velocities)]

		return np.array(stats_len), np.array(stats_tim), np.array(stats_vel)

	else:
		return np.array([0,0,0,0,0,0]), np.array([0,0,0,0,0,0]), np.array([0,0,0,0,0,0])


def create_features_from_slice(key_down, key_up, mouse_move, mouse_scroll, time_col_id):
	f1 = create_keyboard_features(key_down, time_col_id=time_col_id)
	f2 = create_keyboard_features(key_up, time_col_id=time_col_id)
	f3, f4 = create_mouse_features(mouse_move, time_col_id=time_col_id)
	f5 = create_keyupdown_features(key_up, key_down, time_col_id=time_col_id)
	f6 = create_trigram_features(key_down, time_col_id=time_col_id)
	f7 = create_trigram_features(key_up, time_col_id=time_col_id)
	f8, f9, f10 = create_scroll_featues(mouse_scroll, time_col_id=time_col_id)

	return np.concatenate((f1, f2, f3, f4, f5, f6, f7, f8, f9, f10))


def split(key_down, key_up, mouse_move, mouse_scroll, n_split=None, window_size=50, window_shift=50, time_col_id='time'):
	# Find first and last timestamp
	key_start = key_down.iloc[0,:][time_col_id]
	key_end = key_down.iloc[-1,:][time_col_id]

	mouse_start = mouse_move.iloc[0,:][time_col_id]
	mouse_end = mouse_move.iloc[-1,:][time_col_id]

	scroll_start = mouse_scroll.iloc[0,:][time_col_id]
	scroll_end = mouse_scroll.iloc[-1,:][time_col_id]

	start = np.min([key_start, mouse_start, scroll_start])
	end = np.max([key_end, mouse_end, scroll_end])

	# Compute window/batch and step size
	if n_split is None:     # Use window_size and window_shift if no number of splits have been specified
		step_size = window_shift * 1000      # 1000ms = 1s
		frame_size = window_size * 1000
	else:
		step_size = (end - start) / n_split
		frame_size = step_size

	# Split data into batches (sliding window approach)
	result = []
	last_end = start
	while last_end < end:
		kd = key_down[(key_down[time_col_id] > last_end) & (key_down[time_col_id] <= last_end + frame_size)]
		ku = key_up[(key_up[time_col_id] > last_end) & (key_up[time_col_id] <= last_end + frame_size)]
		mm = mouse_move[(mouse_move[time_col_id] > last_end) & (mouse_move[time_col_id] <= last_end + frame_size)]
		ms = mouse_scroll[(mouse_scroll[time_col_id] > last_end) & (mouse_scroll[time_col_id] <= last_end + frame_size)]

		if not(kd.shape[0] == 0 and ku.shape[0] == 0 and mm.shape[0] == 0 and ms.shape[0] == 0): # Drop slice if it does not contain any data
			result.append((kd, ku, mm, ms))

		last_end += step_size

	return result



def create_data(dir, n_split=None, window_size=50, window_shift=50, time_col_id='time'):
	result = []

	# Load data
	data_keydown = pd.read_csv(os.path.join(dir, "0.csv"))
	data_keyup = pd.read_csv(os.path.join(dir, "1.csv"))
	data_mousemove = pd.read_csv(os.path.join(dir, "2.csv"))
	data_mousescroll = pd.read_csv(os.path.join(dir, "6.csv"))

	# Split data
	data = split(data_keydown, data_keyup, data_mousemove, data_mousescroll, n_split, window_size, window_shift, time_col_id)
	for kd, ku, mm, ms in data:
		# For each slice/batch: Compute feature vector
		result.append(create_features_from_slice(kd, ku, mm, ms, time_col_id))

	return result
