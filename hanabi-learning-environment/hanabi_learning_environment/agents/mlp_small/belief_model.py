
def get_hand_probas(observation):
	"""Claculate hand probabilities (belief model -  level 0)

	Args:
	  fireworks
	  observed_hands
	  discard_pile
	  card_knowledge

	Returns:
	  probas: list, For each card in hand, probability of it belonging to one of the 25 
	    possiblecards. Contains 125 elements (25 probas for each of the 5 cards in hand)
	"""

	# Extract 4 relevant variables from observation
	fi = observation['fireworks']
	oh = observation['observed_hands']
	dp = observation['discard_pile']
	ck  = observation['card_knowledge']

	# Pre process the input

	fireworks = [fi['R'], fi['Y'], fi['G'], fi['W'], fi['B']]

	# print("fi: ", fi)
	# print("fireworks: ", fireworks)

	observed_hands = []
	if len(oh) > 0:
		for e in oh[1]:
			observed_hands.append(e['color'] + str(e['rank']))

	# print("oh: ", oh)
	# print("observed_hands: ", observed_hands)

	discard_pile = []
	if len(dp) > 0:
		for e in dp:
			discard_pile.append(e['color'] + str(e['rank']))

	# print("dp: ", dp)
	# print("discard_pile: ", discard_pile)

	# Create a list of all visible cards in the game
	visible_cards = []
	colors = 'RYGWB'

	# Add all possible cards from the fireworks list
	for i, j in enumerate(fireworks):
	  j = int(j)
	  if j > 0:
	    for k in range(0, j):
	      visible_cards.append(colors[i] + str(k))

	# Add team mates hand and discard pile to visible cards
	visible_cards.extend(observed_hands)
	visible_cards.extend(discard_pile)

	# print("visible_cards: ", visible_cards)

	# Create a string representation of each possible card that contains the color, rank, and possible count
	probas = []
	positions = ["R0_3", "R1_2", "R2_2", "R3_2", "R4_1", "Y0_3", "Y1_2", "Y2_2", "Y3_2", "Y4_1", "G0_3", "G1_2", "G2_2", "G3_2", "G4_1", "W0_3", "W1_2", "W2_2", "W3_2", "W4_1", "B0_3", "B1_2", "B2_2", "B3_2", "B4_1"]

	# Use hints in card knowledge to gather possible ranks and colors for each position

	for e in ck[0]:
		if e['color'] is None:
			colors = 'RYGWB'
		else:
			colors = e['color']
		if e['rank'] is None:
			ranks = '01234'
		else:
			ranks = str(e['rank'])

		l2 = []
		# Check how many cards are possible for each position based on hints and visible cards
		for position in positions:
			card = position.split("_")[0]
			possibles = int(position.split("_")[1])
			if card[0] in colors and card[1] in ranks:
				# print("card: ", card)
				# print("possibles: ", possibles)
				# print("vc: ", visible_cards.count(card))
				# print("visible_cards: ", visible_cards)
				l2.append(possibles - visible_cards.count(card))
			else:
				l2.append(0)

		# print("ck[0]: ", ck[0])
		# print("colors: ", colors)
		# print("ranks: ", ranks)
		# print("visible_cards: ", visible_cards)

		# print("l2: ", l2)
		# Caclulate probabilities based on possible card counts
		probas.extend([e/sum(l2) for e in l2])

	return probas