import string

NUM_CHARS = len(string.printable)

# auxiliary tokens
BLANK = 0

NUM_TOKENS = NUM_CHARS + 1

char2ids = {c : i + 1 for i, c in enumerate(string.printable)}
ids2chars = {(i+1): c for i, c in enumerate(string.printable)}
ids2chars[BLANK] = '-'
