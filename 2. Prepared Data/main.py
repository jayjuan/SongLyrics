import pandas as pd

file = pd.read_csv("tcc_ceds_music.csv")
for i in file['lyrics']:
	print(len(i))
print('run')