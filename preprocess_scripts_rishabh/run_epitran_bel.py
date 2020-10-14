import epitran
from glob import glob

lang_script = {'bel': 'Cyrl', 'tur': 'Latn', 'rus': 'Cyrl'}

if __name__=='__main__':
	for lang in ['bel']: # no bel
		folder = 'data/ted_raw/'+lang+'_eng/*'
		files = glob(folder)
		for file in files:
			if file+'.epi' in files:		# skip if epi done
				continue
			if file.endswith('.bel'):
				epi_lang = epitran.Epitran('ukr-Cyrl')
			elif file.endswith('.eng'):
				epi_lang = epitran.Epitran('eng-Latn')
				continue
			else:
				continue

			print ('Started file : ', file)
			f = open(file, 'r')
			wf = open(file+'.epi', 'w')
			lines = f.readlines()
			for idx, line in enumerate(lines):
				print (f'{idx}/{len(lines)}', end = '\r')
				newline = ' '.join([epi_lang.transliterate(l) for l in line.strip().split()])
				#print (newline)
				#import pdb; pdb.set_trace()
				wf.write(newline+'\n')
			f.close()
			wf.close()
			print ()
			print ('Wrote file : '+file+'.epi')



		
