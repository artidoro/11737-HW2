import glob
data = '/home/ubuntu/courses/ass2_multilingual/assign2/data/ted_raw/'
langmap = {'aze': 'azc', 'bel': 'bec'}
folders = ['azc_eng', 'bec_eng']

if __name__ == '__main__':
	for lang in ['aze', 'bel']:
		print (lang)
		for dtype in ['train', 'test', 'dev']:
			print (dtype)
			for ftype in ['orig', 'mtok']:
				print (ftype)
				fname = 'ted-'+dtype+'.'+ftype+'.'+lang
				folder = data + langmap[lang]+'_eng/'
				f1 = open(folder+fname)
				f2 = open(folder+fname+'.epi')
				f1lines = f1.readlines()
				f2lines = f2.readlines()
				f3 = open(folder+'ted-'+dtype+'.'+ftype+'.'+langmap[lang], 'w')
				assert len(f1lines) == len(f2lines)
				for i, line in enumerate(f1lines):
					f3.write(line.strip() + ' ' + f2lines[i].strip() + '\n')




