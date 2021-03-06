import pandas as pd
import numpy as np
import plotly as py
import plotly.express as px
import plotly.graph_objs as go
import fun_new
from plotly.subplots import make_subplots
from scipy.stats import chisquare

def read_pla(TABL):

	PHA1 = []
	PHA2 = []
	PHA3 = []

	i = 0
	j = 0
	k = 0
	l = 0
	for line in TABL: 
		line = line.strip()
		line = line.split()

	
		if 2 <= i <= 68:
			PHA1.append([])
			
			for u in line:
				PHA1[j].append(float(u))
			j=j+1
		
		if 140 <= i <= 206:
			PHA2.append([])
			
			for u in line:
				PHA2[k].append(float(u))
			k=k+1	  

		if 71 <= i <= 137:
			PHA3.append([])#PHA3 is PHA2
			
			for u in line:
				PHA3[l].append(float(u))
			l=l+1		
		
		i=i+1

	return PHA1, PHA2, PHA3

			


def get_canal (PHA1, PHA2, PHA3, data, CAL, T, canal, ID):
	
	#exec("%s = %d" % (canal_name,canal_name)) = np.array([], dtype = ['title 1', canal_name])
	G = []
	G_er = []
	matrix = []
	f = ' '
	if canal == 'G1' or canal == 'G2' or canal == 'G3':
		matrix = PHA1
		matrix1 = PHA3
	else:
		matrix = PHA2	

	for j in range(1,len(matrix[0])):
		g = []	
		for i in range(3, len(matrix)):
			if CAL*matrix[i][0] >= data.loc['Emin'] and CAL*matrix[i][0] < data.loc['Emax']:

				if j > 4:#проверка адеватности pla файла

					if matrix[i][j] == 0 and canal in ['G1', 'G2', 'G3']  and i != 3 and i != len(matrix)-1:	
						f = ' f'
						matrix[i][j] = matrix1[i][j]

						
					if j < len(matrix[i])-1 and matrix[i][j] == 0 and matrix[i][j+1] != 0 and j <= len(T) and canal in ['G1', 'G2', 'G3', 'G4'] and i != 3 and i != len(matrix)-1:
						f = ' f'
						matrix[i][j] = matrix[i][j-1]*matrix[1][j]/matrix[1][j-1]		

					elif j < len(matrix[i])-1 and matrix[i][j] == 0 and matrix[i][j+1] == 0 and j <= len(T) and canal in ['G1', 'G2', 'G3', 'G4'] and i != 3 and i != len(matrix)-1:			
						wr = 'ID = {}, в канале {}, при энергии от {}, начиная со спектра S{} G = 0\n'.format(ID, canal, matrix[i][0], j+1)
						ER = open('ER.txt', 'a')
						ER.write(str(wr))
						ER.close()
						return ['ER'], [], []  

					#if canal in ['G1', 'G2', 'G3', 'G4']
					elif matrix[i][j] == 0 and j > len(T):
						f = ' f_BG'
						matrix[i][j] = matrix[i][j-1]*matrix[1][j]/matrix[1][j-1]		

				g.append(matrix[i][j])

		G.append(sum(g)/matrix[1][j])
		G_er.append(((sum(g))**0.5)/matrix[1][j])

	for j in range(4, len(G)):

		if canal in ['G5', 'G6', 'G7'] and j < len(G)-1 and G[j] == 0 and G[j+1] != 0:
			 G[j] == G[j-1]
			 f = ' f'
		elif j < len(G) and canal in ['G5', 'G6', 'G7'] and G[j] == 0 and G[j+1] == 0:
			wr = 'ID = {}, в канале {}, начиная со спектра S{} G = 0\n'.format(ID, canal, j+1)
			ER = open('ER.txt', 'a')
			ER.write(str(wr))
			ER.close()
			return ['ER'], [], []  
				
	return G, G_er, f



	
		

	
	



def get_time(PHA1):
	t = []
	T = []

	for j in range(1,len(PHA1[0])):
		
		T.append(PHA1[0][j])
		t.append(PHA1[1][j])

	return T, t


def get_time_BG(T, T100):
	BT = []

	for i in T:
		if i >= T100:
			BT=T[T.index(i)+1:len(T)]
			del T[T.index(i)+1:len(T)]		
			continue	

	return T, BT



def get_BG_and_chi2(G, T):

	BG = []

	if len(T) + 2 < len(G):
		f_obs = G[len(T)-1:len(G)]
		BG = sum(f_obs)/len(f_obs)
		ddof = int(len(f_obs) - 1)
	
	elif len(T) + 2 >= len(G):
		f_obs = G[len(G)-5:len(G)]
		BG = sum(f_obs)/len(f_obs)
		ddof = int(len(f_obs) - 1)
		f = ' l'     

	f_exp = []
	for i in range(len(f_obs)):
		f_exp.append(BG) 

	
	c = chisquare(f_exp ,f_obs, ddof=0)

	del G[len(T)-1:len(G)]		

	return G, BG, c


def graf(G, T, G_er, BG, fig, canal, Range, T100):
	T_ = []
	G_ = []
	G_er_ = []

	for i in range(0,len(G),1):
		G_.append(G[i])
		G_.append(G[i])	
		G_er_.append(G_er[i])
		G_er_.append(G_er[i])

	for i in range(0,len(T)-1,1):	
		T_.append(T[i]) 
		T_.append(T[i+1]) 

	S = get_S(G_, BG, G_er_, canal)

	if canal == 'G4':
		row = 1
		col = 1
	elif canal == 'G5':
		row = 2
		col = 1	
	elif canal == 'G6':
		row = 3
		col = 1
	elif canal == 'G7':
		row = 4
		col = 1
	else:
		return S

	z = np.linspace(0, T_[len(T_)-1], 100)
	fig.append_trace(go.Scatter(x=T_, y=G_, marker={'color': 'blue'} ,mode = 'lines', fill=None, name = ''), row = row, col = col)

	d = np.linspace(BG, BG, 100)	
	fig.append_trace(go.Scatter(x=z, y=d, marker={'color': 'red'}, mode = 'lines', fill=None, name = ''), row = row, col = col)

	if BG == 0:
		return S

	text = canal

	i=0
	while i<len(S)-1:
		
		J=0
		_T=[]
		_G=[]
	
		while i+J+1 <= len(S) and S[i+J] > 5:
			_T.append(T_[i+J])
			_G.append(G_[i+J])
			J=J+1		
		else:
			J=J-1
		
		if J<1:
			i = i + 1
		else:
			i = i + J

			
	
		if len(_T) != 0:
			zz = np.linspace(_T[0], _T[len(_T)-1], 100)
			dd = np.linspace(BG, BG, 100)
				
			fig.append_trace(go.Scatter(x=zz, y=dd, marker={'color': 'red'}, mode = 'lines', fill=None,  name = ''), row = row, col = col)
			fig.append_trace(go.Scatter(x=_T, y=_G, marker={'color': 'purple'},  mode = 'lines', fill = 'tonexty', fillcolor = 'rgba(190,0,0,.15)', name = ''), row = row, col = col)
			
			text = canal + '!'
		
	fig.append_trace(
		go.Scatter(
		x=[Range-Range/30],
		y=[max(G_)],
       	mode="text",
      	text=[text],
       	textposition="bottom center",
		name = ''
		 ),
		row=row, col=col
		)


	fig.update_xaxes(row=row, col=col, range=[0, Range], nticks=20)
		
	return S
	


def get_S(G_, BG, G_er, canal):

	S = []
	for i in range(len(G_)):
		if G_er[i] != 0:
			S.append((G_[i]-BG)/G_er[i])
		else:
			S.append(-999)
	return S


def Range_(T, BT, T100):
	if len(BT) > 0:
		Range = float(T[len(T)-1])
	else:
		Range = float(T[len(T)-1])
	
	return Range




def read_cal(detector):	
	CAL=dict()
	with open('cal_' + detector + '_fixed_man.txt') as CAL_S:
		
		for _ in range(1):
			next(CAL_S)	

		for line in CAL_S:
			if line.isspace() == True:
				continue
			line = line.strip()
			line = line.split()
			ID = line[0]
			cal = float(line[11])
			CAL[ID] = cal
	
	return CAL	

def main():

	with open('TASK.txt', 'w') as TASK:
		wrr = "  {:10s} {:10s} {:10s} {:10s} {:10s} {:10s} {:10s} {:10s} {:10s} {:10s} {:10s} {:10s} {:10s} {:10s} {:10s} {:10s} {:10s} {:10s} {:10s} {:10s} {:10s} {:10s}\n".format(
		'ID', 'Data', 'Time', 'S_G2', 'S_G3', 'S_G4', 'S_G5', 'S_G6', 'S_G7', 'B_G2', 'P-value_G2', 'B_G3', 'P-value_G3', 'B_G4', 'P-value_G4', 'B_G5', 'P-value_G5', 'B_G6', 'P-value_G6', 'B_G7', 'P-value_G7', 'f')
		TASK.write(str(wrr))

	ID_ = []
	day_ = dict()
	time_ = dict()
	T100_ = dict()
	Det = dict()

	with open('ID.txt') as ID:
		for line in ID:
			line = line.strip()
			line = line.split()
			ID_.append(line[0])
			day_[line[0]] = line[1]
			time_[line[0]] = line[2]
			T100_[line[0]] = float(line[3])
			Det[line[0]] = line[4]
	
	CAL = read_cal('S1')
	CAL2 = read_cal('S2')
		
	CAL.update(CAL2)
	 
	tabl = pd.DataFrame(columns = ['ID', 'Data', 'Time', 'S_G2', 'S_G3', 'S_G4', 'S_G5', 'S_G6', 'S_G7', 'B_G2', 'P-value_G2', 'B_G3', 'P-value_G3', 'B_G4', 'P-value_G4', 'B_G5', 'P-value_G5', 'B_G6', 'P-value_G6', 'B_G7', 'P-value_G7', 'f'])
	
	#print('ID:')
	#ID__ = [input()]
	i = 0
	for ID in ID_:

		if ID not in CAL.keys():
			if Det[ID] == '1':
				CAL[ID] = 2.061
			if Det[ID] == '2':
				CAL[ID] = 1.637
		
		print(ID)

		TABL = open('pla\\' + ID + '.pla')
		PHA1, PHA2, PHA3 = read_pla(TABL)

		data = pd.DataFrame({
			'Emin': [0, 48.481, 199.015,  248, 730, 1866, 4590],
			'Emax': [48.481, 199.015, 757.318, 730, 1866, 4590, 10000]
			}, index=['G1', 'G2', 'G3', 'G4', 'G5', 'G6', 'G7'])

		#subplot_titles = ['G4', 'G6', 'G5',  'G7']
		fig = make_subplots(
	    rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.01) #subplot_titles=subplot_titles)
		
		T, t = get_time(PHA1)
		T, BT = get_time_BG(T, T100_[ID])

		Range = Range_(T, BT, T100_[ID])
		
		j = 0
		for canal in ['G2', 'G3', 'G4', 'G5', 'G6', 'G7']:
			G, G_er, f = get_canal (PHA1, PHA2, PHA3, data.loc[canal], CAL[ID], T, canal, ID)
			
			if G[0] == 'ER':
				if tabl.size/22 > i:
					tabl = tabl.drop([i])
				break

			G, BG, c = get_BG_and_chi2(G, T)	

			S = graf(G, T, G_er, BG, fig, canal, Range, T100_[ID])
			tabl.loc[i, 'S_'+canal] = max(S)
			tabl.loc[i, 'B_'+canal] = BG
			tabl.loc[i, 'P-value_'+canal] = c[1]
		
			j = j + 1
			
			ff = ' '
			if f == ' f':
				ff = ' f'
			if f == ' l':
				ff = ' l'
			if f == ' f_BG':
				ff = ' f_BG'		
	
		if G[0] != 'ER':

			fig.update_yaxes(nticks=5)
			fig.update_layout(title_text = 'ID' + ID +  ff)
			#py.offline.plot(fig, filename = 'test_vsplesk/' + ID + '.html')
			fig.write_image('images/' + ID + '.jpg', width=5, scale = 5)

			tabl.loc[i, 'ID'] = ID	
			tabl.loc[i, 'Data'] = day_[ID]	
			tabl.loc[i, 'Time'] = time_[ID]
			tabl.loc[i, 'f'] = ff

			with open('TASK.txt', 'a') as TASK:
				wr = "{:4s} {:10s} {:10.3f} {:10.3f} {:10.3f} {:10.3f} {:10.3f} {:10.3f} {:10.3f} {:10.3f} {:10.5f} {:10.3f} {:10.5f} {:10.3f} {:10.5f} {:10.3f} {:10.5f} {:10.3f} {:10.5f} {:10.3f} {:10.5f} {:10s}\n".format(
					tabl.loc[i, 'ID'], tabl.loc[i, 'Data'], float(tabl.loc[i, 'Time']), tabl.loc[i, 'S_G2'], tabl.loc[i, 'S_G3'], tabl.loc[i, 'S_G4'], tabl.loc[i, 'S_G5'], tabl.loc[i, 'S_G6'], tabl.loc[i, 'S_G7'], tabl.loc[i, 'B_G2'],  tabl.loc[i, 'P-value_G2'], tabl.loc[i, 'B_G3'],  tabl.loc[i, 'P-value_G3'], tabl.loc[i, 'B_G4'],  tabl.loc[i, 'P-value_G4'], tabl.loc[i, 'B_G5'],  tabl.loc[i, 'P-value_G5'], tabl.loc[i, 'B_G6'],  tabl.loc[i, 'P-value_G6'], tabl.loc[i, 'B_G7'],  tabl.loc[i, 'P-value_G7'], tabl.loc[i, 'f'])
				i = i + 1
				TASK.write(str(wr))

	
	#TASK2 = open('TASK2.txt', 'w')
	#pd.options.display.max_rows = 10000
	#pd.options.display.max_columns = 10000
	#print(tabl, file = TASK2, flush = True)


main()