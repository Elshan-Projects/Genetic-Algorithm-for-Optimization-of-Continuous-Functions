from tkinter import *
import matplotlib.pyplot as plt


# Initialization Window

window = Tk()

window.title("Genetic Optimization of Continuous Functions")
window.iconbitmap("GAE.ico")
window.resizable(False, False)
window.geometry('950x380')


lbl1 = Label(window, text="Function: ")
lbl1.grid(column=0, row=0, columnspan = 2)


e1_var=StringVar()
e1 = Entry(window ,textvariable = e1_var, width=140)
e1.insert(END, '-1*x[0]**2-100')
e1.grid(column=2, row=0, columnspan = 6, pady=(20, 10), padx=(10, 10))


v2 = IntVar()
s2 = Scale( window, variable = v2, from_ = 50, to = 200, tickinterval=10, orient = HORIZONTAL, length=900, label = "Population Size :")
s2.grid(column=0, row=1, columnspan = 8, padx=(20, 20), pady=(30, 30))


lbl2 = Label(window, text="Alpha: ")
lbl2.grid(column=0, row=2, columnspan = 1)


e2_var=StringVar()
e2 = Entry(window ,textvariable = e2_var, width=15)
e2.insert(END, '0.5')
e2.grid(column=1, row=2, columnspan = 2)


lbl3 = Label(window, text="Deviance: ")
lbl3.grid(column=3, row=2, columnspan = 1)


e3_var=StringVar()
e3 = Entry(window ,textvariable = e3_var, width=15)
e3.insert(END, '2.5')
e3.grid(column=4, row=2, columnspan = 2)


lbl4 = Label(window, text="Mutation Rate: ")
lbl4.grid(column=6, row=2, columnspan = 1)


e4_var=StringVar()
e4 = Entry(window ,textvariable = e4_var, width=15)
e4.insert(END, '0.0001')
e4.grid(column=7, row=2, columnspan = 1)


v3 = IntVar()
s3 = Scale( window, variable = v3, from_ = 100000, to = 1000000, tickinterval=100000, orient = HORIZONTAL, length=900, label = "Number of Iterations :")
s3.grid(column=0, row=3, columnspan = 8,padx=(20, 20), pady=(10, 10))



def GetData():
    global funksiya
    funksiya = str(e1_var.get())

    global pop_size
    pop_size = int(v2.get())

    global al_CO
    al_CO = float(e2_var.get())

    global dev
    dev = float(e3_var.get())

    global mut1
    mut1 = float(e4_var.get())

    global n_it
    n_it = int(v3.get())

    window.destroy()


    
B = Button(window, text ="R U N", command = GetData, width = 30, height=3)
B.grid(column=0, row=7, columnspan = 8,pady=5)

window.mainloop()



#__________________________________________________________________________

from math import *
import random
import numpy as np
import itertools




# Dimension detection

inpp = funksiya

if (inpp.find('[9]') != -1):
	dim111 = 10

elif (inpp.find('[8]') != -1):
	dim111 = 9

elif (inpp.find('[7]') != -1):
	dim111 = 8

elif (inpp.find('[6]') != -1):
	dim111 = 7

elif (inpp.find('[5]') != -1):
	dim111 = 6

elif (inpp.find('[4]') != -1):
	dim111 = 5

elif (inpp.find('[3]') != -1):
	dim111 = 4

elif (inpp.find('[2]') != -1):
	dim111 = 3

elif (inpp.find('[1]') != -1):
	dim111 = 2
	
else:
	dim111 = 1




# ___________________________________________

# Parameters
dim = dim111

n_iter = n_it

pop = pop_size

funct = funksiya

alpha = al_CO

mut = mut1
# If mut is 1 then mutation doesnt happen

mut_dev = dev
#______________________________________





# Genetic Algorthm Functions


# Evaluation of of a single function
def evaluate(func,a):
	x=a
	res1 = eval(func)
	return res1



# List-Fitness-Population
def LFP(popp):
	res2 = [None]*pop
	for k in range(pop):
		res2[k] = evaluate(funct,popp[k])
	return res2



# Selection Function
def selection(popp):
	LFPr = LFP(popp)

	array = np.array(LFPr)

	temp = array.argsort()

	ranks = np.empty_like(temp)

	ranks[temp] = np.arange(len(array))

	ranks = ranks + 1

	rank = ranks.tolist()

	summa = sum(rank)


	fit = [None]*pop
	for l in range(pop):
		fit[l] = rank[l]/summa
	# random 2*pop sample according to "fit"s
	popp_c= [*range(pop)]
	sampling = np.random.choice(popp_c, 2*pop, p=fit)
	sampling_l = sampling.tolist()
	
	res_f = [[None for j in range(2)] for i in range(pop)]
	for zxc in range(pop):
		rr1=random.choice(sampling_l)
		ii1 = sampling_l.index(rr1)
		temm = sampling_l.pop(ii1)
		res_f[zxc][0] = rr1
    
		rr2=random.choice(sampling_l)
		ii2 = sampling_l.index(rr2)
		temm = sampling_l.pop(ii2)
		res_f[zxc][1] = rr2

	res_ff = [ [ [ None for qqqq in range(dim) ] for jjjj in range(2) ] for iiii in range(pop) ]
	for aa in range(pop):
		for bb in range(2):
			res_ff[aa][bb]=popp[ res_f[aa][bb] ]
	return res_ff



def crossover(parents):
	b1=[None]*pop
	b2=[None]*pop
	kids_ND = [None]*pop
	for m in range(pop):
		b1[m] = np.array( parents[m][0] ) - alpha*( np.array( parents[m][1] )-np.array( parents[m][0] ) )
		b2[m] = np.array( parents[m][1] ) + alpha*( np.array( parents[m][1] )-np.array( parents[m][0] ) )
	for u in range(pop):
		kids_ND[u] = np.random.uniform(b1[u],b2[u])
	res4 = kids_ND
	return res4



def mutation(kids):
	for t in range(pop):
		rand_num = random.random()
		if (rand_num > mut):
			GAUSS = random.gauss(0,mut_dev)
			kids[t] = np.array(kids[t]) + GAUSS
		res5 = kids
	return res5



def maxpo(l1):
    maxvalue = max(l1)
    maxpos = l1.index(maxvalue)
    return maxpos


def minpo(l2):
    minvalue = min(l2)
    minpos = l2.index(minvalue)
    return minpos


def elitism(p_o, p_n):
	LFP_o = LFP(p_o)
	i_o = maxpo(LFP_o)
	elite = p_o[i_o]

	LFP_n = LFP(p_n)
	i_n = minpo(LFP_n)
	p_n[i_n] = elite
	
	return p_n


track_i=[]
track_m=[]


def GA_loop():
	BBB["state"] = "disabled"
	# Initial population
	popu= [None]*pop
	for j in range(pop):
		popu[j] = [random.randrange(1, 500, 1) for z in range(dim)]


	
	global i
	i = 0
	max_old=[None]*dim
	while (i<n_iter):
		popu_old = popu
		parents1 = selection(popu)
		kids1 = crossover(parents1)
		popu_new = mutation(kids1)
		popu = elitism(popu_old, popu_new)


		tops = LFP(popu)
		top_pos = maxpo(tops)
		maximizer = popu[top_pos]
		current_top = evaluate(funct,popu[top_pos])


		maximizer111 = np.array(maximizer)
		max_old111 = np.array(max_old)

		if not all(maximizer111==max_old111):
			ZZZ =  "At the iteration " + str(i) + "\n" + "The current maximizer is: \n" + str(maximizer) + "\n" + "The current maximum is: \n" + str(current_top) + "\n" + "\n"
			textbox.insert(END,ZZZ)	
			max_old = maximizer
			if (i!=0):
				track_i.append(i)
				track_m.append(current_top)


		maximizer1 = maximizer111.round(decimals=3)
		current_top1 = round(current_top, 3)

		GHG = "i = " + str(i) + " x* = " + str(maximizer1)
		HGH = "f(x) = " + str(current_top1)
		CC2.config(text=GHG)
		CC4.config(text=HGH)


		i=i+1

import threading

def GA_fun():
	GA_thread = threading.Thread(target=GA_loop)
	GA_thread.start()


#_______________________________________________________________________________





# Status Window


root = Tk()
root.title("Genetic Optimization of Continuous Functions")
root.geometry('800x600')
root.iconbitmap("GAE.ico")
root.resizable(False, False)


scrollbar = Scrollbar(root)
scrollbar.pack(side=RIGHT, fill=Y)
textbox = Text(root, width=100)
textbox.pack()

textbox.config(yscrollcommand=scrollbar.set)
scrollbar.config(command=textbox.yview)

HP1 = "Dim=" + str(dim) + " f(x)=" + str(funct)
H1 = Label(root, text = HP1, font=("Arial", 9))
H1.pack()

HP2 = "Pop=" + str(pop) + " N_iter=" + str(n_iter) + "Alpha=" + str(alpha) + " Deviance=" + str(mut_dev) + " MutRate=" + str(mut)
H2 = Label(root, text = HP2, font=("Arial", 8))
H2.pack()


CC1 = Label(root, text = "The current maximizer ", font=("Arial", 11))
CC1.pack()

CC2 = Label(root, text = "__________________", font=("Arial", 11))
CC2.pack()

CC3 = Label(root, text = "The current maximum ", font=("Arial", 11))
CC3.pack()

CC4 = Label(root, text = "__________________", font=("Arial", 11))
CC4.pack()

BBB = Button(root, text ="S T A R T", command = GA_fun)
BBB.pack(padx=5)



def PLOTT():
	plt.plot(track_i, track_m)

	plt.title("GA PROGRESS GRAPH")

	plt.xlabel("Iteration")

	plt.ylabel("Maximum")

	plt.show(block=False)

www = Button(root, text ="P L O T   T H E   C U R R E N T   P R O G R E S S", command = PLOTT)
www.pack(padx=5)


root.mainloop()



