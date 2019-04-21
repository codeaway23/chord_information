import numpy as np
from scipy.signal import get_window
from scipy.io.wavfile import read
from matplotlib import pyplot as plt
import stft as STFT
import os

def chromaRep(inputFile, window, M, N, H) :
	"""
	Convert the short time fourier transform into a chroma
	representation. Without losing time domain information due to the use of 
	STFT, plotting the magnitude spectrum using chromaRep gives us the chord
	information.
	"""
	fs,x = read(inputFile)                            # Read input file.     
	w = get_window(window, M)                         # Get window function.
	
	notebins = np.zeros(100)                           # Create notebins array of 88 notes.
	notebins[0] = 27.50
	for i in range(1,len(notebins)):
	        notebins[i] = notebins[0]*np.power(2,i/12.0)
	notebins = np.log2(notebins)

	afreq = np.zeros(N)                               # Create frequency values array as on STFT.
	afreq[0] = 0.0
	for i in range(N-1):
	        afreq[i+1] = fs*(i+1)/N
	afreq = np.log2(afreq)                            # Take log of frequency values. 

	xmX,xpX = STFT.stftAnal(x, w, N, H)               # Take Short Time Fourier Transform.(STFT)

	rows,cols = np.shape(xmX)                         # Make dummySTFT matrix with 100 freq bins and 12.
	dummymX = np.zeros((rows,100))
	chrommX = np.zeros((rows,12))

	for i in range(len(notebins)):                    # Convert STFT in the 100 note format.
		for j in range(len(afreq)):
			if (afreq[j]<=notebins[i]+1/24.0 and afreq[j]>notebins[i]-1/24.0):
				dummymX[:,i] = dummymX[:,i] + xmX[:,j]
		if i < 12:
			chrommX[:,i] = dummymX[:,i]
		else:
			chrommX[:,i%12-1] = chrommX[:,i%12-1] + dummymX[:,i]

	return chrommX,dummymX,xmX


fs,x = read('eqt-chromo-sc.wav')
w = get_window('blackman',1024)
xmx,xpx = STFT.stftAnal(x,w,4096,250)

chroma,keybrd,spectrum = chromaRep('eqt-chromo-sc.wav', 'blackmanharris', 512, 4096, 100)

cmap = plt.get_cmap('PiYG')

plt.pcolormesh(np.transpose(chroma), cmap=cmap)
plt.ylabel('note values')
plt.xlabel('time')
plt.title('Chroma Representation')
plt.show()

plt.pcolormesh(np.transpose(keybrd), cmap=cmap)
plt.ylabel('note values')
plt.xlabel('time')
plt.title('Note value Representation')
plt.show()
