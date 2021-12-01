import numpy as np

import matplotlib
import matplotlib.pyplot as plt



def plot_flow(img, coords, flows, flow_res=None) :
	if flow_res is None  :
		flow_res = img.shape
	scale_x = img.shape[0] / flow_res[0]
	scale_y = img.shape[1] / flow_res[1]
	
	plt.figure(figsize=(16, 12))
	
	plt.imshow(img, cmap='gray')
	
	for i in range(0, len(coords), 10) :
		plt.plot([coords[i, 0] * scale_x, coords[i, 0] * scale_x + flows[i, 0] * scale_x],
			     [coords[i, 1] * scale_y, coords[i, 1] * scale_y + flows[i, 1] * scale_y])
