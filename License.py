import cv2
import glob
import easyocr
import numpy as np
import matplotlib.pyplot as plt

path = glob.glob("C:/Users/Piyush Mishra/Desktop/job/InputImage/*.jpg")
ct = 1
for n in path:
	read = easyocr.Reader(['en'])
	op = read.readtext(n)
	img_1 = cv2.imread(n)
	img_1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2RGB)
	plt.imshow(img_1)

	def overlay_ocr_text(n, save_name): #loads an image, recognizes text, and overlay the text on the image.
		img = cv2.imread(n) # loads image
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
		dpi = 100
		fig_width, fig_height = int(img.shape[0]/dpi), int(img.shape[1]/dpi)
		
		for (block, text, prob) in op:
			
			print(f'Detected text: {text} (Probability: {prob:.2f})') # display for own understanding for 

			# get top-left and bottom-right bbox vertices
			(top_left, top_right, bottom_right, bottom_left) = block
			top_left = (int(top_left[0]), int(top_left[1]))
			bottom_right = (int(bottom_right[0]), int(bottom_right[1]))

			img1 = cv2.rectangle(img=img, pt1=top_left, pt2=bottom_right, color=(255, 0, 0), thickness=1)# create a rectangle for block display

			if top_left[1] - 20 <0:
				img2 = cv2.putText(img=img, text=text, org=(bottom_left[0], bottom_left[1] + 10), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 255, 0), thickness=2)
			else:
				img2 = cv2.putText(img=img, text=text, org=(top_left[0], top_left[1] - 10), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 255, 0), thickness=2)

		# show and save image
		global ct
		plt.figure()
		plt.imshow(img2)
		plt.savefig(f'./LicenseOutputs/{save_name}_overlay.jpg', bbox_inches='tight')
		ct = ct+1
	overlay_ocr_text(n, str(ct))	# Displaying the huge amount