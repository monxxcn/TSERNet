from skimage import io
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
from PIL import Image
import glob
from data_loader import RescaleT, ToTensorLab, SalObjDataset
from model import TSERNet


def normPRED(d):
	ma = torch.max(d)
	mi = torch.min(d)

	dn = (d-mi)/(ma-mi)

	return dn


def save_output(image_name, pred, d_dir):

	predict = pred
	predict = predict.squeeze()
	predict_np = predict.cpu().data.numpy()

	im = Image.fromarray(predict_np*255).convert('RGB')
	img_name = image_name.split("/")[-1]
	image = io.imread(image_name)
	imo = im.resize((image.shape[1],image.shape[0]),resample=Image.BILINEAR)

	pb_np = np.array(imo)

	aaa = img_name.split(".")
	bbb = aaa[0:-1]
	imidx = bbb[0]
	for i in range(1, len(bbb)):
		imidx = imidx + "." + bbb[i]
	name = imidx.split("\\")[-1]
	imo.save(d_dir + name + '.png')


if __name__ == '__main__':
	# --------- 1. get image path and name ---------
	
	image_dir = ''
	prediction_dir = ''
	model_dir = ''
	img_ext = ''
	
	img_name_list = glob.glob(image_dir + img_ext)
	
	# --------- 2. dataloader ---------
	#1. dataload
	test_salobj_dataset = SalObjDataset(img_name_list=img_name_list, lbl_name_list=[], supervision_name_list=[], transform=transforms.Compose(
		[RescaleT(256), ToTensorLab(flag=0)]))
	test_salobj_dataloader = DataLoader(test_salobj_dataset, batch_size=1, shuffle=False, num_workers=1)
	
	# --------- 3. model define ---------
	print("...load TSERNet...")
	net = TSERNet(3, 1)
	net.load_state_dict(torch.load(model_dir))
	if torch.cuda.is_available():
		net.cuda()
	net.eval()
	
	# --------- 4. inference for each image ---------
	for i_test, data_test in enumerate(test_salobj_dataloader):
	
		print("inferencing:", img_name_list[i_test].split("/")[-1])
	
		inputs_test = data_test['image']
		inputs_test = inputs_test.type(torch.FloatTensor)
	
		if torch.cuda.is_available():
			inputs_test = Variable(inputs_test.cuda())
		else:
			inputs_test = Variable(inputs_test)
	
		dout, d1, d2, d3, d4, d5, d6, db, d_edge, d1_ref, d2_ref, d3_ref, d4_ref = net(inputs_test)
	
		# normalization
		pred = dout[:, 0, :, :]
		pred = normPRED(pred)
	
		# save results to test_results folder
		save_output(img_name_list[i_test], pred, prediction_dir)
	
		del dout, d1, d2, d3, d4, d5, d6, db, d_edge, d1_ref, d2_ref, d3_ref, d4_ref
