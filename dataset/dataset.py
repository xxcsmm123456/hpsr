from torch.utils.data.dataset import Dataset

class HPSRdataset(Dataset):
    def __init__(self, dataset_dir,datamodel, upscale_factor, input_transform=None, target_transform=None):
        super(HPSRdataset, self).__init__()
        self.dataset_dir = dataset_dir
        self.upscale_factor = upscale_factor
        self.datamodel = datamodel
        
        with open(self.dataset_dir+'/'+self.datamodel+'data5km_1105.txt','r') as f:
            imagenamelist=[line.split('\n')[0] for line in f.readlines()]
        with open(self.dataset_dir+'/'+self.datamodel+'data1km_1105.txt','r') as f:
            targetnamelist=[line.split('\n')[0] for line in f.readlines()]
        with open(self.dataset_dir+'/'+self.datamodel+'datadem_1105.txt','r') as f:
            demnamelist=[line.split('\n')[0] for line in f.readlines()]
        with open(self.dataset_dir+'/'+self.datamodel+'data1kmCLCD_1105.txt','r') as f:
            CLCDnamelist=[line.split('\n')[0] for line in f.readlines()]
        self.imagenamelist = imagenamelist 
        self.targetnamelist = targetnamelist  
        self.demnamelist = demnamelist
        self.CLCDnamelist = CLCDnamelist
        self.input_transform = input_transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        image = (np.load(self.imagenamelist[index]).astype(np.float32))[:,:,np.newaxis]
        target = (np.load(self.targetnamelist[index]).astype(np.float32))[:,:,np.newaxis]
        dem = (np.load(self.demnamelist[index]).astype(np.float32))[:,:,np.newaxis]
        CLCD = (np.load(self.CLCDnamelist[index]).astype(np.float32))[:,:,np.newaxis]
        if self.input_transform:
            image = self.input_transform(image)
        if self.target_transform:
            dem = self.target_transform(dem)
        if self.target_transform:
            CLCD = self.target_transform(CLCD)
        if self.target_transform:
            target = self.target_transform(target)
        return image,dem,CLCD,target

    def __len__(self):
        return len(self.imagenamelist)