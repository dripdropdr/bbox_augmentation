class BoundingBoxAug(object):
    def __init__(self, p=0.5):
        self.p = p
        self.thr = 0.5

    def hflip(self, img, target):
        img_arr = np.array(img)
        
        if "boxes" in target:
            boxes = target["boxes"]
            
            for box in boxes:
                if random.random() < self.p:
                    box_coord = list(map(int, box.tolist()))
                    box_arr = img_arr[box_coord[1]:box_coord[3], box_coord[0]:box_coord[2]]
                    box_arr_h = box_arr[::-1, :]
                    img_arr[box_coord[1]:box_coord[3], box_coord[0]:box_coord[2]] = box_arr_h
            
            boxflipped_img = PIL.Image.fromarray(img_arr)          
            return boxflipped_img

    def vflip(self, img, target):
        img_arr = np.array(img)
        
        if "boxes" in target:
            boxes = target["boxes"]
            
            for box in boxes:
                if random.random() < self.p:
                    box_coord = list(map(int, box.tolist()))
                    box_arr = img_arr[box_coord[1]:box_coord[3], box_coord[0]:box_coord[2]]
                    box_arr_h = box_arr[::-1, :]
                    img_arr[box_coord[1]:box_coord[3], box_coord[0]:box_coord[2]] = box_arr_h
            
            boxflipped_img = PIL.Image.fromarray(img_arr)          
            return boxflipped_img

    def __call__(self, img, target):

        label_tensor = target['labels']
        label_tensor = label_tensor.to('cpu')
        label_tensor_unique = torch.unique(label_tensor)

        # app_cls = [22, 26, 34, 38, 46, 47, 49, 51, 57, 28, 32]
        # check_list = [idx.item() for idx in label_tensor_unique if idx.item() in app_cls] #pz
        if random.random() > self.thr: #and len(check_list) > 1:
            img = self.hflip(img, target)
            img = self.vflip(img, target)

        return img, target
