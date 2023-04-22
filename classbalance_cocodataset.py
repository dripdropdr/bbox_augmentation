class limitedCocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, ann_file, transforms, return_masks):
        super(limitedCocoDetection, self).__init__(img_folder, ann_file)
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_masks)

        self.counts = 10000 #limitations of classes in each section
        self.label_dict = defaultdict(int) # for count datapoints of each class
        self.valid_indices = self.filter_indices()
        self.len = len(self.valid_indices)

    def filter_indices(self):
        valid_indices = []

        for idx in tqdm(range(super(limitedCocoDetection, self).__len__()), desc='limited dataset'):
            try:
                _, target = super(limitedCocoDetection, self).__getitem__(idx)
            except:
                print("Error idx: {}".format(idx))
                continue

            labels = np.array([t['category_id'] for t in target]).astype('int64')
            labels_unique = np.unique(labels)

            limit_class = [label for label in labels_unique 
                        if self.label_dict[label] > self.counts]

            if len(limit_class) == 0:
                valid_indices.append(idx)

                _bin = np.bincount(labels)
                for label in labels_unique:
                    self.label_dict[label] += _bin[label]

        return valid_indices

    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        """
        Output:
            - target: dict of multiple items
                - boxes: Tensor[num_box, 4]. \
                    Init type: x0,y0,x1,y1. unnormalized data.
                    Final type: cx,cy,w,h. normalized data. 
        """
        try:
            valid_idx = self.valid_indices[idx]
            img, target = super(limitedCocoDetection, self).__getitem__(valid_idx)
        except:
            print("Error idx: {}".format(valid_idx))
            idx += 1
            valid_idx = self.valid_indices[idx]
            img, target = super(limitedCocoDetection, self).__getitem__(valid_idx)
        image_id = self.ids[valid_idx]
        target = {'image_id': image_id, 'annotations': target}
            
        img, target = self.prepare(img, target)
        
        if self._transforms is not None:
            img, target = self._transforms(img, target)

        return img, target
