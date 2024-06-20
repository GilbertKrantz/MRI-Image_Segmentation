import skimage.draw
import tifffile
import shutil
import json
import os

class JSON_handler:
    def __init__(self, image_path, json_path, target_image_path, target_mask_path):
        self.image_path = image_path
        self.json_path = json_path
        self.target_image_path = target_image_path
        self.target_mask_path = target_mask_path
        
    @staticmethod
    def create_mask(image_info, annotations, output_folder, max_print=3):
        mask_np = np.zeros((image_info['height'], image_info['width']), dtype=np.uint8)
        object_number = 1
        printed_masks = 0  # Count the number of printed masks
        for ann in annotations:
            if ann['image_id'] == image_info['id']:
                for seg_idx, seg in enumerate(ann['segmentation']):
                    rr, cc = skimage.draw.polygon(seg[1::2], seg[0::2], mask_np.shape)
                    
                    seg_mask = np.zeros_like(mask_np, dtype=np.uint8)
                    seg_mask[rr, cc] = 255 
                    mask_path = os.path.join(output_folder, f"{image_info['file_name'].replace('.jpg', '')}_seg_{seg_idx}.tif")
                    tifffile.imwrite(mask_path, seg_mask)
                    printed_masks += 1
                    if printed_masks >= max_print:
                        return 
    
    def json_to_mask(self):
        # Load COCO JSON annotations
        with open(self.json_path , 'r') as f:
            data = json.load(f)

        images = data['images']
        annotations = data['annotations']

        # Ensure the output directories exist
        if not os.path.exists(self.target_mask_path):
            os.makedirs(self.target_mask_path)
        if not os.path.exists(self.target_image_path):
            os.makedirs(self.target_image_path)

        for img in images:
            self.create_mask(img, annotations, self.target_mask_path)

            original_image_path = os.path.join(self.image_path, img['file_name'])
            new_image_path = os.path.join(self.target_image_path, os.path.basename(original_image_path))
            shutil.copy2(original_image_path, new_image_path)