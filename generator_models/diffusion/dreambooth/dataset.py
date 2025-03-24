from torch.utils.data import Dataset
from torchvision import transforms
from pathlib import Path
from PIL import Image

from ..preprocess.change_image_size import preprocess_image_to_correct_size
from ..util import tokenize_prompt

class DreamBoothCustomDataset(Data):
    def __init__(
        self,
        dataset_dir,
        instance_prompt,
        class_prompt,
        tokenizer,
        tokenizer_max_length=100,
        size=512,
    ):
        self.size = size
        self.train_images = [img_file for img_file in Path(dataset_dir).glob("**/*") if img_file.name.endsWith(".png") or img_file.name.endsWith(".jpg")]
        self.tokenizer = tokenizer

        self._length = len(self.train_images)

        self.image_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

        self.instance_prompt = instance_prompt
        self.class_prompt = class_prompt

        self.tokenizer_max_length = tokenizer_max_length


    def __len__(self):
        return self._length

    def __getitem__(self, index):
        output = {}
        img = Image.open(self.train_images[index])
        img_preprocessed = preprocess_image_to_correct_size(img_preprocessed)

        if not img_preprocessed.mode == "RGB":
            img_preprocessed = img_preprocessed.convert("RGB")

        text_inputs = tokenize_prompt(self.tokenizer, self.instance_prompt, tokenizer_max_length=self.tokenizer_max_length)
        class_text_inputs = tokenize_prompt(self.tokenizer, self.class_prompt, tokenizer_max_length=self.tokenizer_max_length)

        output['instance_images'] = self.image_transforms(img_preprocessed)
        output['instance_prompt_ids'] = text_inputs.input_ids
        output['instance_attention_mask'] = text_inputs.attention_mask
        output['class_prompt_ids'] = class_text_inputs.input_ids
        output['class_attention_mask'] = class_text_inputs.attention_mask

        return output