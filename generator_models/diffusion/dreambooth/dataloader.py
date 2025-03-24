import torch
from .dataset import DreamBoothCustomDataset

def collate_fn(batched_data):
    assert "instance_attention_mask" in batched_data[0], "required key 'instance_attention_mask' not in data: {data}"

    input_prompt_ids = [data['instance_prompt_ids'] for data in batched_data]
    input_images = [data['instance_images'] for data in batched_data]

    attention_mask = [data['instance_attention_mask'] for data in batched_data]

    input_prompt_ids = torch.cat(input_prompt_ids, dim=0)
    input_images = torch.stack(input_images).to('cuda')

    batch = {
        'input_ids': input_prompt_ids,
        'images': input_images,
        'attention_mask': attention_mask
    }

    return batch

def create_dreambooth_dataloader(data_dir, instance_prompt, class_prompt, tokenizer, batch_size):
    dreambooth_dataset = DreamBoothCustomDataset(
        dataset_dir=data_dir,
        instance_prompt=instance_prompt,
        class_prompt=class_prompt,
        tokenizer=tokenizer
    )

    return torch.utils.data.DataLoader(
        dataset=dreambooth_dataset,
        batch_size=batch_size,
        shuffle=Ture,
        collate_fn=lambda batched_data: collate_fn(batched_data)
    )
    