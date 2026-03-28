import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader


class PrepareInference(Dataset):
    """PyTorch Dataset for batch OCR inference on cropped cell images.

    Args:
        df: DataFrame with a 'Localiser' column (image file paths) and a
            'Hash' column (unique identifier per image for deduplication).
        processor: TrOCRProcessor for image preprocessing.
    """

    def __init__(self, df, processor):
        self.df = df.reset_index(drop=True)
        self.processor = processor

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_path = self.df['Localiser'][idx]
        h_val = self.df['Hash'][idx]
        image = Image.open(image_path).convert("RGB")
        pixel_values = self.processor(image, return_tensors="pt").pixel_values
        return {"pixel_values": pixel_values.squeeze(), "Hash": h_val}


def ocr_inference(image_records, image_processor, model, device="cuda", batch_size=4, max_new_tokens=128):
    """
    Run batch OCR inference.

    Args:
        image_records: DataFrame with 'Localiser' (image file paths) and 'Hash'
                       (unique identifier) columns.
        image_processor: TrOCRProcessor for image preprocessing and decoding.
        model: Fine-tuned VisionEncoderDecoderModel.
        device: Target device for inference (default: "cuda").
        batch_size: Number of images per inference batch (default: 4).
        max_new_tokens: Maximum tokens to generate per image (default: 128).

    Returns:
        dict: Mapping of {image_hash: predicted_text} for each input image.
    """
    if len(image_records) == 0:
        return {}

    model = model.to(device)
    model.eval()

    inferred_text = {}
    dataset = PrepareInference(image_records, image_processor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    with torch.no_grad():
        for batch in dataloader:
            pixel_values = batch["pixel_values"].to(device)
            hash_values = batch["Hash"]

            generated_ids = model.generate(
                pixel_values,
                max_new_tokens=max_new_tokens,
            )
            generated_texts = image_processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )

            for h_val, text in zip(hash_values, generated_texts):
                inferred_text[h_val] = text

    return inferred_text
