import torch
from torch.utils.data import DataLoader


def ocr_inference(image_records, image_processor, model, device="cuda", batch_size=4):
    """
    Run batch OCR inference on a set of cropped cell images.

    Args:
        image_records: DataFrame or iterable containing image data with a 'Hash' 
                       column for deduplication and image pixel data.
        image_processor: TrOCR processor for image preprocessing.
        model: Fine-tuned VisionEncoderDecoderModel.
        device: Target device for inference (default: "cuda").
        batch_size: Number of images per inference batch (default: 4).

    Returns:
        dict: Mapping of {image_hash: predicted_text} for each input image.
    """
    if len(image_records) == 0:
        return {}

    inferred_text = {}
    dataset = PrepareInference(image_records, image_processor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            pixel_values = batch[0]["pixel_values"].to(device)
            hash_values = batch[1]["Hash"]

            max_new_tokens = int(pixel_values.size(-1) / 1.5)
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
