"""CLIPSeg wrapper with frozen CLIP encoders for fine-tuning."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor

from config import CLIPSEG_CHECKPOINT, IMAGE_SIZE


class CLIPSegFinetune(nn.Module):
    """CLIPSeg model with frozen CLIP backbone — only decoder is trainable."""

    def __init__(self, checkpoint: str = CLIPSEG_CHECKPOINT):
        super().__init__()
        self.model = CLIPSegForImageSegmentation.from_pretrained(checkpoint)
        self.processor = CLIPSegProcessor.from_pretrained(checkpoint)

        # Freeze CLIP vision + text encoders
        for name, param in self.model.named_parameters():
            if "clip" in name.lower():
                param.requires_grad = False

        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        print(f"[CLIPSeg] Trainable: {trainable:,} / {total:,} params "
              f"({100 * trainable / total:.1f}%)")

    def forward(self, images: torch.Tensor, prompts: list[str]) -> torch.Tensor:
        """
        Args:
            images: (B, 3, H, W) tensor, already normalized with CLIP stats
            prompts: list of B text prompts

        Returns:
            logits: (B, H, W) raw logits (apply sigmoid for probabilities)
        """
        device = images.device

        # Tokenize text prompts
        text_inputs = self.processor.tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True
        )
        input_ids = text_inputs["input_ids"].to(device)
        attention_mask = text_inputs["attention_mask"].to(device)

        # CLIPSeg expects pixel_values — our images are already preprocessed
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=images,
        )

        # outputs.logits shape: (B, H_out, W_out) — resize to target
        logits = outputs.logits  # (B, H_out, W_out)

        if logits.dim() == 2:
            # Single sample edge case
            logits = logits.unsqueeze(0)

        # Resize to match target mask size
        logits = F.interpolate(
            logits.unsqueeze(1),  # (B, 1, H_out, W_out)
            size=(IMAGE_SIZE, IMAGE_SIZE),
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)  # (B, H, W)

        return logits

    def trainable_parameters(self):
        return [p for p in self.model.parameters() if p.requires_grad]
