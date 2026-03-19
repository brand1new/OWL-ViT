import torch
from PIL import Image, ImageOps
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor


class QWenVLWrapper:
    def __init__(self, model_name, device="cuda"):
        self.device = device

        self.model, self.processor = self.load_model_from_pretreain(model_name)


    def load_model_from_pretreain(self, model_name):
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name, torch_dtype="auto", device_map=self.device
        )
        model.eval()

        processor = AutoProcessor.from_pretrained(model_name, use_fast=True, padding_side='left')
        return model, processor


    def generate_text_from_image(self, pil_image_list, prompt):
        output_text_list = [""] * len(pil_image_list)
        pil_image_checked_list, idx_checked_list = self.preprocess_image_list(pil_image_list)
        if len(pil_image_checked_list) == 0:
            return output_text_list

        messages = [
            {
                "role": "user",
                "content": [{"type": "image"}, {"type": "text", "text": prompt}],
            }
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[text] * len(pil_image_checked_list),
            images=pil_image_checked_list,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.device)

        with torch.inference_mode():
            generated_ids = self.model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        model_out_text_list = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        for idx, output_text in zip(idx_checked_list, model_out_text_list):
            output_text_list[idx] = output_text

        return output_text_list


    def generate_text_from_image_with_multi_messages(self, pil_image_list, messages_list, max_new_tokens=4096, temperature=None):
        output_text_list = [""] * len(pil_image_list)
        pil_image_checked_list, idx_checked_list = self.preprocess_image_list(pil_image_list)
        if len(pil_image_checked_list) == 0:
            return output_text_list

        text_list = []
        for messages in messages_list:
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            text_list.append(text)

        inputs = self.processor(
            text=text_list,
            images=pil_image_checked_list,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.device)

        with torch.inference_mode():
            if temperature is None:
                generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
            else:
                if temperature == 0.0:
                    generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, temperature=None)
                else:
                    generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True, temperature=temperature)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        model_out_text_list = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        for idx, output_text in zip(idx_checked_list, model_out_text_list):
            output_text_list[idx] = output_text

        return output_text_list


    def generate_messages_from_history(self, historys_list):
        messages_list = [[] for _ in range(len(historys_list))]
        for idx, historys in enumerate(historys_list):
            for history in historys:
                messages_list[idx].append({
                    "role": history["role"],
                    "content": history["content"],
                })
        return messages_list


    def preprocess_image_list(self, image_list, fill_color=(0, 0, 0)):
        MIN_QWENVL_IMG_H_W = 28
        MAX_QWENVL_IMG_H_W = 640

        idx_checked_list = []
        processed_images = []
        for idx, image in enumerate(image_list):
            image_h, image_w = image.size
            if image_h == 0 or image_w == 0:
                continue

            target_size = max(image_h, image_w)
            image = ImageOps.pad(
                image, 
                size=(target_size, target_size),
                color=fill_color,
                centering=(0.5, 0.5))

            if target_size > MAX_QWENVL_IMG_H_W:
                target_size = MAX_QWENVL_IMG_H_W
                image = image.resize((target_size, target_size), Image.LANCZOS)
            if target_size < MIN_QWENVL_IMG_H_W:
                target_size = MIN_QWENVL_IMG_H_W
                image = image.resize((target_size, target_size), Image.LANCZOS)

            processed_images.append(image)
            idx_checked_list.append(idx)

        return processed_images, idx_checked_list

    
    def check_image_size(self, image_list):
        MAX_QWENVL_IMG_SIZE = 1028*28*28
        MIN_QWENVL_IMG_H_W = 28

        image_checked_list = []
        idx_checked_list = []
        for idx, image in enumerate(image_list):
            image_h, image_w = image.size
            if image_h == 0 or image_w == 0:
                continue

            image_size = image_h * image_w
            if image_size > MAX_QWENVL_IMG_SIZE:
                if image_h <= MIN_QWENVL_IMG_H_W or image_w <= MIN_QWENVL_IMG_H_W:
                    continue
                ratio = round(MAX_QWENVL_IMG_SIZE / image_size, 2)
                image = image.resize((int(image_h*ratio), int(image_w*ratio)), Image.LANCZOS)

            if image_w <= MIN_QWENVL_IMG_H_W or image_h <= MIN_QWENVL_IMG_H_W:
                if image_h < image_w:
                    h_w_ratio = image_w / image_h
                    image = image.resize((MIN_QWENVL_IMG_H_W+1, int(h_w_ratio * (MIN_QWENVL_IMG_H_W+1))), Image.LANCZOS)
                else:
                    h_w_ratio = image_h / image_w
                    image = image.resize((int(h_w_ratio * (MIN_QWENVL_IMG_H_W+1)), MIN_QWENVL_IMG_H_W+1), Image.LANCZOS)

            image_checked_list.append(image)
            idx_checked_list.append(idx)

        return image_checked_list, idx_checked_list
    
