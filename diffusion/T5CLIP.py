import torch
from transformers import AutoTokenizer, CLIPTextModelWithProjection, T5EncoderModel, logging

class ConditioningEncoders():
    def __init__(self, device, seq_max_length: int = 64, torch_dtype = torch.float32, debugging: bool = True):
        self.device = device
        self.seq_max_length = seq_max_length
        self.debugging = debugging
        
        logging.set_verbosity_error()
        self.t5_tokenizer = AutoTokenizer.from_pretrained("google/t5-v1_1-small")
        if debugging is False:
            self.clip_l_tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        self.clip_s_tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")

        self.t5_encoder = T5EncoderModel.from_pretrained("google/t5-v1_1-small", weights_only=True, torch_dtype=torch_dtype).to(device).eval()
        if debugging is False:
            self.clip_l_encoder = CLIPTextModelWithProjection.from_pretrained("openai/clip-vit-large-patch14", weights_only=True, torch_dtype=torch_dtype).to(device).eval()

        self.clip_s_encoder = CLIPTextModelWithProjection.from_pretrained("openai/clip-vit-base-patch32", weights_only=True, torch_dtype=torch_dtype).to(device).eval()
        
        #----
        self.t5_dim = self.t5_encoder.config.d_model
        self.pooled_dim = self.clip_s_encoder.config.projection_dim
        if debugging is False:
            self.pooled_dim += self.clip_l_encoder.config.projection_dim
    
    @torch.no_grad()
    def encode(self, captions: list) -> tuple[torch.Tensor]:
        t5_tokens = self.t5_tokenizer(captions, padding="max_length", max_length=self.seq_max_length, truncation=True, return_tensors="pt").to(self.device)
        if self.debugging is False:
            clip_l_tokens = self.clip_l_tokenizer(captions, padding="max_length", max_length=self.seq_max_length, truncation=True, return_tensors="pt").to(self.device)
        clip_s_tokens = self.clip_s_tokenizer(captions, padding="max_length", max_length=self.seq_max_length, truncation=True, return_tensors="pt").to(self.device)
    
        if self.debugging is False:
            pooled_l = self.clip_l_encoder(clip_l_tokens.input_ids, output_hidden_states=False, attention_mask=clip_l_tokens.attention_mask).text_embeds
        pooled_s = self.clip_s_encoder(clip_s_tokens.input_ids, output_hidden_states=True, attention_mask=clip_s_tokens.attention_mask).text_embeds
        # In SD3, t5_seq is suppose to be concatted and padded with the clip hidden_states. Seen as optional in FLUX.1
        t5_seq = self.t5_encoder(t5_tokens.input_ids, output_hidden_states=False, attention_mask=t5_tokens.attention_mask).last_hidden_state
        
        if self.debugging is False:
            pooled_cat = torch.cat([pooled_l, pooled_s], dim = -1)
        else:
            pooled_cat = pooled_s
            
        return t5_seq, pooled_cat

if __name__ == '__main__':
    captions = ["Are you a smart fella or a fart smella?",
                "He was looking at me like lobsters were crawling out of my ears!",
                "I am in fact about to explode."
                ]

    T5CLIP = ConditioningEncoders(device="cuda")
    seq, pool = T5CLIP.encode(captions)

    # BxT(longest tokenized capation)xt5-seq-size, Bx(CLIP-L + CLIP+)
    print(seq.shape, pool.shape) # torch.Size([3, 17, 512]) torch.Size([3, 1280])