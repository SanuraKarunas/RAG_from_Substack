from typing import List, Any
from langchain.llms import BaseLLM
from langchain.schema import LLMResult
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, GenerationConfig
from pydantic import BaseModel, Field
import torch

class TransformersWrapper(BaseLLM, BaseModel):
    model_name: str = Field(..., description="The name of the model to use.")
    model: Any = Field(None, description="The Transformers model instance.")
    tokenizer: Any = Field(None, description="The Transformers tokenizer instance.")
    generation_config: Any = Field(None, description="Generation parameters")

    def __init__(self, model_name: str, **kwargs):
        super().__init__(model_name=model_name, **kwargs)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if device == "cuda" else torch.float32
        
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
        # self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            # attn_implementation="eager",
            torch_dtype=torch_dtype,
            trust_remote_code=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        pad_token = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        self.generation_config = GenerationConfig(
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
            max_new_tokens=2000,
            pad_token_id=pad_token
        )

    def _generate(self, prompts: List[str], **kwargs) -> LLMResult:
        inputs = self.tokenizer(
            prompts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True,
            return_attention_mask=True
        ).to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            generation_config=self.generation_config
        )
    
        generated_texts = [
            self.tokenizer.decode(
                output, 
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            ) for output in outputs
        ]
        
        return LLMResult(generations=[[{"text": text}] for text in generated_texts])

    def _llm_type(self) -> str:
        return "transformers"