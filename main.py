from fastapi import FastAPI
from transformers import GPT2LMHeadModel,GPT2Tokenizer
import os
from os import path
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
    
)

@app.get("/CMSAI/isGPT2ModelAvailable")
def isGPT2ModelAvailable():
    gpt2ModelFolder = path.relpath("models/GPT2")
    if(os.path.isdir(gpt2ModelFolder)):
         #try:            
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        model = GPT2LMHeadModel.from_pretrained("gpt2",pad_token_id = tokenizer.eos_token_id)
        tokenizer.save_pretrained(gpt2ModelFolder)
        model.save_pretrained(gpt2ModelFolder)
            #return True
        # except:
             #return False

    else:
        return False    

@app.post("/CMSAI/GPT2Model")
def textGenerationUsingGPT2(input:str):
    gpt2ModelFolder = path.relpath("models/GPT2")
    tokenizer = GPT2Tokenizer.from_pretrained(gpt2ModelFolder)
    model = GPT2LMHeadModel.from_pretrained(gpt2ModelFolder,local_files_only= True)
    input_ids = tokenizer.encode(input, return_tensors = 'pt')

    #hardcoded different settings just for demo 
    output = model.generate(input_ids, max_length = 100, num_beams = 5, no_repeat_ngram_size = 2, early_stopping = True, temperature = 0.8, do_sample = True)
    try:
        return tokenizer.decode(output[0], skip_special_tokens = True, clean_up_tokenization_spaces = True)
    except:
        return "Server Error: Please try again later"


