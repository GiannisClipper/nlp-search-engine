from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer, pipeline # type: ignore
from ..settings import pretrained_models

# +--------------------------------+
# | Code to download a judge model |
# +--------------------------------+

# RUN: python -m src.pretrained_models.JudgeModelDownloader
if __name__ == "__main__": 

    # $ huggingface-cli login
    # Enter your token (input will not be visible): 
    # Add token as git credential? (Y/n) 
    # Token is valid (permission: read).
    # The token `first-for-tinyllama` has been saved to /home/user/.cache/huggingface/stored_tokens

    # model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0" # Access denied. Make sure your token has the correct permissions.
    # model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0" # ok when signed up and got token
    # model_id = "microsoft/phi-2" # Access denied. Make sure your token has the correct permissions.
    # model_id = "google/flan-t5-small" # no good results, responded no instead of yes in simple queries
    model_id = "google/flan-t5-base" # better responses than 'small' model

    print( f'Loading {model_id} tokenizer...' )
    tokenizer = AutoTokenizer.from_pretrained( model_id )
    print( f'Saving tokenizer...' )
    tokenizer.save_pretrained( pretrained_models[ 'judge-model-tokenizer' ] )

    print( f'Loading {model_id} model...' )
    # model = AutoModelForCausalLM.from_pretrained( model_id )
    model = AutoModelForSeq2SeqLM.from_pretrained( model_id )
    print( f'Saving model...' )
    model.save_pretrained( pretrained_models[ 'judge-model' ] )
