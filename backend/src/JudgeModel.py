from typing import cast
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer, pipeline # type: ignore
from .settings import pretrained_models

# +-----------------------------------+
# | Code to make use of a judge model |
# +-----------------------------------+

class JudgeModel:

    def __init__( self ):

        tokenizer_id = pretrained_models[ 'judge-model-tokenizer' ]
        model_id = pretrained_models[ 'judge-model' ]

        self._tokenizer = AutoTokenizer.from_pretrained( tokenizer_id )
        self._model = AutoModelForSeq2SeqLM.from_pretrained( model_id )

        # build a generation pipeline
        self._generator = pipeline( "text2text-generation", model=self._model, tokenizer=self._tokenizer, max_new_tokens=10 )
        # max_new_tokens limits the number of tokens generated, since we only want "yes" or "no" a short output is enough

    def judge( self, query:str, answer :str) -> str:

        prompt = f'''
            QUERY: {query}
            CANDITATE: {answer}
            Is CANDITATE relevant to QUERY? Answer "yes" or "no":
        '''

        # self._generator = pipeline( "text-generation", model=self._model, tokenizer=self._tokenizer, max_new_tokens=10 )
        result = self._generator( prompt )[0][ 'generated_text' ] # type: ignore

        # print( "Result:", result )
        return cast( str, result )


# +----------------------------------------+
# | For development and debugging purposes |
# +----------------------------------------+

# RUN: python -m src.JudgeModel
if __name__ == "__main__": 

    model = JudgeModel()

    queries = [ 
        "How does photosynthesis work?",
        "How does photosynthesis work?",
        "What is climate change?",
        "What is climate change?"
    ]

    answers = [ 
        "Photosynthesis is the process by which green plants use sunlight to synthesize food from carbon dioxide and water.",
        "My name is John and I am studying on IT domain. I am going to finish the current.",
        "Climate change refers to long-term shifts in temperatures and weather patterns.",
        "Earth comprises of water and lands. Water can be found as seas, lakes, rivers. Climate change refers to long-term shifts in temperatures and weather patterns. Mountains, hills and plains characterize the lands... Many other things may be added in text but still the answer is included, let's try it..."
    ]

    counter = 0
    for query, answer in zip( queries, answers ):
        result = model.judge( query, answer )
        counter += 1
        print( f"{counter}." )
        print( f"Q:{query}" )
        print( f"A:{answer}" )
        print( f"R:{result}" )
