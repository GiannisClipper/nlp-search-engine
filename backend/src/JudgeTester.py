from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline # type: ignore
from .settings import pretrained_models

tokenizer_id = pretrained_models[ 'judge-model-tokenizer' ]
model_id = pretrained_models[ 'judge-model' ]

tokenizer = AutoTokenizer.from_pretrained( tokenizer_id )
model = AutoModelForCausalLM.from_pretrained( model_id )

# build a generation pipeline
generator = pipeline( "text-generation", model=model, tokenizer=tokenizer, max_new_tokens=10 )
# max_new_tokens limits the number of tokens generated, since we only want "yes" or "no" a short output is enough

# define query-passage pair
queries = [ 
    "How does photosynthesis work?",
    "What is climate change?"
]

answers = [ 
    "Photosynthesis is the process by which green plants use sunlight to synthesize food from carbon dioxide and water.",
    "Climate change refers to long-term shifts in temperatures and weather patterns."
]

for query, answer in zip( queries, answers ):

    prompt = f"""
    Query: {query}
    Candidate: {answer}
    Is this relevant to the query? Answer "yes" or "no":
    """

    # run LLM
    result = generator( prompt )#[0][ 'generated_text' ]

    print( "Result:", result )
