import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint
from langchain import PromptTemplate, LLMChain
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Load environment variables from .env file
load_dotenv()

# Set the Hugging Face API key from the environment variable
os.environ['huggingfaceid'] = os.getenv('HUGGINGFACE_API_KEY')

# Initialize the Hugging Face endpoint
repo_id = 'mistralai/Mistral-7B-Instruct-v0.3'
llm = HuggingFaceEndpoint(repo_id=repo_id, max_length=128, temperature=0.7, token=os.environ['huggingfaceid'])

# Create a prompt template
question = 'What is the capital of France?'
template = '''Question: {question}
Answer: Let's think step by step.'''
prompt = PromptTemplate(template=template, input_variables=['question'])
print(prompt)

# Create a LangChain LLMChain with the prompt and the LLM
llm_chain = LLMChain(llm=llm, prompt=prompt)
print(llm_chain.invoke({'question': question}))

# Initialize a Hugging Face pipeline
model_id = 'gpt-2'
model = AutoModelForCausalLM.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)

pipe = pipeline('text-generation', model=model, tokenizer=tokenizer, max_new_tokens=100)
hf = HuggingFacePipeline(pipe)

# Invoke the pipeline with an example prompt
print(hf.invoke('Once upon a time'))

# Initialize the pipeline for GPU usage
gpu_llm = HuggingFacePipeline.from_model_id(
    model_id='gpt-2',
    task='text-generation',
    device=0,
    pipeline_kwargs={'max_new_tokens': 100}
)

# Create a prompt template
template = """Question: {question}
Answer: Let's think step by step."""
prompt = PromptTemplate.from_template(template)

# Create a chain with the prompt and the GPU LLM
chain = prompt | gpu_llm

# Provide a question and invoke the chain
question = 'How does photosynthesis work?'
response = chain.invoke({"question": question})
print(response)
