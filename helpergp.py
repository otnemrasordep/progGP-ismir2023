from dadagp import *
from inference import *

encode_inputdir = 'tab_prompts/'
encode_outputdir = 'tab_prompts_encoded/'

model_inputdir = 'tab_prompts_encoded/'
model_outputdir = 'generated_tabs_encoded/'
decode_inputdir = 'generated_tabs_encoded/'
decode_outputdir = 'generated_tabs/'

models_dir = 'models/'

def encodeTabPrompt(tab_prompt):
    dadaGP('encode', encode_inputdir + tab_prompt + '_prompt.gp5', encode_outputdir + tab_prompt + '.txt')
    
def generateSamples(num_samples, tab_prompt, n_tokens):
    print('Generating samples...')
    
    # Get correct model number
    model_epoch = 215
    # print(model_epoch)
    prompt_filename = encode_outputdir + tab_prompt + '.txt'
    
    # Generate encoded tabs
    generateFromModel(num_samples, model_epoch, models_dir, prompt_filename, model_outputdir, n_tokens)
    
    # Decode generated .txt to .gp5 tabs
    dadaGP('batch_decode', decode_inputdir, decode_outputdir)