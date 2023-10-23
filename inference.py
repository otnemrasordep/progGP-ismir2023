from model import TransformerXL
import pickle
import random
import os
import time
import torch
import random
import yaml
import json

import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def generateFromModel(num_samples, epoch, models_dir, prompt_filename, model_outputdir, n_tokens):
       
    model_path = os.path.join(models_dir, 'ep_{}.pth.tar'.format(str(epoch)))
    # print(model_path)
    output_prefix = str(epoch)+ '_'
    
    # Insert folder path for pre-trained model
    pretrainCfg = yaml.full_load(open(os.path.join(models_dir,"full-data-config.yml"), 'r')) 
    modelConfig = pretrainCfg['MODEL']
    # print(modelConfig['n_token'])

    # load dictionary
    #event2word, word2event = pickle.load(open(inferenceConfig['dictionary_path'], 'rb'))
    event2word = pickle.load(open("vocab_song_artist.pkl", 'rb'))
    word2event = pickle.load(open("rev_vocab_song_artist.pkl", 'rb'))
    #print(event2word)

    # declare model
    #device = torch.device("cuda" if not inferenceConfig["no_cuda"] and torch.cuda.is_available() else "cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Device to generate:', device)

    # declare model
    model =  TransformerXL(
            modelConfig,
            device,
            event2word=event2word, 
            word2event=word2event, 
            is_training=False)

    # inference
    song_time_list = []
    words_len_list = []
    primer = 1      # use the empty prompt
    for idx in range(num_samples):
        print(f'==={idx+1}/{num_samples}===')
        #print(midi_folder, output_prefix + str(idx))
        song_time, word_len = model.inference(
            model_path = model_path,
            token_lim=n_tokens,
            strategies=['temperature', 'nucleus'],
            params={'t': 1.2 ,'p': 0.9},
            primer=primer,
            id = idx, 
            output_path = model_outputdir,
            prompt_filename = prompt_filename)

            
        #print('song time:',  song_time)
        #print('word_len:', word_len)
        words_len_list.append(word_len)
        song_time_list.append(song_time)
    

    #print('ave token time:', sum(words_len_list) / sum(song_time_list))
    #print('ave song time:', np.mean(song_time_list))

    runtime_result = {
        'song_time':song_time_list,
        'words_len_list': words_len_list,
        'ave token time:': sum(words_len_list) / sum(song_time_list),
        'ave song time': float(np.mean(song_time_list)),
    }
    

    with open('runtime_stats.json', 'w') as f:
        json.dump(runtime_result, f)