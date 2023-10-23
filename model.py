import sys
import torch
import torch.nn as nn
import torch.optim as optim
import math
import numpy as np
import pandas as pd
import shutil
import copy
import os
import time
import json
from sklearn.model_selection import train_test_split
from modules import MemTransformerLM
from glob import glob
# for scheduling
from transformers import AdamW, WarmUp, get_polynomial_decay_schedule_with_warmup

import collections
import pickle 
import numpy as np

import saver

# ================================ #
def network_paras(model):
    # compute only trainable params
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params


class TransformerXL(object):
    def __init__(self, modelConfig, device, event2word, word2event, is_training=True):

        self.event2word = event2word
        self.word2event = word2event
        self.modelConfig = modelConfig

        # model settings    
        self.n_layer= modelConfig['n_layer']
        self.d_model = modelConfig['d_model']
        self.seq_len= modelConfig['seq_len']
        self.mem_len =  modelConfig['mem_len']

        self.tgt_len = modelConfig['tgt_len']
        self.ext_len = modelConfig['ext_len']
        self.eval_tgt_len = modelConfig['eval_tgt_len']

        self.init = modelConfig['init']
        self.init_range = modelConfig['init_range']
        self.init_std = modelConfig['init_std']
        self.proj_init_std = modelConfig['proj_init_std']

        #model
        self.is_training = is_training
        self.device = device  
        

    def init_weight(self, weight):
        if self.init == 'uniform':
            nn.init.uniform_(weight, -self.init_range, self.init_range)
        elif self.init == 'normal':
            nn.init.normal_(weight, 0.0, self.init_std)

    def init_bias(self, bias):
        nn.init.constant_(bias, 0.0)
            
    def weights_init(self,m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            if hasattr(m, 'weight') and m.weight is not None:
                self.init_weight(m.weight)
            if hasattr(m, 'bias') and m.bias is not None:
                self.init_bias(m.bias)
        elif classname.find('Embedding') != -1:
            if hasattr(m, 'weight'):
                self.init_weight(m.weight)
        elif classname.find('LayerNorm') != -1:
            if hasattr(m, 'weight'):
                nn.init.normal_(m.weight, 1.0, self.init_std)
            if hasattr(m, 'bias') and m.bias is not None:
                self.init_bias(m.bias)
        elif classname.find('TransformerLM') != -1:
            if hasattr(m, 'r_emb'):
                self.init_weight(m.r_emb)
            if hasattr(m, 'r_w_bias'):
                self.init_weight(m.r_w_bias)
            if hasattr(m, 'r_r_bias'):
                self.init_weight(m.r_r_bias)
            if hasattr(m, 'r_bias'):
                self.init_bias(m.r_bias)


    def get_model(self, pretrain_model=None):
        model = MemTransformerLM(self.modelConfig, is_training=self.is_training)

        st_eopch = 0
        if pretrain_model:
            checkpoint = torch.load(pretrain_model, map_location='cuda:0')
            # print('Pretrained model config:')
            # print('epoch: ', checkpoint['epoch'])
            # print('best_loss: ', checkpoint['best_loss'])
            # print(json.dumps(checkpoint['model_setting'], indent=1, sort_keys=True))
            # print(json.dumps(checkpoint['train_setting'], indent=1, sort_keys=True))

            try:
                model.load_state_dict(checkpoint['state_dict'])
                # print('{} loaded.'.format(pretrain_model))  
            except:
                print('Loaded weights have different shapes with the model. Please check your model setting.')
                exit()
            st_eopch = checkpoint['epoch'] + 1

        else:
            model.apply(self.weights_init)
            model.word_emb.apply(self.weights_init) 
        return st_eopch ,model.to(self.device)


    def save_checkpoint(self, state, root, save_freq=10):
        if state['epoch'] % save_freq == 0:
            torch.save(state, os.path.join(root,'ep_{}.pth.tar'.format(state['epoch'])))

    def inference(self, model_path, token_lim, strategies, params, id, output_path, primer=1, prompt_filename=''):
        

        if not os.path.exists(output_path):
            os.mkdir(output_path)
        
        _, model = self.get_model(model_path)
        model.eval()
        
        # initial start
        words = [[]]

        # text string to save into file
        final = str()

              
        if primer==1:
            with open(prompt_filename) as file:
                prompt_prog = [line.rstrip() for line in file]
                prompt_prog.pop() # get rid of end line
            beg_list = prompt_prog


        print(beg_list)

 
        beg_event2word = list()
        for ele in beg_list:
            beg_event2word.append(self.event2word[ele])

        words[-1] += beg_event2word
        final = "\n".join(beg_list) 
        final+='\n'
        # initialize mem
        mems = tuple()
        song_init_time = time.time()
        # generate
        initial_flag = True
        generate_n_bar = 0
        batch_size = 1
        n_tokens = len(words[0])
        while len(words[0]) < token_lim:
            # prepare input
            if initial_flag:
                temp_x = np.zeros((len(words[0]), batch_size))

                for b in range(batch_size):
                    for z, t in enumerate(words[b]):
                        temp_x[z][b] = t
                
                initial_flag = False
            else:
                temp_x = np.zeros((1, batch_size))
                
                for b in range(batch_size):
                    temp_x[0][b] = words[b][-1] ####?####

            temp_x = torch.from_numpy(temp_x).long().to(self.device)     
            st_time = time.time()
            
            _logits, mems = model.generate(temp_x, *mems)
            logits = _logits.cpu().squeeze().detach().numpy()

            # temperature or not
            if 'temperature' in strategies:
                probs = self.temperature(logits=logits, temperature=params['t'])
                
            else:
                probs = self.temperature(logits=logits, temperature=1.)
            # sampling

            word = self.nucleus(probs=probs, p=params['p']) 

            #CONDITION TO TACKLE THE "measure:repeat"
            flag=0
            while flag==0:
                #print(self.word2event[word])
                if  "measure" in self.word2event[word] and "repeat" in self.word2event[word] and len(self.word2event[word].split(":"))>2 and int(self.word2event[word].split(":")[-1])>4:
                    probs = self.temperature(logits=logits, temperature=10) #10
                    word = self.nucleus(probs=probs, p=0.9) #0.99
                else:
                   flag=1

            # CONDITION TO TACKLE THE "artist:"
            # which seems to pop up if the prompt is empty (?)
            if 'artist' in self.word2event[word]:
                probs = self.temperature(logits=logits, temperature=10) #10
                word = self.nucleus(probs=probs, p=0.99) #0.99

            # CONDITION TO TACKLE THE "end"
            if self.word2event[word] == 'end':
                probs = self.temperature(logits=logits, temperature=10) #5
                word = self.nucleus(probs=probs, p=0.99) #0.99

            
            words[0].append(word)
            
            print(len(words[0]), self.word2event[word])

            
            final += self.word2event[word] + '\n'

            # record n_bar
            if word == self.event2word['new_measure']:     #previously "Bar_None"
                generate_n_bar += 1

        temperatura = params['t']
        parametro_n = params['p']

        generated_file_name = output_path + prompt_filename.split('/')[-1].split('.')[0] + '_based' + "_id_"+ str(id) +".txt"


        with open(generated_file_name, "w") as text_file:
            final += 'end\n'
            text_file.write(final)    
        

        song_total_time = time.time() - song_init_time
        print('Total words generated: ', len(words[0]))
        return song_total_time, len(words[0])

    ########################################
    # search strategy: temperature (re-shape)
    ########################################
    def temperature(self, logits, temperature):
        probs = np.exp(logits / temperature) / np.sum(np.exp(logits / temperature))
        #print(type(probs))
        return probs

    ########################################
    # search strategy: topk (truncate)
    ########################################
    def topk(self, probs, k):
        sorted_index = np.argsort(probs)[::-1]
        candi_index = sorted_index[:k]
        candi_probs = [probs[i] for i in candi_index]
        candi_probs /= sum(candi_probs)
        word = np.random.choice(candi_index, size=1, p=candi_probs)[0]
        return word

    ########################################
    # search strategy: nucleus (truncate)
    ########################################
    def nucleus(self, probs, p):
        probs /= sum(probs)
        sorted_probs = np.sort(probs)[::-1]
        sorted_index = np.argsort(probs)[::-1]
        cusum_sorted_probs = np.cumsum(sorted_probs)
        after_threshold = cusum_sorted_probs > p
        if sum(after_threshold) > 0:
            last_index = np.where(after_threshold)[0][0] + 1
            candi_index = sorted_index[:last_index]
        else:
            candi_index = sorted_index[:3] # just assign a value
        candi_probs = [probs[i] for i in candi_index]
        candi_probs /= sum(candi_probs)
        word = np.random.choice(candi_index, size=1, p=candi_probs)[0]
        return word