# -*- coding: UTF-8 -*-


import sys
import os
import math
import pickle
import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')





def draw_scatter():

    with open('nocontrained/motifnet_predcls_sg_eval_result_mean_recall.pkl', 'rb') as f:
        cvpr18 = pickle.load(f)
    with open('nocontrained/kern_predcls_sg_eval_result_mean_recall.pkl', 'rb') as f:
        our = pickle.load(f)
    with open('count_pred_all.json', 'r') as f:
        data = json.load(f)
    rel_list = []
    num_list = []
    for rel, num in data:
        rel_list.append(rel)
        num_list.append(num)
    num_list = np.array(num_list) * 100
    cvpr18_list = []
    our_list = []
    for r in rel_list:
        cvpr18_list.append(cvpr18[r]['R@50'])
        our_list.append(our[r]['R@50'])
    plt.figure(figsize=(10,5))
    x = np.arange(50)+1
    y1 = np.array(cvpr18_list) * 100
    y2 = np.array(our_list) * 100
    y3 = y2 - y1
    plt.scatter(num_list, y3 ,s=30,color='green',marker='o',alpha=0.5)
  
    plt.xlabel('Relationship proportion (%)', fontsize=14)
    plt.ylabel('Improvement proportion (%)', fontsize=14)

    plt.grid(True, linestyle='--', axis='y')
    # plt.legend()
    plt.tight_layout()
    plt.savefig('scatter.pdf')



def draw_difference_compare():

    

    with open('nocontrained/motifnet_predcls_sg_eval_result_mean_recall.pkl', 'rb') as f:
        cvpr18 = pickle.load(f)
    with open('nocontrained/kern_predcls_sg_eval_result_mean_recall.pkl', 'rb') as f:
        our = pickle.load(f)
    with open('count_pred_all.json', 'r') as f:
        data = json.load(f)
    rel_list = []
    num_list = []
    for rel, num in data:
        rel_list.append(rel)
        num_list.append(num)
    cvpr18_list = []
    our_list = []
    for r in rel_list:
        cvpr18_list.append(cvpr18[r]['R@50'])
        our_list.append(our[r]['R@50'])
    plt.figure(figsize=(10,5))
    x = np.arange(50)+1
    y1 = np.array(cvpr18_list) * 100
    y2 = np.array(our_list) * 100
    y3 = y2 - y1
    plt.bar(x, y3, alpha=0.9, width = 0.7, facecolor = 'green', edgecolor = 'white', label='SMN', lw=1)
    y4 = y3.copy()
    y4[y4>0] = 0
    plt.bar(x, y4, alpha=0.9, width = 0.7, facecolor = 'red', edgecolor = 'white', label='SMN', lw=1)
    
    xticks1=rel_list
    plt.xticks(x,xticks1,fontsize=14,rotation=90) 

    plt.ylabel('R@50 Improvement (%)', fontsize=14) 

    for a,b,c,d in zip(x,y1,y2, y3):
        plt.text(a, d if d >= 0 else d - 10, '%+-.2f' % d, ha='center', va= 'bottom',fontsize=14, rotation=90) 
    plt.xlim(0,51)
    plt.ylim(-5, 35) 
    plt.ylim(-15, 45)
    plt.grid(True, linestyle='--', axis='y')
    # plt.legend()
    plt.tight_layout()
    plt.savefig('difference_compare.pdf')






def draw_compare():
    with open('nocontrained/motifnet_predcls_sg_eval_result_mean_recall.pkl', 'rb') as f:
        cvpr18 = pickle.load(f)
    with open('nocontrained/kern_predcls_sg_eval_result_mean_recall.pkl', 'rb') as f:
        our = pickle.load(f)
    with open('count_pred_all.json', 'r') as f:
        data = json.load(f)

    rel_list = []
    num_list = []
    for rel, num in data:
        rel_list.append(rel)
        num_list.append(num)
    cvpr18_list = []
    our_list = []
    for r in rel_list:
        cvpr18_list.append(cvpr18[r]['R@50'])
        our_list.append(our[r]['R@50'])
    plt.figure(figsize=(10,5))
    x = np.arange(50)+1
    y1 = np.array(cvpr18_list) * 100
    y2 = np.array(our_list) * 100
    y3 = y2 - y1
    plt.bar(x-0.175, y2, alpha=1, width = 0.35, facecolor = 'coral', edgecolor = 'white', label='Ours', lw=1)
    plt.bar(x+0.175, y1, alpha=1, width = 0.35, facecolor = 'c', edgecolor = 'white', label='SMN', lw=1)

    y4 = y3.copy()
    y4[y4>0] = 0

    xticks1=rel_list

    plt.xticks(x,xticks1,fontsize=14,rotation=90) 
 
    plt.ylabel('R@50 (%)', fontsize=14) 

    for a,b,c,d in zip(x,y1,y2, y3):
        plt.text(a, b+0.5 if b>c else c+0.5, '%+-.2f' % d, ha='center', va= 'bottom',fontsize=14, rotation=90)
    plt.xlim(0,51)
    plt.ylim(0, 119) 
    plt.grid(True, linestyle='--', axis='y')
    plt.legend(fontsize=14)
    plt.tight_layout()
    plt.savefig('compare.pdf')





def draw_bar(file_name):
    with open(file_name, 'r') as f:
        data = json.load(f)
    rel_list = []
    num_list = []
    for rel, num in data:
        rel_list.append(rel)
        num_list.append(num)

    

    # in CVPR paper, the font size is 14

    plt.figure(figsize=(10,5))

    x=np.arange(50)+1 

    y=np.array(num_list) * 100
    xticks1=rel_list 

    plt.bar(x,y,width = 0.7,align='center',color = 'lightcoral',alpha=1, edgecolor = 'white')

    plt.tick_params(labelsize=14)
    plt.xticks(x,xticks1,fontsize=12,rotation=90) 

    # plt.xlabel('Relationship')
    plt.ylabel('Proportion (%)', fontsize=14) 

    for a,b in zip(x,y):

        plt.text(a, b+0.5, '%.3f' % b, ha='center', va= 'bottom',fontsize=12, rotation=90)

    plt.ylim(0, 42)
    plt.xlim(0, 51)
    plt.grid(True, linestyle='--', axis='y')
    plt.tight_layout()

    plt.savefig('count_pred_all.pdf')


if __name__ == "__main__":


    draw_bar('count_pred_all.json')

    draw_compare()

    draw_scatter()
    draw_difference_compare()
