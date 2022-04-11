#!/usr/bin/env python
# coding: utf-8

# In[1]:


# coding=utf-8
from __future__ import unicode_literals

# In[2]:


import pyecharts.options as opts
from pyecharts.charts import Line
import numpy as np
import pandas as pd
# 图像输出库
from pyecharts.render import make_snapshot
from snapshot_selenium import snapshot

# #### 算法有效性

# In[3]:


file1 = open("图1.txt", "r")
list1 = file1.readlines()
train_loss_str = list1[0][14:-2]
train_loss = train_loss_str.split(',')
test_loss_str = list1[2][13:-2]
test_loss = test_loss_str.split(',')
len(test_loss)

# In[4]:


c1 = (Line()
      .add_xaxis(list(range(1200)))
      .add_yaxis('IPNN Train', train_loss, is_symbol_show=False, is_smooth=True,
                 linestyle_opts=opts.LineStyleOpts(width=1.5), color='#FF0000')
      .add_yaxis('IPNN Test', test_loss, is_symbol_show=False, is_smooth=True,
                 linestyle_opts=opts.LineStyleOpts(width=1.5), color='#13447C')
      .set_global_opts(xaxis_opts=opts.AxisOpts(type_='value',
                                                name="Steps",
                                                name_location='center',
                                                name_gap=23,
                                                interval=100,
                                                max_=1100),
                       yaxis_opts=opts.AxisOpts(name='MSE Loss',
                                                max_=0.04,
                                                name_location='middle',
                                                name_gap=33),
                       title_opts=opts.TitleOpts(  # title="Algorithm Effectiveness",
                           pos_left='center'),
                       legend_opts=opts.LegendOpts(pos_left='85%',
                                                   pos_top='15%',
                                                   orient='vertical',
                                                   textstyle_opts=opts.TextStyleOpts(font_size=13,
                                                                                     font_family='Times New Roman'), )
                       ))
c1.render_notebook()
# 输出成png图片
# make_snapshot(snapshot, c1.render(), "line1.png")


# #### Baseline对比

# In[41]:


file2 = open("图2.txt", "r")
list2 = file2.readlines()
ipnn_test = list(map(float, list2[0][8:-2].split(', ')))
fm_test = list(map(float, list2[1][6:-2].split(', ')))
lr_test = list(map(float, list2[2][6:-2].split(', ')))
mlp_test = list(map(float, list2[3][7:-2].split(', ')))

# In[43]:


list_4test = [ipnn_test, fm_test, lr_test, mlp_test]

# In[44]:


A = []
for ix in list_4test:
    array_ipnn = np.array(ix)
    aa = np.sum(array_ipnn) / 1081
    A.append(aa)
print(A)

# In[30]:


c2 = (Line()
      .add_xaxis(list(range(1200)))
      .add_yaxis('LR Test', lr_test, is_symbol_show=False,
                 linestyle_opts=opts.LineStyleOpts(width=0.8), color='#008754')
      .add_yaxis('FM Test', fm_test, is_symbol_show=False,
                 linestyle_opts=opts.LineStyleOpts(width=0.8), color='#FA842B')
      .add_yaxis('MLP Test', mlp_test, is_symbol_show=False,
                 linestyle_opts=opts.LineStyleOpts(width=0.8), color='#3481B8')
      .add_yaxis('IPNN Test', ipnn_test, is_symbol_show=False,
                 linestyle_opts=opts.LineStyleOpts(width=2.5))

      .set_global_opts(xaxis_opts=opts.AxisOpts(type_='value',
                                                name="Step",
                                                name_location='center',
                                                name_gap=23,
                                                interval=100,
                                                max_=1100),
                       yaxis_opts=opts.AxisOpts(name='MSE Loss',
                                                max_=0.12,
                                                name_location='middle',
                                                name_gap=33),
                       title_opts=opts.TitleOpts(  # title="Baseline",
                           pos_left='center'),
                       legend_opts=opts.LegendOpts(pos_left='90%',
                                                   pos_top='10%',
                                                   orient='vertical',
                                                   textstyle_opts=opts.TextStyleOpts(font_size=13,
                                                                                     font_family='Times New Roman'), )
                       ))
c2.render_notebook()

# #### 数据集大小Baseline对比

# In[11]:


mlp = [0.023247607, 0.014356669, 0.0135581475, 0.012564473, 0.012115602, 0.010049683, 0.009325102, 0.009064125,
       0.007739864, 0.0075026294, 0.0073862947, 0.007210679, 0.0070657646, 0.0062178555, 0.0056320643, 0.0054395553,
       0.005268827, 0.0050810794, 0.0031496142]
ipnn = [0.027631503, 0.014782149, 0.012958493, 0.0120807635, 0.011889749, 0.009702757, 0.009607976, 0.0073371176,
        0.0072315424, 0.0069291, 0.0065841763, 0.006157991, 0.0049522826, 0.0046759914, 0.0031964967, 0.003192961,
        0.0019372273, 0.0018209473, 0.0006681722]
fm = [0.022109373, 0.014290156, 0.014019048, 0.012838228, 0.011057898, 0.010894461, 0.01007993, 0.010071381,
      0.009818837, 0.0097520845, 0.009457377, 0.009100895, 0.008913092, 0.0075484784, 0.006767017, 0.005468934,
      0.005102824, 0.003989942, 0.0038300075]
lr = [0.022241117, 0.022096768, 0.021161005, 0.021071568, 0.020160737, 0.019645965, 0.01929296, 0.018316636,
      0.018310778, 0.01726826, 0.017096967, 0.014176544, 0.013890382, 0.011071431, 0.010396447, 0.009528853,
      0.008543863, 0.007175311, 0.006823742]

# In[37]:


c3 = (Line()
      .add_xaxis([i * 15 for i in range(1, 20)])
      .add_yaxis('LR', lr, is_symbol_show=False, linestyle_opts=opts.LineStyleOpts(width=0.8), color='#3481B8')
      .add_yaxis('FM', fm, is_symbol_show=False, linestyle_opts=opts.LineStyleOpts(width=0.8), color='#28713E')
      .add_yaxis('MLP', mlp, is_symbol_show=False, linestyle_opts=opts.LineStyleOpts(width=0.8), color='#8F4E35')
      .add_yaxis('IPNN', ipnn, is_symbol_show=False, linestyle_opts=opts.LineStyleOpts(width=3))
      .set_global_opts(xaxis_opts=opts.AxisOpts(type_='value',
                                                name="Data Sizes",
                                                name_location='center',
                                                name_gap=23
                                                ),
                       yaxis_opts=opts.AxisOpts(name='MSE Loss',
                                                max_=0.03,
                                                name_location='middle',
                                                name_gap=38),
                       title_opts=opts.TitleOpts(  # title="Different Sizes of Datasets",
                           pos_left='center',
                           title_textstyle_opts=opts.TextStyleOpts(color="black", font_size=16),
                       ),
                       legend_opts=opts.LegendOpts(pos_left='85%',
                                                   pos_top='15%',
                                                   orient='vertical',
                                                   textstyle_opts=opts.TextStyleOpts(font_size=13,
                                                                                     font_family='Times New Roman'), )
                       ))
c3.render_notebook()

# #### ipnn结构对比

# In[27]:


file4 = open("图4.txt", "r")
list4 = file4.readlines()
ipnn_16_1 = list(map(float, list4[0][13:-2].split(', ')))
ipnn_32_1 = list(map(float, list4[1][13:-2].split(', ')))
ipnn_64_1 = list(map(float, list4[2][13:-2].split(', ')))
ipnn_32_16_1 = list(map(float, list4[3][16:-2].split(', ')))
ipnn_64_32_1 = list(map(float, list4[4][16:-2].split(', ')))
ipnn_128_64_1 = list(map(float, list4[5][17:-2].split(', ')))
ipnn_64_32_16_1 = list(map(float, list4[6][19:-2].split(', ')))
ipnn_128_64_32_1 = list(map(float, list4[7][20:-2].split(', ')))
ipnn_256_128_64_1 = list(map(float, list4[8][21:-2].split(', ')))
# 1081


# In[39]:


ipnn_16_1

# In[55]:


list_ipnn = [ipnn_16_1, ipnn_32_1, ipnn_64_1, ipnn_32_16_1, ipnn_64_32_1, ipnn_128_64_1, ipnn_64_32_16_1,
             ipnn_128_64_32_1, ipnn_256_128_64_1]

# In[58]:


A = []
for ix in list_ipnn:
    array_ipnn = np.array(ix)
    aa = np.sum(array_ipnn) / 1081
    A.append(aa)
print(A)

# In[36]:


c4 = (Line()
      .add_xaxis(list(range(1100)))
      .add_yaxis('IPNN_16_1', ipnn_16_1, is_symbol_show=False)
      .add_yaxis('IPNN_32_1', ipnn_32_1, is_symbol_show=False, color='#E72512',
                 linestyle_opts=opts.LineStyleOpts(width=2.5))
      .add_yaxis('IPNN_64_1', ipnn_64_1, is_symbol_show=False)
      .add_yaxis('IPNN_32_16_1', ipnn_32_16_1, is_symbol_show=False)
      .add_yaxis('IPNN_64_32_1', ipnn_64_32_1, is_symbol_show=False)
      .add_yaxis('IPNN_128_64_1', ipnn_128_64_1, is_symbol_show=False)
      .add_yaxis('IPNN_64_32_16_1', ipnn_64_32_16_1, is_symbol_show=False)
      .add_yaxis('IPNN_128_64_32_1', ipnn_128_64_32_1, is_symbol_show=False)
      .add_yaxis('IPNN_256_128_64_1', ipnn_256_128_64_1, is_symbol_show=False)
      .set_global_opts(xaxis_opts=opts.AxisOpts(type_='value',
                                                name="Times",
                                                name_location='center',
                                                name_gap=23,
                                                interval=100,
                                                max_=1200),
                       yaxis_opts=opts.AxisOpts(name='MSE Loss',
                                                max_=0.08,
                                                name_location='middle',
                                                name_gap=38),
                       title_opts=opts.TitleOpts(  # title="IPNN Structurs Comparison",
                           pos_left='center'),
                       legend_opts=opts.LegendOpts(pos_left='81%',
                                                   pos_top='5%',
                                                   orient='vertical',
                                                   textstyle_opts=opts.TextStyleOpts(font_size=13,
                                                                                     font_family='Times New Roman'), )
                       ))
c4.render_notebook()
# 输出保存为图片
# make_snapshot(snapshot, c4.render(), "/Users/wangda/Desktop/123.png")

