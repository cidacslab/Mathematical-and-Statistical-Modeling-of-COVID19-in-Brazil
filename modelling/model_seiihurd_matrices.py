#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 19 18:08:01 2020

@author: Felipe A. C. Pereira

Implementação do ajuste do modelo SEIIHURD com separação de grupos. Necessita
de mais verificações e funções para simplificar o input. Baseado nas classes
disponíveis no modelos.py
"""

import numpy as np
from functools import reduce
import scipy.integrate as spi
from scipy.optimize import least_squares
from platypus import NSGAII, Problem, Real
from pyswarms.single.global_best import GlobalBestPSO
import pyswarms as ps
from pyswarms.backend.topology import Star
from pyswarms.utils.plotters import plot_cost_history
from itertools import repeat
import multiprocessing as mp
import copy
import joblib

'''
Social contact matrices from 
PREM, Kiesha; COOK, Alex R.; JIT, Mark. Projecting social contact matrices in 
152 countries using contact surveys and demographic data. PLoS computational
biology, v. 13, n. 9, p. e1005697, 2017.
'''
ages_Mu_min = 5 * np.arange(16)

Mu_house = np.array([[0.47868515, 0.50507561, 0.29848922, 0.15763748, 0.26276959,
        0.40185462, 0.46855027, 0.42581354, 0.2150961 , 0.0856771 ,
        0.08705463, 0.07551931, 0.05129175, 0.02344832, 0.00793644,
        0.01072846],
       [0.35580205, 0.77874482, 0.51392686, 0.21151069, 0.08597966,
        0.28306027, 0.49982218, 0.52854893, 0.41220947, 0.15848728,
        0.07491245, 0.07658339, 0.04772343, 0.02588962, 0.01125956,
        0.01073152],
       [0.25903114, 0.63488713, 1.36175618, 0.50016515, 0.11748191,
        0.10264613, 0.24113458, 0.47274372, 0.54026417, 0.26708819,
        0.11007723, 0.04406045, 0.02746409, 0.02825033, 0.02044872,
        0.01214665],
       [0.14223192, 0.24383932, 0.53761638, 1.05325205, 0.28778496,
        0.10925453, 0.0651564 , 0.2432454 , 0.39011334, 0.41381277,
        0.23194909, 0.07541471, 0.03428398, 0.02122257, 0.01033573,
        0.00864859],
       [0.27381886, 0.15430529, 0.16053062, 0.5104134 , 0.95175366,
        0.3586594 , 0.09248672, 0.04774269, 0.15814197, 0.36581739,
        0.25544811, 0.13338965, 0.03461345, 0.01062458, 0.00844199,
        0.00868782],
       [0.59409802, 0.26971847, 0.10669146, 0.18330524, 0.39561893,
        0.81955947, 0.26376865, 0.06604084, 0.03824556, 0.11560004,
        0.23218163, 0.15331788, 0.07336147, 0.02312255, 0.00412646,
        0.01025778],
       [0.63860889, 0.75760606, 0.43109156, 0.09913293, 0.13935789,
        0.32056062, 0.65710277, 0.25488454, 0.1062129 , 0.0430932 ,
        0.06880784, 0.09938458, 0.09010691, 0.02233902, 0.01155556,
        0.00695246],
       [0.56209348, 0.87334544, 0.75598244, 0.33199136, 0.07233271,
        0.08674171, 0.20243583, 0.60062714, 0.17793601, 0.06307045,
        0.04445926, 0.04082447, 0.06275133, 0.04051762, 0.01712777,
        0.00598721],
       [0.35751289, 0.66234582, 0.77180208, 0.54993616, 0.17368099,
        0.07361914, 0.13016852, 0.19937327, 0.46551558, 0.15412263,
        0.06123041, 0.0182514 , 0.04234381, 0.04312892, 0.01656267,
        0.01175358],
       [0.208131  , 0.41591452, 0.56510014, 0.67760241, 0.38146504,
        0.14185001, 0.06160354, 0.12945701, 0.16470166, 0.41150841,
        0.14596804, 0.04404807, 0.02395316, 0.01731295, 0.01469059,
        0.02275339],
       [0.30472548, 0.26744442, 0.41631962, 0.46516888, 0.41751365,
        0.28520772, 0.13931619, 0.07682945, 0.11404965, 0.16122096,
        0.33813266, 0.1349378 , 0.03755396, 0.01429426, 0.01356763,
        0.02551792],
       [0.52762004, 0.52787011, 0.33622117, 0.43037934, 0.36416323,
        0.42655672, 0.33780201, 0.13492044, 0.0798784 , 0.15795568,
        0.20367727, 0.33176385, 0.12256126, 0.05573807, 0.0124446 ,
        0.02190564],
       [0.53741472, 0.50750067, 0.3229994 , 0.30706704, 0.21340314,
        0.27424513, 0.32838657, 0.26023515, 0.13222548, 0.07284901,
        0.11950584, 0.16376401, 0.25560123, 0.09269703, 0.02451284,
        0.00631762],
       [0.37949376, 0.55324102, 0.47449156, 0.24796638, 0.19276924,
        0.20675484, 0.3267867 , 0.39525729, 0.3070043 , 0.10088992,
        0.10256839, 0.13016641, 0.1231421 , 0.24067708, 0.05475668,
        0.01401368],
       [0.16359554, 0.48536065, 0.40533723, 0.31542539, 0.06890518,
        0.15670328, 0.12884062, 0.27912381, 0.25685832, 0.20143856,
        0.12497647, 0.07565566, 0.10331686, 0.08830789, 0.15657321,
        0.05744065],
       [0.29555039, 0.39898035, 0.60257982, 0.5009724 , 0.13799378,
        0.11716593, 0.14366306, 0.31602298, 0.34691652, 0.30960511,
        0.31253708, 0.14557295, 0.06065554, 0.10654772, 0.06390924,
        0.09827735]])

Mu_school = np.array([[3.21885854e-001, 4.31659966e-002, 7.88269419e-003,
        8.09548363e-003, 5.35038146e-003, 2.18201974e-002,
        4.01633514e-002, 2.99376002e-002, 1.40680283e-002,
        1.66587853e-002, 9.47774696e-003, 7.41041622e-003,
        1.28200661e-003, 7.79120405e-004, 8.23608272e-066,
        6.37926405e-120],
       [5.40133328e-002, 4.84870697e+000, 2.70046494e-001,
        3.14778450e-002, 3.11206331e-002, 8.56826951e-002,
        1.08251879e-001, 9.46101139e-002, 8.63528188e-002,
        5.51141159e-002, 4.19385198e-002, 1.20958942e-002,
        4.77242219e-003, 1.39787217e-003, 3.47452943e-004,
        8.08973738e-039],
       [4.56461982e-004, 1.04840235e+000, 6.09152459e+000,
        1.98915822e-001, 1.99709921e-002, 6.68319525e-002,
        6.58949586e-002, 9.70851505e-002, 9.54147078e-002,
        6.70538232e-002, 4.24864096e-002, 1.98701346e-002,
        5.11869429e-003, 7.27320438e-004, 4.93746124e-025,
        1.82153965e-004],
       [2.59613205e-003, 4.73315233e-002, 1.99337834e+000,
        7.20040500e+000, 8.57326037e-002, 7.90668822e-002,
        8.54208542e-002, 1.10816964e-001, 8.76955236e-002,
        9.22975521e-002, 4.58035025e-002, 2.51130956e-002,
        5.71391798e-003, 1.07818752e-003, 6.21174558e-033,
        1.70710246e-070],
       [7.19158720e-003, 2.48833195e-002, 9.89727235e-003,
        8.76815025e-001, 4.33963352e-001, 5.05185217e-002,
        3.30594492e-002, 3.81384107e-002, 2.34709676e-002,
        2.67235372e-002, 1.32913985e-002, 9.00655556e-003,
        6.94913059e-004, 1.25675951e-003, 1.77164197e-004,
        1.21957619e-047],
       [7.04119204e-003, 1.19412206e-001, 3.75016980e-002,
        2.02193056e-001, 2.79822908e-001, 1.68610223e-001,
        2.86939363e-002, 3.56961469e-002, 4.09234494e-002,
        3.32290896e-002, 8.12074348e-003, 1.26152144e-002,
        4.27869081e-003, 2.41737477e-003, 4.63116893e-004,
        1.28597237e-003],
       [1.41486320e-002, 3.86561429e-001, 2.55902236e-001,
        1.69973534e-001, 4.98104010e-002, 8.98122446e-002,
        7.95333394e-002, 5.19274611e-002, 5.46612930e-002,
        2.64567137e-002, 2.03241595e-002, 2.96263220e-003,
        5.42888613e-003, 4.47585970e-004, 1.65440335e-048,
        3.11189454e-055],
       [2.40945305e-002, 2.11030046e-001, 1.54767246e-001,
        8.17929897e-002, 1.84061608e-002, 5.43009779e-002,
        7.39351186e-002, 5.21677009e-002, 5.63267084e-002,
        2.51807147e-002, 3.53972554e-003, 7.96646343e-003,
        5.56929776e-004, 2.08530461e-003, 1.84428290e-123,
        9.69555083e-067],
       [7.81313905e-003, 1.14371898e-001, 9.09011945e-002,
        3.80212104e-001, 8.54533192e-003, 2.62430162e-002,
        2.51880009e-002, 3.22563508e-002, 6.73506045e-002,
        2.24997143e-002, 2.39241043e-002, 6.50627191e-003,
        5.50892674e-003, 4.78308850e-004, 4.81213215e-068,
        2.40231425e-092],
       [6.55265016e-002, 2.31163536e-001, 1.49970765e-001,
        5.53563093e-001, 5.74032526e-003, 3.02865481e-002,
        5.72506883e-002, 4.70559232e-002, 4.28736553e-002,
        2.42614518e-002, 2.86665377e-002, 1.29570473e-002,
        3.24362518e-003, 1.67930318e-003, 6.20916950e-134,
        3.27297624e-072],
       [1.72765646e-002, 3.43744913e-001, 4.30902785e-001,
        4.74293073e-001, 5.39328187e-003, 1.44128740e-002,
        3.95545363e-002, 3.73781860e-002, 4.56834488e-002,
        5.92135906e-002, 2.91473801e-002, 1.54857502e-002,
        4.53105390e-003, 8.87272668e-024, 1.23797452e-117,
        5.64262349e-078],
       [6.14363036e-002, 2.98367348e-001, 2.59092700e-001,
        3.00800812e-001, 5.92454596e-003, 5.26458862e-002,
        2.02188672e-002, 3.27897605e-002, 4.07753741e-002,
        2.83422407e-002, 2.43657809e-002, 2.73993226e-002,
        8.87990718e-003, 1.13279180e-031, 7.81960493e-004,
        7.62467510e-004],
       [3.63695643e-002, 5.96870355e-002, 3.05072624e-002,
        1.45523978e-001, 1.26062984e-002, 1.69458169e-003,
        1.55127292e-002, 4.22097670e-002, 9.21792425e-003,
        1.42200652e-002, 1.10967529e-002, 5.77020348e-003,
        2.04474044e-002, 1.11075734e-002, 4.42271199e-067,
        2.12068625e-037],
       [1.67937029e-003, 2.72971001e-002, 1.05886266e-002,
        7.61087735e-032, 1.97191559e-003, 1.92885006e-003,
        1.24343737e-002, 5.39297787e-003, 5.41684968e-003,
        8.63502071e-003, 1.94554498e-003, 1.49082274e-002,
        8.11781100e-003, 1.74395489e-002, 1.11239023e-002,
        3.45693088e-126],
       [1.28088348e-028, 5.11065200e-026, 1.93019797e-040,
        7.60476035e-003, 2.63586947e-022, 1.69749024e-024,
        1.25875005e-026, 7.62109877e-003, 7.84979948e-003,
        2.11516023e-002, 3.52117832e-002, 2.14360383e-002,
        7.73902109e-003, 8.01328325e-003, 7.91285055e-003,
        2.13825814e-002],
       [2.81655586e-094, 2.11305187e-002, 8.46562506e-042,
        2.12592841e-002, 4.89802057e-036, 7.59232387e-003,
        9.77247001e-069, 2.23108239e-060, 1.43715978e-048,
        8.56015694e-060, 4.69469043e-042, 1.59822047e-046,
        2.20978550e-083, 8.85861277e-107, 1.02042815e-080,
        6.61413913e-113]])

Mu_work = np.array([[0.00000000e+000, 0.00000000e+000, 0.00000000e+000,
        0.00000000e+000, 0.00000000e+000, 0.00000000e+000,
        0.00000000e+000, 0.00000000e+000, 0.00000000e+000,
        0.00000000e+000, 0.00000000e+000, 0.00000000e+000,
        0.00000000e+000, 8.20604524e-092, 1.20585150e-005,
        3.16436834e-125],
       [0.00000000e+000, 1.16840561e-003, 9.90713236e-072,
        4.42646396e-059, 2.91874286e-006, 9.98773031e-003,
        2.58779981e-002, 5.66104376e-003, 2.12699812e-002,
        5.72117462e-003, 1.48212306e-003, 1.23926126e-003,
        1.28212945e-056, 1.34955578e-005, 7.64591325e-079,
        2.38392073e-065],
       [0.00000000e+000, 2.56552144e-003, 1.12756182e-001,
        2.40351143e-002, 2.62981485e-002, 7.56512432e-003,
        6.19587609e-002, 1.73269871e-002, 5.87405128e-002,
        3.26749742e-002, 1.24709193e-002, 2.93054408e-008,
        3.71596993e-017, 2.79780317e-053, 4.95800770e-006,
        3.77718083e-102],
       [0.00000000e+000, 1.07213881e-002, 4.28390448e-002,
        7.22769090e-001, 5.93479736e-001, 3.39341952e-001,
        3.17013715e-001, 2.89168861e-001, 3.11143180e-001,
        2.34889238e-001, 1.32953769e-001, 6.01944097e-002,
        1.47306181e-002, 8.34699602e-006, 2.85972822e-006,
        1.88926122e-031],
       [0.00000000e+000, 9.14252587e-003, 5.74508682e-002,
        4.00000235e-001, 7.93386618e-001, 7.55975146e-001,
        6.32277283e-001, 6.83601459e-001, 4.98506972e-001,
        3.82309992e-001, 2.81363576e-001, 1.23338103e-001,
        4.15708021e-002, 9.86113407e-006, 1.32609387e-005,
        3.74318048e-006],
       [0.00000000e+000, 1.04243481e-002, 7.34587492e-002,
        3.49556755e-001, 7.50680101e-001, 1.25683393e+000,
        9.01245714e-001, 8.63446835e-001, 7.70443641e-001,
        5.17237071e-001, 4.09810981e-001, 1.80645400e-001,
        5.51284783e-002, 1.60674627e-005, 1.01182608e-005,
        3.01442534e-006],
       [0.00000000e+000, 1.65842404e-002, 8.34076781e-002,
        1.89301935e-001, 5.21246906e-001, 8.54460001e-001,
        1.12054931e+000, 9.64310078e-001, 8.34675180e-001,
        6.52534012e-001, 3.79383514e-001, 2.11198205e-001,
        5.17285688e-002, 1.63795563e-005, 4.10100851e-006,
        3.49478980e-006],
       [0.00000000e+000, 1.11666639e-002, 5.03319748e-002,
        3.70510313e-001, 4.24294782e-001, 7.87535547e-001,
        8.45085693e-001, 1.14590365e+000, 1.07673077e+000,
        7.13492115e-001, 5.00740004e-001, 1.90102207e-001,
        3.59740115e-002, 1.22988530e-005, 9.13512833e-006,
        6.02097416e-006],
       [0.00000000e+000, 6.07792440e-003, 5.49337607e-002,
        2.23499535e-001, 4.82353827e-001, 7.52291991e-001,
        8.89187601e-001, 9.33765370e-001, 1.10492283e+000,
        8.50124391e-001, 5.88941528e-001, 1.94947085e-001,
        5.09477228e-002, 1.43626161e-005, 1.02721567e-005,
        1.29503893e-005],
       [0.00000000e+000, 3.31622551e-003, 7.01829848e-002,
        2.67512972e-001, 3.14796392e-001, 5.41516885e-001,
        6.95769048e-001, 7.50620518e-001, 7.50038547e-001,
        7.00954088e-001, 4.35197983e-001, 2.11283335e-001,
        3.88576200e-002, 1.62810370e-005, 1.08243610e-005,
        6.09172339e-006],
       [0.00000000e+000, 4.39576425e-004, 7.17737968e-002,
        1.89254612e-001, 2.47832532e-001, 5.16027731e-001,
        6.02783971e-001, 6.15949277e-001, 8.05581107e-001,
        7.44063535e-001, 5.44855374e-001, 2.52198706e-001,
        4.39235685e-002, 1.18079721e-005, 1.18226645e-005,
        1.01613165e-005],
       [0.00000000e+000, 4.91737561e-003, 1.08686672e-001,
        1.24987806e-001, 1.64110983e-001, 3.00118829e-001,
        4.18159745e-001, 3.86897613e-001, 4.77718241e-001,
        3.60854250e-001, 3.22466456e-001, 1.92516925e-001,
        4.07209694e-002, 1.34978304e-005, 6.58739925e-006,
        6.65716756e-006],
       [0.00000000e+000, 6.35447018e-004, 3.96329620e-002,
        1.83072502e-002, 7.04596701e-002, 1.24861117e-001,
        1.37834574e-001, 1.59845720e-001, 1.66933479e-001,
        1.56084857e-001, 1.14949158e-001, 8.46570798e-002,
        1.50879843e-002, 2.03019580e-005, 8.26102156e-006,
        1.48398182e-005],
       [7.60299521e-006, 3.36326754e-006, 7.64855296e-006,
        2.27621532e-005, 3.14933351e-005, 7.89308410e-005,
        7.24212842e-005, 2.91748203e-005, 6.61873732e-005,
        5.95693238e-005, 7.70713500e-005, 5.30687748e-005,
        4.66030117e-005, 1.41633235e-005, 2.49066205e-005,
        1.19109038e-005],
       [5.78863840e-055, 7.88785149e-042, 2.54830412e-006,
        2.60648191e-005, 1.68036205e-005, 2.12446739e-005,
        3.57267603e-005, 4.02377033e-005, 3.56401935e-005,
        3.09769252e-005, 2.13053382e-005, 4.49709414e-005,
        2.61368373e-005, 1.68266203e-005, 1.66514322e-005,
        2.60822813e-005],
       [2.35721271e-141, 9.06871674e-097, 1.18637122e-089,
        9.39934076e-022, 4.66000452e-005, 4.69664011e-005,
        4.69316082e-005, 8.42184044e-005, 2.77788168e-005,
        1.03294378e-005, 1.06803618e-005, 7.26341826e-075,
        1.10073971e-065, 1.02831671e-005, 5.16902994e-049,
        8.28040509e-043]])

Mu_other = np.array([[0.95537734, 0.46860132, 0.27110607, 0.19447667, 0.32135073,
        0.48782072, 0.54963024, 0.42195593, 0.27152038, 0.17864251,
        0.20155642, 0.16358271, 0.1040159 , 0.0874149 , 0.05129938,
        0.02153823],
       [0.51023519, 2.17757364, 0.9022516 , 0.24304235, 0.20119518,
        0.39689588, 0.47242431, 0.46949918, 0.37741651, 0.16843746,
        0.12590504, 0.12682331, 0.11282247, 0.08222718, 0.03648526,
        0.02404257],
       [0.18585796, 1.11958124, 4.47729443, 0.67959759, 0.43936317,
        0.36934142, 0.41566744, 0.44467286, 0.48797422, 0.28795385,
        0.17659191, 0.10674831, 0.07175567, 0.07249261, 0.04815305,
        0.03697862],
       [0.09854482, 0.3514869 , 1.84902386, 5.38491613, 1.27425161,
        0.59242579, 0.36578735, 0.39181798, 0.38131832, 0.31501028,
        0.13275648, 0.06408612, 0.04499218, 0.04000664, 0.02232326,
        0.01322698],
       [0.13674436, 0.1973461 , 0.33264088, 2.08016394, 3.28810184,
        1.29198125, 0.74642201, 0.44357051, 0.32781391, 0.35511243,
        0.20132011, 0.12961   , 0.04994553, 0.03748657, 0.03841073,
        0.02700581],
       [0.23495203, 0.13839031, 0.14085679, 0.5347385 , 1.46021275,
        1.85222022, 1.02681162, 0.61513602, 0.39086271, 0.32871844,
        0.25938947, 0.13520412, 0.05101963, 0.03714278, 0.02177751,
        0.00979745],
       [0.23139098, 0.18634831, 0.32002214, 0.2477269 , 0.64111274,
        0.93691022, 1.14560725, 0.73176025, 0.43760432, 0.31057135,
        0.29406937, 0.20632155, 0.09044896, 0.06448983, 0.03041877,
        0.02522842],
       [0.18786196, 0.25090485, 0.21366969, 0.15358412, 0.35761286,
        0.62390736, 0.76125666, 0.82975354, 0.54980593, 0.32778339,
        0.20858991, 0.1607099 , 0.13218526, 0.09042909, 0.04990491,
        0.01762718],
       [0.12220241, 0.17968132, 0.31826246, 0.19846971, 0.34823183,
        0.41563737, 0.55930999, 0.54070187, 0.5573184 , 0.31526474,
        0.20194048, 0.09234293, 0.08377534, 0.05819374, 0.0414762 ,
        0.01563101],
       [0.03429527, 0.06388018, 0.09407867, 0.17418896, 0.23404519,
        0.28879108, 0.34528852, 0.34507961, 0.31461973, 0.29954426,
        0.21759668, 0.09684718, 0.06596679, 0.04274337, 0.0356891 ,
        0.02459849],
       [0.05092152, 0.10829561, 0.13898902, 0.2005828 , 0.35807132,
        0.45181815, 0.32281821, 0.28014803, 0.30125545, 0.31260137,
        0.22923948, 0.17657382, 0.10276889, 0.05555467, 0.03430327,
        0.02064256],
       [0.06739051, 0.06795035, 0.0826437 , 0.09522087, 0.23309189,
        0.39055444, 0.39458465, 0.29290532, 0.27204846, 0.17810118,
        0.24399007, 0.22146653, 0.13732849, 0.07585801, 0.03938794,
        0.0190908 ],
       [0.04337917, 0.05375367, 0.05230119, 0.08066901, 0.16619572,
        0.25423056, 0.25580913, 0.27430323, 0.22478799, 0.16909017,
        0.14284879, 0.17211604, 0.14336033, 0.10344522, 0.06797049,
        0.02546014],
       [0.04080687, 0.06113728, 0.04392062, 0.04488748, 0.12808591,
        0.19886058, 0.24542711, 0.19678011, 0.17800136, 0.13147441,
        0.13564091, 0.14280335, 0.12969805, 0.11181631, 0.05550193,
        0.02956066],
       [0.01432324, 0.03441212, 0.05604694, 0.10154456, 0.09204   ,
        0.13341443, 0.13396901, 0.16682638, 0.18562675, 0.1299677 ,
        0.09922375, 0.09634331, 0.15184583, 0.13541738, 0.1169359 ,
        0.03805293],
       [0.01972631, 0.02274412, 0.03797545, 0.02036785, 0.04357298,
        0.05783639, 0.10706321, 0.07688271, 0.06969759, 0.08029393,
        0.05466604, 0.05129046, 0.04648653, 0.06132882, 0.05004289,
        0.03030569]])

def generate_reduced_matrices(age_sep, Ni):
    '''
    Receives the age_separation and populations to generate the average contact
    matrices, returns a (4, len(age_sep)+1, len(age_sep)+1) with the 4 partial
    contact matrices: house, school, work and other
    Ni is the population for each population component (16 5-years age groups)
    '''
    nMat = len(age_sep) + 1
    Ms = np.empty((4, nMat, nMat))
    age_indexes = list()
    age_indexes.append(np.flatnonzero(ages_Mu_min <= age_sep[0]))
    for i in range(1, len(age_sep)):
        age_indexes.append(np.flatnonzero((ages_Mu_min > age_sep[i-1]) * 
                                          (ages_Mu_min <= age_sep[i])))
    age_indexes.append(np.flatnonzero(ages_Mu_min > age_sep[-1]))
    for i in range(nMat):
        Nia = Ni[age_indexes[i]]
        Na = Nia.sum()
        for j in range(nMat):
            Ms[0,i,j] = (Nia * ((Mu_house[age_indexes[i]][:,age_indexes[j]]).sum(axis=1))).sum()/Na
            Ms[1,i,j] = (Nia * ((Mu_school[age_indexes[i]][:,age_indexes[j]]).sum(axis=1))).sum()/Na
            Ms[2,i,j] = (Nia * ((Mu_work[age_indexes[i]][:,age_indexes[j]]).sum(axis=1))).sum()/Na
            Ms[3,i,j] = (Nia * ((Mu_other[age_indexes[i]][:,age_indexes[j]]).sum(axis=1))).sum()/Na
    return Ms 

class SEIIHURD_age:
    ''' SEIIHURD Model'''
    def __init__(self,tamanhoPop,numeroProcessadores=None):
        self.N = tamanhoPop
        self.numeroProcessadores = numeroProcessadores
        self.pos = None

#pars dict betas, delta, kappa, p, gammaA, gammaS, h, epsilon, gammaH, gammaU, muU, muH, wU, wH
# seguindo a notação beta_12 é 2 infectando 1, onde 1 é a linha e 2 a coluna.
    def _SEIIHURD_age_eq(self, X, t, pars):
        S, E, Ia, Is, H, U, R, D, Nw = np.split(X, 9)
        StE = S * (pars['beta'] @  ((Ia * pars['delta'] + Is).reshape((-1,1)))).flatten()
        dS = - StE
        dE = StE - pars['kappa'] * E
        dIa = (1. - pars['p']) * pars['kappa'] * E - pars['gammaA'] * Ia
        dIs = pars['p'] * pars['kappa'] * E - pars['gammaS'] * Is
        dH = pars['h'] * pars['xi'] * pars['gammaS'] * Is + (1 - pars['muU'] +\
            pars['wU'] * pars['muU']) * pars['gammaU'] * U - pars['gammaH'] * H    
        dU = pars['h'] * (1 - pars['xi']) * pars['gammaS'] * Is + pars['wH'] *\
            pars['gammaH'] * H - pars['gammaU'] * U
        dR = pars['gammaA'] * Ia + (1. - pars['h']) * pars['gammaS'] * Is + \
            (1 - pars['muH']) * (1 - pars['wH']) * pars['gammaH'] * H
        dD = (1 - pars['wH']) * pars['muH'] * pars['gammaH'] * H + \
            (1 - pars['wU']) * pars['muU'] * pars['gammaU'] * U
        dNw = pars['p'] * pars['kappa'] * E 
        return np.r_[dS, dE, dIa, dIs, dH, dU, dR, dD, dNw]
    
    
    def _call_ODE(self, ts, ppars):
        betas = ppars['beta'].copy()
        pars = copy.deepcopy(ppars)
        if 'tcut' not in ppars.keys():
            tcorte = None
        else:
            tcorte = pars['tcut']
        if type(ts) in [int, float]:
            ts = np.arange(ts)
        if tcorte == None:
            tcorte = [ts[-1]]
            if type(betas) != list:
                betas = [betas]
        if tcorte[-1] < ts[-1]:
            tcorte.append(ts[-1])
        tcorte = [ts[0]] + tcorte
        tcorte.sort()
        Is0 = pars['x0'].reshape((3,-1)).sum(axis=0)
        x0 = np.r_[1. - Is0, pars['x0'], np.zeros(4*len(Is0)), pars['x0'][2*len(Is0):]]
        saida = x0.reshape((1,-1))
        Y = saida.copy()
        for i in range(1, len(tcorte)):
            cut_last = False
            pars['beta'] = betas[i-1]
            t = ts[(ts >= tcorte[i-1]) * (ts<= tcorte[i])]
            if len(t) > 0:
                if t[0] > tcorte[i-1]:
                    t = np.r_[tcorte[i-1], t]
                if t[-1] < tcorte[i]:
                    t = np.r_[t, tcorte[i]]
                    cut_last = True
                Y = spi.odeint(self._SEIIHURD_age_eq, Y[-1], t, args=(pars,))
                if cut_last:
                    saida = np.r_[saida, Y[1:-1]]
                else:
                    saida = np.r_[saida, Y[1:]]
            else:
                Y = spi.odeint(self._SEIIHURD_age_eq, Y[-1], tcorte[i-1:i+1], args=(pars,))
            
        return ts, saida


    def _fill_paramPSO(self, paramPSO):
        if 'options' not in paramPSO.keys():
            paramPSO['options'] = {'c1': 0.1, 'c2': 0.3, 'w': 0.9,'k':5,'p':2}
        if 'n_particles' not in paramPSO.keys():
            paramPSO['n_particles'] = 300
        if 'iter' not in paramPSO.keys():
            paramPSO['iter'] = 1000
        return paramPSO
    
    
    def _prepare_input(self, data):
        list_states = ['S', 'E', 'Ia', 'Is', 'H', 'U', 'R', 'D', 'Nw']
        i_integ = list()
        Y = list()
        for ke in data.keys():
            if ke == 't':
                t = data[ke]
            else:
                Y.append(data[ke])
                simb, num = ke.split("_")
                n0 = self.nages * list_states.index(simb)
                if '_ALL' in ke:
                    i_integ.append(list(range(n0,n0 + self.nages)))
                else:
                    i_integ.append(int(num) + n0)
        return i_integ, Y, t
                
                
    def _prepare_conversor(self, p2f, pothers, bound):
        padjus = list()
        if  bound != None:
            bound_new = [[], []]
        for i, par in enumerate(p2f):
            if 'beta' in par:
                if '_ALL' in par:
                    for l in range(len(pothers['beta'])):
                        for j in range(pothers['beta'][i].shape[0]):
                            for k in range(pothers['beta'][i].shape[1]):
                                padjus.append('beta_{}_{}_{}'.format(l,j,k))
                                if  bound != None:
                                    bound_new[0].append(bound[0][i])
                                    bound_new[1].append(bound[1][i])
                else:
                    padjus.append(par)
                    if  bound != None:
                        bound_new[0].append(bound[0][i])
                        bound_new[1].append(bound[1][i])
                    
            elif '_ALL' in par:
                name = par.split('_')[0]
                for j in range(len(pothers[name])):
                    padjus.append('{}_{}'.format(name, j))
                    if  bound != None:
                        bound_new[0].append(bound[0][i])
                        bound_new[1].append(bound[1][i])
            else:
                padjus.append(par)
                if  bound != None:
                    bound_new[0].append(bound[0][i])
                    bound_new[1].append(bound[1][i])
        if  bound != None:
            bound_new[0] = np.array(bound_new[0])
            bound_new[1] = np.array(bound_new[1])
        return bound_new, padjus
    
    def _conversor(self, coefs, pars0, padjus):
        pars = copy.deepcopy(pars0)
        for i, coef in enumerate(coefs):
            if 'beta' in padjus[i]:
                if '_M_' in padjus[i]:
                    indx = int(padjus[i].split('_')[-1])
                    pars['beta'][indx] = coef * pars['beta'][indx]
                else:
                    indx = padjus[i].split('_')
                    pars['beta'][int(indx[1])][int(indx[2]), int(indx[3])] = coef
            elif '_' in padjus[i]:
                name, indx = padjus[i].split('_')
                pars[name][int(indx)] = coef                    
            else:
                pars[padjus[i]] = coef
        return pars
        
    
    def objectiveFunction(self, coefs_list, stand_error=False, weights=None):
        errsq = np.zeros(coefs_list.shape[0])
        for i, coefs in enumerate(coefs_list):
            errs = self._residuals(coefs, stand_error, weights)
            errsq[i] = (errs*errs).mean()
        return errsq

    def _residuals(self, coefs, stand_error=False, weights=None):
        if type(weights) == type(None):
            weights = np.ones(len(self.Y))
        error_func = (lambda x: np.sqrt(x+1)) if stand_error else (lambda x:np.ones_like(x))
        errs = np.empty((0,))
        ts, mY = self._call_ODE(self.t, self._conversor(coefs, self.pars_init, self.padjus))
        for indY, indODE in enumerate(self.i_integ):
            if type(indODE) == list:
                temp = (self.N.reshape((1,-1)) *  mY[:,indODE]).sum(axis=1)
                errs = np.r_[errs, weights[indY] * ((self.Y[indY] - temp) / error_func(temp)) ]
            else:
                try:
                    errs = np.r_[errs, weights[indY] * ((self.Y[indY] - self.N[indODE%self.nages] *  mY[:,indODE]) / error_func(mY[:,indODE])) ]
                except:
                    print(self.t, self._conversor(coefs, self.pars_init, self.padjus))
                    raise
        errs = errs[~np.isnan(errs)]
        return errs
        
    def prepare_to_fit(self, data, pars, pars_to_fit, bound=None, nages=1, stand_error=False):
        self.pars_init = copy.deepcopy(pars)
        self.nages = nages
        self.i_integ, self.Y, self.t = self._prepare_input(data)
        self.bound, self.padjus = self._prepare_conversor(pars_to_fit, pars, bound)
        self.n_to_fit = len(self.padjus)
        
    
    def fit(self, data, pars, pars_to_fit, bound=None, nages=2, paramPSO=dict(),  stand_error=False):
        '''
        data: dictionary:
            t -> times
            X_N -> variable:
                X is the simbol of the parameter: S, E, Ia, Is, H, U, R, D, Nw
                N is the index of the age-group, starting on 0
        
        pars: dictionary, with the variable names as keys. 
        
        pars_to_fit: the name of the parameters to fits, if the parameter is a list,
        add _N with the index you want to if or _ALL to fit all
        the 'beta' parameter has 3 indexes: beta_I_J_K, with I indicating the
        which tcut it belongs and J_K indicating the position in the matrix.
        the beta also has a option 'beta_M_I' that fits a multiplicative
        constant of the infection matrix, without changing the relative weights
        (the _M_ and _ALL_ options are incompatible by now, and _M_ requires
        testing)
        
        bound = intervalo de limite para procura de cada parametro, onde None = sem limite
        
        bound => (lista_min_bound, lista_max_bound)
        '''
        paramPSO = self._fill_paramPSO(paramPSO)
        self.prepare_to_fit(data, pars, pars_to_fit, bound=bound, nages=nages, stand_error=stand_error)
        optimizer = ps.single.LocalBestPSO(n_particles=paramPSO['n_particles'], dimensions=self.n_to_fit, options=paramPSO['options'],bounds=self.bound)
        cost = pos = None
        cost, pos = optimizer.optimize(self.objectiveFunction,paramPSO['iter'],  stand_error=stand_error, n_processes=self.numeroProcessadores)
        self.pos = pos
        self.pars_opt = self._conversor(pos, self.pars_init, self.padjus )
        self.rmse = cost
        self.optimize = optimizer

    def fit_lsquares(self, data, pars, pars_to_fit, bound=None, nages=2,  stand_error=False, init=None, nrand=10):
        self.prepare_to_fit(data, pars, pars_to_fit, bound=bound, nages=nages, stand_error=stand_error)
        if init == None:
            cost_best = np.inf
            res_best = None
            #BUG: the parallel code does not work if PSO code had run previously
            if type(self.pos) != type(None) or self.numeroProcessadores == None or self.numeroProcessadores <= 1:
                for i in range(nrand):
                    print("{} / {}".format(i, nrand))
                    par0 = np.random.rand(self.n_to_fit)
                    par0 = self.bound[0] + par0 * (self.bound[1] - self.bound[0])
                    res = least_squares(self._residuals, par0, bounds=self.bound)
                    if res.cost < cost_best:
                        cost_best = res.cost
                        res_best = res
            else:
                par0 = np.random.rand(nrand, self.n_to_fit)
                par0 = self.bound[0].reshape((1,-1)) + par0 * (self.bound[1] - self.bound[0]).reshape((1,-1))
                f = lambda p0: least_squares(self._residuals, p0, bounds=self.bound)
                all_res = joblib.Parallel(n_jobs=self.numeroProcessadores)(joblib.delayed(f)(p0,) for p0 in par0)
                costs = np.array([res.cost for res in all_res])
                cost_best = all_res[costs.argmin()].cost
                res_best = all_res[costs.argmin()]
        else:
            res_best = least_squares(self._residuals, init, bounds=bound )
        self.pos_ls = res_best.x
        self.pars_opt_ls = self._conversor(res_best.x, self.pars_init, self.padjus )
        self.rmse_ls = (res_best.fun**2).mean()
        self.result_ls = res_best
        
    def predict(self, t=None, coefs=None, model_output=False):
        if type(t) == type(None):
            t = self.t
        if type(coefs) == type(None):
            coefs = self.pos
        elif type(coefs) == str and coefs  == 'LS':
            coefs = self.pos_ls
        ts, mY = self._call_ODE(t, self._conversor(coefs, self.pars_init, self.padjus))
        saida = np.zeros((len(ts), 0))
        for i in self.i_integ:
            if type(i) == list:
                ytemp = (mY[:,i] *self.N.reshape((1,-1))).sum(axis=1)
            else:
                ytemp = mY[:,i] * self.N[i%self.nages]
            saida = np.c_[saida, ytemp.reshape((-1,1))]
        
        if model_output:
            return ts, saida, mY
        else:
            return ts, saida
    
  

#ts, X = call_ODE(X0, tmax, betas, param, tcorte=tcorte)
#plt.plot(ts, X[:,:2], '.-')
