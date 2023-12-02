#pragma once

//structure of VGG16:      0,   1,       2,   3,       4,     5,   6,          7,   8,     9,          10,  11,  12,           13,  14,   15
int conv_Hin[]        = { 224, 224,    112, 112,      56,    56,  56,         28,  28,    28,         14,  14,  14,           7,    1,   1 };
int conv_Win[]        = { 224, 224,    112, 112,      56,    56,  56,         28,  28,    28,         14,  14,  14,           7,    1,   1 };
int conv_CHin[]       = { 3,  64,      64, 128,     128,   256, 256,        256, 512,   512,        512, 512, 512,         512, 4096,4096 };
int conv_CHout[]      = { 64,  64,     128, 128,     256,   256, 256,        512, 512,   512,        512, 512, 512,        4096, 4096,1000 };
int conv_Ky[]         = { 3,   3,       3,   3,       3,     3,   3,          3,   3,     3,          3,   3,   3,           7,    1,   1 };
int conv_Kx[]         = { 3,   3,       3,   3,       3,     3,   3,          3,   3,     3,          3,   3,   3,           7,    1,   1 };
int Sx[]              = { 1,   1,       1,   1,       1,     1,   1,          1,   1,     1,          1,   1,   1,           1,    1,   1 };
int Sy[]              = { 1,   1,       1,   1,       1,     1,   1,          1,   1,     1,          1,   1,   1,           1,    1,   1 };
int pad_left[]        = { 1,   1,       1,   1,       1,     1,   1,          1,   1,     1,          1,   1,   1,           0,    0,   0 };
int pad_right[]       = { 1,   1,       1,   1,       1,     1,   1,          1,   1,     1,          1,   1,   1,           0,    0,   0 };
int pad_up[]          = { 1,   1,       1,   1,       1,     1,   1,          1,   1,     1,          1,   1,   1,           0,    0,   0 };
int pad_down[]        = { 1,   1,       1,   1,       1,     1,   1,          1,   1,     1,          1,   1,   1,           0,    0,   0 };
int conv_Hout[]       = { 224, 224,     112, 112,      56,    56,  56,         28,  28,    28,         14,  14,  14,           1,    1,   1 };
int conv_Wout[]       = { 224, 224,     112, 112,      56,    56,  56,         28,  28,    28,         14,  14,  14,           1,    1,   1 };

int conv_bias_scale[] = { 5,   6,       7,   7,       7,     7,   7,          7,   7,     7,          7,   7,   3,           7,    6,   7 };

//structure of VGG16:     0,   1,       2,   3,       4,     5,   6,          7,   8,    9,          10,  11,  12,           13,   14,  15
//int conv_wt_scale[]   = { 8,  10,      10,  11,      11,     8,  12,         12,   8,   12,          12,  12,  12,           11,   14,  13 };
//int conv_in_scale[]   = {-1,  -2,      -3,  -3,      -4,    -8,  -4,         -3,  -7,   -3,          -2,  -1,   1,           -3,    4,   5 };
//int conv_out_scale[]  = {-2,  -3,      -3,  -4,      -8,    -4,  -3,         -7,  -3,   -2,          -1,   1,  -3,            4,    5,   4 };
//structure of VGG16:     0,   1,       2,   3,       4,     5,   6,          7,   8,    9,          10,  11,  12,           13,   14,  15
int conv_wt_scale[]   = { 0,  10,      10,  11,      11,     8,  12,         12,   8,   12,          12,  12,  12,           11,   14,  13 };
int conv_in_scale[]   = { 0,  -2,      -3,  -3,      -4,    -8,  -4,         -3,  -7,   -3,          -2,  -1,   1,           -3,    4,   5 };
int conv_out_scale[]  = { 0,  -3,      -3,  -4,      -8,    -4,  -3,         -7,  -3,   -2,          -1,   1,  -3,            4,    5,   4 };

int conv_in_bit[]     = { 8,   8,       8,   8,       8,     4,   8,          8,   4,    8,           8,   8,   8,           4,    8,   8 };
int conv_wt_bit[]     = { 8,   8,       8,   8,       8,     4,   8,          8,   4,    8,           8,   8,   8,           4,    8,   8 };
int conv_out_bit[]    = { 8,   8,        8,   8,       4,     8,   8,          4,   8,    8,           8,   8,   4,           8,    8,   8 };

int pool_in_bit[]     = {          8,              8,                       8,                     8,                  4 };
int pool_out_bit[]    = {          8,              8,                       8,                     8,                  4 };
int pool_in_scale[]   = {          -2,             -4,                      -3,                    -2,                 -3 };
int pool_out_scale[]  = {          -3,             -4,                      -3,                    -2,                 -3 };

int pool_Hin[]        = {          224,           112,                     56,                    28,                  14 };
int pool_Win[]        = {          224,           112,                     56,                    28,                  14 };
int pool_CHin[]       = {          64,            128,                     256,                   512,                 512 };
int pool_CHout[]      = {          64,            128,                     256,                   512,                 512 };
int pool_Ky[]         = {          2,              2,                       2,                     2,                   2 };
int pool_Kx[]         = {          2,              2,                       2,                     2,                   2 };
int pool_Sx[]         = {          2,              2,                       2,                     2,                   2 };
int pool_Sy[]         = {          2,              2,                       2,                     2,                   2 };
int pool_pad_left[]   = {          0,              0,                       0,                     0,                   0 };
int pool_pad_right[]  = {          0,              0,                       0,                     0,                   0 };
int pool_pad_up[]     = {          0,              0,                       0,                     0,                   0 };
int pool_pad_down[]   = {          0,              0,                       0,                     0,                   0 };
int pool_Hout[]       = {          112,            56,                      28,                    14,                  7 };
int pool_Wout[]       = {          112,            56,                      28,                    14,                  7 };

