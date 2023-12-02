#!/usr/bin/python3
from pynq import Overlay
from pynq import Xlnk
import numpy as np
import time
xlnk=Xlnk()

ol=Overlay("ZCU104_M32x32_mp_col_systolic_8Mb_150M_WNS-3.1.bit")
for i in ol.ip_dict:
    print(i);
ol.download();

#gpio=ol.axi_gpio_0
#gpio.write(0,0xF);
#
