<a href="https://colab.research.google.com/github/Ankur3107/large-scale-multi-label-classification/blob/master/large_scale_multilabelclassification.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>


```
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing
```


```
!wget -O datasets.zip http://nlp.cs.aueb.gr/software_and_datasets/EURLEX57K/datasets.zip

```

    --2020-08-23 11:37:34--  http://nlp.cs.aueb.gr/software_and_datasets/EURLEX57K/datasets.zip
    Resolving nlp.cs.aueb.gr (nlp.cs.aueb.gr)... 195.251.248.252
    Connecting to nlp.cs.aueb.gr (nlp.cs.aueb.gr)|195.251.248.252|:80... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 135996905 (130M) [application/zip]
    Saving to: ‘datasets.zip’
    
    datasets.zip        100%[===================>] 129.70M  2.29MB/s    in 2m 12s  
    
    2020-08-23 11:39:48 (1006 KB/s) - ‘datasets.zip’ saved [135996905/135996905]
    



```
!unzip datasets.zip -d EURLEX57K
```

    [1;30;43mStreaming output truncated to the last 5000 lines.[0m
      inflating: EURLEX57K/dataset/dev/32004R1662.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32004R1662.json  
      inflating: EURLEX57K/dataset/dev/32004R0970.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32004R0970.json  
      inflating: EURLEX57K/dataset/dev/31999R0014.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31999R0014.json  
      inflating: EURLEX57K/dataset/dev/32007R1267.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32007R1267.json  
      inflating: EURLEX57K/dataset/dev/31987R3625.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31987R3625.json  
      inflating: EURLEX57K/dataset/dev/31999R2183.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31999R2183.json  
      inflating: EURLEX57K/dataset/dev/32012R0997.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32012R0997.json  
      inflating: EURLEX57K/dataset/dev/31994R2926.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31994R2926.json  
      inflating: EURLEX57K/dataset/dev/31986R1595.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31986R1595.json  
      inflating: EURLEX57K/dataset/dev/32001R1381.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32001R1381.json  
      inflating: EURLEX57K/dataset/dev/32011R1280.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32011R1280.json  
      inflating: EURLEX57K/dataset/dev/32007R1322.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32007R1322.json  
      inflating: EURLEX57K/dataset/dev/32005R0254.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32005R0254.json  
      inflating: EURLEX57K/dataset/dev/32004R0136.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32004R0136.json  
      inflating: EURLEX57K/dataset/dev/31993D0332.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31993D0332.json  
      inflating: EURLEX57K/dataset/dev/32010R0209.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32010R0209.json  
      inflating: EURLEX57K/dataset/dev/31990R1964.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31990R1964.json  
      inflating: EURLEX57K/dataset/dev/31991D0005.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31991D0005.json  
      inflating: EURLEX57K/dataset/dev/31995R1838.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31995R1838.json  
      inflating: EURLEX57K/dataset/dev/31992D0050.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31992D0050.json  
      inflating: EURLEX57K/dataset/dev/31989D0668.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31989D0668.json  
      inflating: EURLEX57K/dataset/dev/31989R0379.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31989R0379.json  
      inflating: EURLEX57K/dataset/dev/31994D1031.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31994D1031.json  
      inflating: EURLEX57K/dataset/dev/32001R1552.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32001R1552.json  
      inflating: EURLEX57K/dataset/dev/31996R0905.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31996R0905.json  
      inflating: EURLEX57K/dataset/dev/32008D0876.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32008D0876.json  
      inflating: EURLEX57K/dataset/dev/32001D0202.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32001D0202.json  
      inflating: EURLEX57K/dataset/dev/32006R0128.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32006R0128.json  
      inflating: EURLEX57K/dataset/dev/31994D0620.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31994D0620.json  
      inflating: EURLEX57K/dataset/dev/32001D0652.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32001D0652.json  
      inflating: EURLEX57K/dataset/dev/32005R0487.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32005R0487.json  
      inflating: EURLEX57K/dataset/dev/31987R1927.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31987R1927.json  
      inflating: EURLEX57K/dataset/dev/31996D0517.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31996D0517.json  
      inflating: EURLEX57K/dataset/dev/32003R0074.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32003R0074.json  
      inflating: EURLEX57K/dataset/dev/31982D0297.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31982D0297.json  
      inflating: EURLEX57K/dataset/dev/32010R0065.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32010R0065.json  
      inflating: EURLEX57K/dataset/dev/32015D0078.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32015D0078.json  
      inflating: EURLEX57K/dataset/dev/31985R2180.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31985R2180.json  
      inflating: EURLEX57K/dataset/dev/31994R2249.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31994R2249.json  
      inflating: EURLEX57K/dataset/dev/32013D0171.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32013D0171.json  
      inflating: EURLEX57K/dataset/dev/31993R2022.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31993R2022.json  
      inflating: EURLEX57K/dataset/dev/32005R0891.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32005R0891.json  
      inflating: EURLEX57K/dataset/dev/32003L0126.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32003L0126.json  
      inflating: EURLEX57K/dataset/dev/32011R1146.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32011R1146.json  
      inflating: EURLEX57K/dataset/dev/32006R1386.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32006R1386.json  
      inflating: EURLEX57K/dataset/dev/31989D0490.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31989D0490.json  
      inflating: EURLEX57K/dataset/dev/32001R1250.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32001R1250.json  
      inflating: EURLEX57K/dataset/dev/31985R1942.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31985R1942.json  
      inflating: EURLEX57K/dataset/dev/31979R2968.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31979R2968.json  
      inflating: EURLEX57K/dataset/dev/31987D0076.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31987D0076.json  
      inflating: EURLEX57K/dataset/dev/31996R0354.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31996R0354.json  
      inflating: EURLEX57K/dataset/dev/31988R3565.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31988R3565.json  
      inflating: EURLEX57K/dataset/dev/31986R0255.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31986R0255.json  
      inflating: EURLEX57K/dataset/dev/31998R1009.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31998R1009.json  
      inflating: EURLEX57K/dataset/dev/32011D0114.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32011D0114.json  
      inflating: EURLEX57K/dataset/dev/31995R2783.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31995R2783.json  
      inflating: EURLEX57K/dataset/dev/31989R2850.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31989R2850.json  
      inflating: EURLEX57K/dataset/dev/32002R0551.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32002R0551.json  
      inflating: EURLEX57K/dataset/dev/32014R0109.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32014R0109.json  
      inflating: EURLEX57K/dataset/dev/31970D0325.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31970D0325.json  
      inflating: EURLEX57K/dataset/dev/32013D0629(02).json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32013D0629(02).json  
      inflating: EURLEX57K/dataset/dev/32011L0042.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32011L0042.json  
      inflating: EURLEX57K/dataset/dev/32009D0450.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32009D0450.json  
      inflating: EURLEX57K/dataset/dev/32008R0223.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32008R0223.json  
      inflating: EURLEX57K/dataset/dev/32010R1176.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32010R1176.json  
      inflating: EURLEX57K/dataset/dev/32013R1089.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32013R1089.json  
      inflating: EURLEX57K/dataset/dev/32009R1029.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32009R1029.json  
      inflating: EURLEX57K/dataset/dev/31996D0079.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31996D0079.json  
      inflating: EURLEX57K/dataset/dev/32004R1863.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32004R1863.json  
      inflating: EURLEX57K/dataset/dev/32014R0220.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32014R0220.json  
      inflating: EURLEX57K/dataset/dev/32002R0678.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32002R0678.json  
      inflating: EURLEX57K/dataset/dev/31994R2662.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31994R2662.json  
      inflating: EURLEX57K/dataset/dev/32007D0665.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32007D0665.json  
      inflating: EURLEX57K/dataset/dev/31993R2059.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31993R2059.json  
      inflating: EURLEX57K/dataset/dev/32002R0228.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32002R0228.json  
      inflating: EURLEX57K/dataset/dev/31976D0806.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31976D0806.json  
      inflating: EURLEX57K/dataset/dev/31988R2118.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31988R2118.json  
      inflating: EURLEX57K/dataset/dev/31994R3073.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31994R3073.json  
      inflating: EURLEX57K/dataset/dev/32006L0001.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32006L0001.json  
      inflating: EURLEX57K/dataset/dev/32003R0849.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32003R0849.json  
      inflating: EURLEX57K/dataset/dev/32006R1607.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32006R1607.json  
      inflating: EURLEX57K/dataset/dev/31981R1013.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31981R1013.json  
      inflating: EURLEX57K/dataset/dev/31985R1738.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31985R1738.json  
      inflating: EURLEX57K/dataset/dev/32004R1025.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32004R1025.json  
      inflating: EURLEX57K/dataset/dev/31988D0121.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31988D0121.json  
      inflating: EURLEX57K/dataset/dev/31996R1396.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31996R1396.json  
      inflating: EURLEX57K/dataset/dev/32002R1086.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32002R1086.json  
      inflating: EURLEX57K/dataset/dev/32001D0279.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32001D0279.json  
      inflating: EURLEX57K/dataset/dev/32003R2198.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32003R2198.json  
      inflating: EURLEX57K/dataset/dev/32007R0661.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32007R0661.json  
      inflating: EURLEX57K/dataset/dev/32014D0224.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32014D0224.json  
      inflating: EURLEX57K/dataset/dev/31975R0154.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31975R0154.json  
      inflating: EURLEX57K/dataset/dev/31996D0242.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31996D0242.json  
      inflating: EURLEX57K/dataset/dev/32009D0811.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32009D0811.json  
      inflating: EURLEX57K/dataset/dev/32001D0107.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32001D0107.json  
      inflating: EURLEX57K/dataset/dev/31987R0530.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31987R0530.json  
      inflating: EURLEX57K/dataset/dev/32002D0502.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32002D0502.json  
      inflating: EURLEX57K/dataset/dev/31991D0183.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31991D0183.json  
      inflating: EURLEX57K/dataset/dev/31997D0570.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31997D0570.json  
      inflating: EURLEX57K/dataset/dev/32008R0761.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32008R0761.json  
      inflating: EURLEX57K/dataset/dev/32001R1207.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32001R1207.json  
      inflating: EURLEX57K/dataset/dev/31999D0196.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31999D0196.json  
      inflating: EURLEX57K/dataset/dev/31995R3080.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31995R3080.json  
      inflating: EURLEX57K/dataset/dev/32008D0620.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32008D0620.json  
      inflating: EURLEX57K/dataset/dev/31994R0937.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31994R0937.json  
      inflating: EURLEX57K/dataset/dev/32008R0624.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32008R0624.json  
      inflating: EURLEX57K/dataset/dev/32005R1486.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32005R1486.json  
      inflating: EURLEX57K/dataset/dev/32000R1470.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32000R1470.json  
      inflating: EURLEX57K/dataset/dev/31998R2367.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31998R2367.json  
      inflating: EURLEX57K/dataset/dev/32002D0017.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32002D0017.json  
      inflating: EURLEX57K/dataset/dev/32005R0297.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32005R0297.json  
      inflating: EURLEX57K/dataset/dev/31985R1850.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31985R1850.json  
      inflating: EURLEX57K/dataset/dev/31985D0253.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31985D0253.json  
      inflating: EURLEX57K/dataset/dev/31992D0093.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31992D0093.json  
      inflating: EURLEX57K/dataset/dev/32003R2209.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32003R2209.json  
      inflating: EURLEX57K/dataset/dev/32006R0942.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32006R0942.json  
      inflating: EURLEX57K/dataset/dev/32007R1532.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32007R1532.json  
      inflating: EURLEX57K/dataset/dev/31981R3583.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31981R3583.json  
      inflating: EURLEX57K/dataset/dev/32009R0395.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32009R0395.json  
      inflating: EURLEX57K/dataset/dev/31988D0199.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31988D0199.json  
      inflating: EURLEX57K/dataset/dev/32001R2217.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32001R2217.json  
      inflating: EURLEX57K/dataset/dev/32005R0014.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32005R0014.json  
      inflating: EURLEX57K/dataset/dev/32010R0419.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32010R0419.json  
      inflating: EURLEX57K/dataset/dev/32011R0281.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32011R0281.json  
      inflating: EURLEX57K/dataset/dev/32006R0411.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32006R0411.json  
      inflating: EURLEX57K/dataset/dev/32015R0115.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32015R0115.json  
      inflating: EURLEX57K/dataset/dev/32007R0266.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32007R0266.json  
      inflating: EURLEX57K/dataset/dev/31988R4074.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31988R4074.json  
      inflating: EURLEX57K/dataset/dev/31989L0342.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31989L0342.json  
      inflating: EURLEX57K/dataset/dev/31992R2229.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31992R2229.json  
      inflating: EURLEX57K/dataset/dev/32005R0151.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32005R0151.json  
      inflating: EURLEX57K/dataset/dev/31981R0710.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31981R0710.json  
      inflating: EURLEX57K/dataset/dev/32009R1091.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32009R1091.json  
      inflating: EURLEX57K/dataset/dev/32002R2307.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32002R2307.json  
      inflating: EURLEX57K/dataset/dev/31991R2786.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31991R2786.json  
      inflating: EURLEX57K/dataset/dev/32004R1971.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32004R1971.json  
      inflating: EURLEX57K/dataset/dev/31990R0573.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31990R0573.json  
      inflating: EURLEX57K/dataset/dev/31993R2848.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31993R2848.json  
      inflating: EURLEX57K/dataset/dev/32006R1715.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32006R1715.json  
      inflating: EURLEX57K/dataset/dev/32008R0018.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32008R0018.json  
      inflating: EURLEX57K/dataset/dev/32001R1641.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32001R1641.json  
      inflating: EURLEX57K/dataset/dev/32013R1027.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32013R1027.json  
      inflating: EURLEX57K/dataset/dev/31986R3592.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31986R3592.json  
      inflating: EURLEX57K/dataset/dev/32011R1310.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32011R1310.json  
      inflating: EURLEX57K/dataset/dev/32001R1211.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32001R1211.json  
      inflating: EURLEX57K/dataset/dev/31994L0035.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31994L0035.json  
      inflating: EURLEX57K/dataset/dev/32012R0807.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32012R0807.json  
      inflating: EURLEX57K/dataset/dev/31996R1154.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31996R1154.json  
      inflating: EURLEX57K/dataset/dev/32001D0541.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32001D0541.json  
      inflating: EURLEX57K/dataset/dev/32013D0327.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32013D0327.json  
      inflating: EURLEX57K/dataset/dev/32002R0005.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32002R0005.json  
      inflating: EURLEX57K/dataset/dev/32002R2038.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32002R2038.json  
      inflating: EURLEX57K/dataset/dev/32010R0399.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32010R0399.json  
      inflating: EURLEX57K/dataset/dev/32004R2131.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32004R2131.json  
      inflating: EURLEX57K/dataset/dev/31998R1048.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31998R1048.json  
      inflating: EURLEX57K/dataset/dev/32011D0440.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32011D0440.json  
      inflating: EURLEX57K/dataset/dev/32003R0222.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32003R0222.json  
      inflating: EURLEX57K/dataset/dev/32010R0726.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32010R0726.json  
      inflating: EURLEX57K/dataset/dev/32001R2128.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32001R2128.json  
      inflating: EURLEX57K/dataset/dev/32010D0237.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32010D0237.json  
      inflating: EURLEX57K/dataset/dev/31995D0344.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31995D0344.json  
      inflating: EURLEX57K/dataset/dev/32011R0014.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32011R0014.json  
      inflating: EURLEX57K/dataset/dev/31997D0073.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31997D0073.json  
      inflating: EURLEX57K/dataset/dev/31990R0709.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31990R0709.json  
      inflating: EURLEX57K/dataset/dev/32009D0411.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32009D0411.json  
      inflating: EURLEX57K/dataset/dev/32001R1704.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32001R1704.json  
      inflating: EURLEX57K/dataset/dev/32014R1309.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32014R1309.json  
      inflating: EURLEX57K/dataset/dev/32009D0041.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32009D0041.json  
      inflating: EURLEX57K/dataset/dev/31984R0965.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31984R0965.json  
      inflating: EURLEX57K/dataset/dev/31999D0495.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31999D0495.json  
      inflating: EURLEX57K/dataset/dev/31996R2514.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31996R2514.json  
      inflating: EURLEX57K/dataset/dev/31992R3838.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31992R3838.json  
      inflating: EURLEX57K/dataset/dev/32000R2533.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32000R2533.json  
      inflating: EURLEX57K/dataset/dev/31986R0582.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31986R0582.json  
      inflating: EURLEX57K/dataset/dev/32012R0368.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32012R0368.json  
      inflating: EURLEX57K/dataset/dev/32012D0383.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32012D0383.json  
      inflating: EURLEX57K/dataset/dev/32004R0360.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32004R0360.json  
      inflating: EURLEX57K/dataset/dev/32002R2254.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32002R2254.json  
      inflating: EURLEX57K/dataset/dev/32000R2499.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32000R2499.json  
      inflating: EURLEX57K/dataset/dev/31993R0025.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31993R0025.json  
      inflating: EURLEX57K/dataset/dev/32014R0631.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32014R0631.json  
      inflating: EURLEX57K/dataset/dev/31986R2803.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31986R2803.json  
      inflating: EURLEX57K/dataset/dev/31999D0753.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31999D0753.json  
      inflating: EURLEX57K/dataset/dev/31988R0471.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31988R0471.json  
      inflating: EURLEX57K/dataset/dev/31992R1413.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31992R1413.json  
      inflating: EURLEX57K/dataset/dev/31996D0487.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31996D0487.json  
      inflating: EURLEX57K/dataset/dev/32005D0006.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32005D0006.json  
      inflating: EURLEX57K/dataset/dev/32011R0278.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32011R0278.json  
      inflating: EURLEX57K/dataset/dev/31992R1940.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31992R1940.json  
      inflating: EURLEX57K/dataset/dev/31998R1361.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31998R1361.json  
      inflating: EURLEX57K/dataset/dev/31995D0128.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31995D0128.json  
      inflating: EURLEX57K/dataset/dev/32006R0112.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32006R0112.json  
      inflating: EURLEX57K/dataset/dev/31988R1260.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31988R1260.json  
      inflating: EURLEX57K/dataset/dev/32001R2344.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32001R2344.json  
      inflating: EURLEX57K/dataset/dev/32001R1991.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32001R1991.json  
      inflating: EURLEX57K/dataset/dev/32010R0264.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32010R0264.json  
      inflating: EURLEX57K/dataset/dev/31978L0765.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31978L0765.json  
      inflating: EURLEX57K/dataset/dev/32000R2358.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32000R2358.json  
      inflating: EURLEX57K/dataset/dev/32011D0047.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32011D0047.json  
      inflating: EURLEX57K/dataset/dev/32004D0919.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32004D0919.json  
      inflating: EURLEX57K/dataset/dev/31994R3209.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31994R3209.json  
      inflating: EURLEX57K/dataset/dev/32009R0012.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32009R0012.json  
      inflating: EURLEX57K/dataset/dev/32008D0661.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32008D0661.json  
      inflating: EURLEX57K/dataset/dev/31979D0044.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31979D0044.json  
      inflating: EURLEX57K/dataset/dev/32012R0915.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32012R0915.json  
      inflating: EURLEX57K/dataset/dev/31991R1387.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31991R1387.json  
      inflating: EURLEX57K/dataset/dev/32009D0446.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32009D0446.json  
      inflating: EURLEX57K/dataset/dev/31991R3640.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31991R3640.json  
      inflating: EURLEX57K/dataset/dev/32003R1464.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32003R1464.json  
      inflating: EURLEX57K/dataset/dev/32000R1431.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32000R1431.json  
      inflating: EURLEX57K/dataset/dev/32005R1097.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32005R1097.json  
      inflating: EURLEX57K/dataset/dev/31982D0429.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31982D0429.json  
      inflating: EURLEX57K/dataset/dev/32006D0268.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32006D0268.json  
      inflating: EURLEX57K/dataset/dev/31986R3879.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31986R3879.json  
      inflating: EURLEX57K/dataset/dev/31991D0087.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31991D0087.json  
      inflating: EURLEX57K/dataset/dev/32009R0854.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32009R0854.json  
      inflating: EURLEX57K/dataset/dev/32001R2485.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32001R2485.json  
      inflating: EURLEX57K/dataset/dev/31992D0528.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31992D0528.json  
      inflating: EURLEX57K/dataset/dev/31979R2211.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31979R2211.json  
      inflating: EURLEX57K/dataset/dev/31996L0093.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31996L0093.json  
      inflating: EURLEX57K/dataset/dev/31993R1663.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31993R1663.json  
      inflating: EURLEX57K/dataset/dev/31999R0350.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31999R0350.json  
      inflating: EURLEX57K/dataset/dev/32001R1180.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32001R1180.json  
      inflating: EURLEX57K/dataset/dev/31998R0032.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31998R0032.json  
      inflating: EURLEX57K/dataset/dev/32004R1526.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32004R1526.json  
      inflating: EURLEX57K/dataset/dev/32007R1573.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32007R1573.json  
      inflating: EURLEX57K/dataset/dev/32009R1195.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32009R1195.json  
      inflating: EURLEX57K/dataset/dev/32007R0698.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32007R0698.json  
      inflating: EURLEX57K/dataset/dev/31982R1953.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31982R1953.json  
      inflating: EURLEX57K/dataset/dev/32012D0706(02).json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32012D0706(02).json  
      inflating: EURLEX57K/dataset/dev/31985R2047.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31985R2047.json  
      inflating: EURLEX57K/dataset/dev/31988R0830.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31988R0830.json  
      inflating: EURLEX57K/dataset/dev/32014D0727.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32014D0727.json  
      inflating: EURLEX57K/dataset/dev/32002R1993.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32002R1993.json  
      inflating: EURLEX57K/dataset/dev/32003R0019.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32003R0019.json  
      inflating: EURLEX57K/dataset/dev/31992R0605.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31992R0605.json  
      inflating: EURLEX57K/dataset/dev/32014R0373.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32014R0373.json  
      inflating: EURLEX57K/dataset/dev/32005R0540.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32005R0540.json  
      inflating: EURLEX57K/dataset/dev/32008R0059.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32008R0059.json  
      inflating: EURLEX57K/dataset/dev/32006R0846.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32006R0846.json  
      inflating: EURLEX57K/dataset/dev/31985L0578.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31985L0578.json  
      inflating: EURLEX57K/dataset/dev/32014R1076.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32014R1076.json  
      inflating: EURLEX57K/dataset/dev/32007R1088.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32007R1088.json  
      inflating: EURLEX57K/dataset/dev/31995R0879.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31995R0879.json  
      inflating: EURLEX57K/dataset/dev/31993R1232.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31993R1232.json  
      inflating: EURLEX57K/dataset/dev/32004R1177.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32004R1177.json  
      inflating: EURLEX57K/dataset/dev/32007R0699.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32007R0699.json  
      inflating: EURLEX57K/dataset/dev/32007R0363.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32007R0363.json  
      inflating: EURLEX57K/dataset/dev/32001R1882.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32001R1882.json  
      inflating: EURLEX57K/dataset/dev/31996D0594.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31996D0594.json  
      inflating: EURLEX57K/dataset/dev/31977D0270.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31977D0270.json  
      inflating: EURLEX57K/dataset/dev/32000D0049.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32000D0049.json  
      inflating: EURLEX57K/dataset/dev/31983D0176.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31983D0176.json  
      inflating: EURLEX57K/dataset/dev/31988R1236.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31988R1236.json  
      inflating: EURLEX57K/dataset/dev/31992R2793.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31992R2793.json  
      inflating: EURLEX57K/dataset/dev/31996R2407.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31996R2407.json  
      inflating: EURLEX57K/dataset/dev/32004D0332.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32004D0332.json  
      inflating: EURLEX57K/dataset/dev/31993D0077.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31993D0077.json  
      inflating: EURLEX57K/dataset/dev/32013R1358.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32013R1358.json  
      inflating: EURLEX57K/dataset/dev/32014R1099.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32014R1099.json  
      inflating: EURLEX57K/dataset/dev/31986R0868.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31986R0868.json  
      inflating: EURLEX57K/dataset/dev/32003R0761.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32003R0761.json  
      inflating: EURLEX57K/dataset/dev/32010R0265.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32010R0265.json  
      inflating: EURLEX57K/dataset/dev/32000R2359.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32000R2359.json  
      inflating: EURLEX57K/dataset/dev/32015D0278.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32015D0278.json  
      inflating: EURLEX57K/dataset/dev/31981D0492.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31981D0492.json  
      inflating: EURLEX57K/dataset/dev/32013D0371.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32013D0371.json  
      inflating: EURLEX57K/dataset/dev/32010R0635.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32010R0635.json  
      inflating: EURLEX57K/dataset/dev/32012R0851.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32012R0851.json  
      inflating: EURLEX57K/dataset/dev/32001L0011.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32001L0011.json  
      inflating: EURLEX57K/dataset/dev/31998R2262.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31998R2262.json  
      inflating: EURLEX57K/dataset/dev/31995R1507.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31995R1507.json  
      inflating: EURLEX57K/dataset/dev/32006R1186.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32006R1186.json  
      inflating: EURLEX57K/dataset/dev/31987R1674.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31987R1674.json  
      inflating: EURLEX57K/dataset/dev/32008D0725.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32008D0725.json  
      inflating: EURLEX57K/dataset/dev/31989D0038.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31989D0038.json  
      inflating: EURLEX57K/dataset/dev/31994D0070.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31994D0070.json  
      inflating: EURLEX57K/dataset/dev/32006D0024(01).json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32006D0024(01).json  
      inflating: EURLEX57K/dataset/dev/32004R2022.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32004R2022.json  
      inflating: EURLEX57K/dataset/dev/31980D0461.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31980D0461.json  
      inflating: EURLEX57K/dataset/dev/32004R1823.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32004R1823.json  
      inflating: EURLEX57K/dataset/dev/32000R2532.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32000R2532.json  
      inflating: EURLEX57K/dataset/dev/31987R2326.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31987R2326.json  
      inflating: EURLEX57K/dataset/dev/31990D0130.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31990D0130.json  
      inflating: EURLEX57K/dataset/dev/31994R2622.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31994R2622.json  
      inflating: EURLEX57K/dataset/dev/32014R0630.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32014R0630.json  
      inflating: EURLEX57K/dataset/dev/31990D0560.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31990D0560.json  
      inflating: EURLEX57K/dataset/dev/32008R0849.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32008R0849.json  
      inflating: EURLEX57K/dataset/dev/31986L0594.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31986L0594.json  
      inflating: EURLEX57K/dataset/dev/31992R3090.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31992R3090.json  
      inflating: EURLEX57K/dataset/dev/31993R1320.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31993R1320.json  
      inflating: EURLEX57K/dataset/dev/31999D0302.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31999D0302.json  
      inflating: EURLEX57K/dataset/dev/32002R1496.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32002R1496.json  
      inflating: EURLEX57K/dataset/dev/31998D0060.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31998D0060.json  
      inflating: EURLEX57K/dataset/dev/32005R1757.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32005R1757.json  
      inflating: EURLEX57K/dataset/dev/31998R0988.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31998R0988.json  
      inflating: EURLEX57K/dataset/dev/31991D0317.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31991D0317.json  
      inflating: EURLEX57K/dataset/dev/31987R2263.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31987R2263.json  
      inflating: EURLEX57K/dataset/dev/32015R0047.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32015R0047.json  
      inflating: EURLEX57K/dataset/dev/31982R0752.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31982R0752.json  
      inflating: EURLEX57K/dataset/dev/31999R1452.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31999R1452.json  
      inflating: EURLEX57K/dataset/dev/31985D0028.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31985D0028.json  
      inflating: EURLEX57K/dataset/dev/31986R0096.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31986R0096.json  
      inflating: EURLEX57K/dataset/dev/31988R4063.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31988R4063.json  
      inflating: EURLEX57K/dataset/dev/32011R0629.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32011R0629.json  
      inflating: EURLEX57K/dataset/dev/31993R3464.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31993R3464.json  
      inflating: EURLEX57K/dataset/dev/31998R2235.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31998R2235.json  
      inflating: EURLEX57K/dataset/dev/31995R1550.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31995R1550.json  
      inflating: EURLEX57K/dataset/dev/31994D0861.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31994D0861.json  
      inflating: EURLEX57K/dataset/dev/31989D0080.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31989D0080.json  
      inflating: EURLEX57K/dataset/dev/31997D0834.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31997D0834.json  
      inflating: EURLEX57K/dataset/dev/31984R1733.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31984R1733.json  
      inflating: EURLEX57K/dataset/dev/32006R1581.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32006R1581.json  
      inflating: EURLEX57K/dataset/dev/31984R0088.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31984R0088.json  
      inflating: EURLEX57K/dataset/dev/32013R0637.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32013R0637.json  
      inflating: EURLEX57K/dataset/dev/31997D0567.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31997D0567.json  
      inflating: EURLEX57K/dataset/dev/32004R2130.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32004R2130.json  
      inflating: EURLEX57K/dataset/dev/31971D0142.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31971D0142.json  
      inflating: EURLEX57K/dataset/dev/32009D0806.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32009D0806.json  
      inflating: EURLEX57K/dataset/dev/32009R0802.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32009R0802.json  
      inflating: EURLEX57K/dataset/dev/32010R0727.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32010R0727.json  
      inflating: EURLEX57K/dataset/dev/32010D0236.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32010D0236.json  
      inflating: EURLEX57K/dataset/dev/31987D0523.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31987D0523.json  
      inflating: EURLEX57K/dataset/dev/31995D0345.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31995D0345.json  
      inflating: EURLEX57K/dataset/dev/32003R1961.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32003R1961.json  
      inflating: EURLEX57K/dataset/dev/32009D0410.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32009D0410.json  
      inflating: EURLEX57K/dataset/dev/31990R3574.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31990R3574.json  
      inflating: EURLEX57K/dataset/dev/31992R3213.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31992R3213.json  
      inflating: EURLEX57K/dataset/dev/31991R3246.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31991R3246.json  
      inflating: EURLEX57K/dataset/dev/32003R1062.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32003R1062.json  
      inflating: EURLEX57K/dataset/dev/31984R0964.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31984R0964.json  
      inflating: EURLEX57K/dataset/dev/32008D0322.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32008D0322.json  
      inflating: EURLEX57K/dataset/dev/32002R1300.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32002R1300.json  
      inflating: EURLEX57K/dataset/dev/31999D0601.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31999D0601.json  
      inflating: EURLEX57K/dataset/dev/31998D0099.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31998D0099.json  
      inflating: EURLEX57K/dataset/dev/31995R2805.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31995R2805.json  
      inflating: EURLEX57K/dataset/dev/31994R1018.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31994R1018.json  
      inflating: EURLEX57K/dataset/dev/32012R1094.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32012R1094.json  
      inflating: EURLEX57K/dataset/dev/31983R1722.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31983R1722.json  
      inflating: EURLEX57K/dataset/dev/31981R2669.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31981R2669.json  
      inflating: EURLEX57K/dataset/dev/31994R2264.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31994R2264.json  
      inflating: EURLEX57K/dataset/dev/32002R2243.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32002R2243.json  
      inflating: EURLEX57K/dataset/dev/31996R2153.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31996R2153.json  
      inflating: EURLEX57K/dataset/dev/32002R0784.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32002R0784.json  
      inflating: EURLEX57K/dataset/dev/31991D0614.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31991D0614.json  
      inflating: EURLEX57K/dataset/dev/31978L0549.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31978L0549.json  
      inflating: EURLEX57K/dataset/dev/31994R0609.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31994R0609.json  
      inflating: EURLEX57K/dataset/dev/32005D0154.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32005D0154.json  
      inflating: EURLEX57K/dataset/dev/32015R0114.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32015R0114.json  
      inflating: EURLEX57K/dataset/dev/32014D0788.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32014D0788.json  
      inflating: EURLEX57K/dataset/dev/32003R2064.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32003R2064.json  
      inflating: EURLEX57K/dataset/dev/32002R2306.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32002R2306.json  
      inflating: EURLEX57K/dataset/dev/32014D0272.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32014D0272.json  
      inflating: EURLEX57K/dataset/dev/32015R0051.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32015R0051.json  
      inflating: EURLEX57K/dataset/dev/31992D0354.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31992D0354.json  
      inflating: EURLEX57K/dataset/dev/32008R1258.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32008R1258.json  
      inflating: EURLEX57K/dataset/dev/32005R0500.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32005R0500.json  
      inflating: EURLEX57K/dataset/dev/32004D0689.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32004D0689.json  
      inflating: EURLEX57K/dataset/dev/32008D0029(01).json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32008D0029(01).json  
      inflating: EURLEX57K/dataset/dev/31997D0208.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31997D0208.json  
      inflating: EURLEX57K/dataset/dev/32001R1085.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32001R1085.json  
      inflating: EURLEX57K/dataset/dev/31996R0928.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31996R0928.json  
      inflating: EURLEX57K/dataset/dev/32007R1476.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32007R1476.json  
      inflating: EURLEX57K/dataset/dev/31993R2849.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31993R2849.json  
      inflating: EURLEX57K/dataset/dev/32002R1480.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32002R1480.json  
      inflating: EURLEX57K/dataset/dev/32006R0386.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32006R0386.json  
      inflating: EURLEX57K/dataset/dev/32012D0402.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32012D0402.json  
      inflating: EURLEX57K/dataset/dev/31984R0134.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31984R0134.json  
      inflating: EURLEX57K/dataset/dev/31997D0121.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31997D0121.json  
      inflating: EURLEX57K/dataset/dev/31991D0182.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31991D0182.json  
      inflating: EURLEX57K/dataset/dev/31994D0524.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31994D0524.json  
      inflating: EURLEX57K/dataset/dev/32003R1131.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32003R1131.json  
      inflating: EURLEX57K/dataset/dev/31991R1328.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31991R1328.json  
      inflating: EURLEX57K/dataset/dev/32005R1038.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32005R1038.json  
      inflating: EURLEX57K/dataset/dev/31995R1546.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31995R1546.json  
      inflating: EURLEX57K/dataset/dev/32005R1192.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32005R1192.json  
      inflating: EURLEX57K/dataset/dev/31994R0936.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31994R0936.json  
      inflating: EURLEX57K/dataset/dev/31979D0511.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31979D0511.json  
      inflating: EURLEX57K/dataset/dev/31988R2267.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31988R2267.json  
      inflating: EURLEX57K/dataset/dev/32003R1074.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32003R1074.json  
      inflating: EURLEX57K/dataset/dev/31987R4067.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31987R4067.json  
      inflating: EURLEX57K/dataset/dev/32002R0854.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32002R0854.json  
      inflating: EURLEX57K/dataset/dev/32013R1175.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32013R1175.json  
      inflating: EURLEX57K/dataset/dev/31989R2555.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31989R2555.json  
      inflating: EURLEX57K/dataset/dev/32009L0100.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32009L0100.json  
      inflating: EURLEX57K/dataset/dev/32003R0665.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32003R0665.json  
      inflating: EURLEX57K/dataset/dev/31989R3714.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31989R3714.json  
      inflating: EURLEX57K/dataset/dev/31985R0743.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31985R0743.json  
      inflating: EURLEX57K/dataset/dev/32012D0117.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32012D0117.json  
      inflating: EURLEX57K/dataset/dev/31984D0130.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31984D0130.json  
      inflating: EURLEX57K/dataset/dev/32012R0406.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32012R0406.json  
      inflating: EURLEX57K/dataset/dev/31988R4219.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31988R4219.json  
      inflating: EURLEX57K/dataset/dev/31998R1264.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31998R1264.json  
      inflating: EURLEX57K/dataset/dev/31985R0597.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31985R0597.json  
      inflating: EURLEX57K/dataset/dev/31992D0616.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31992D0616.json  
      inflating: EURLEX57K/dataset/dev/32002R2214.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32002R2214.json  
      inflating: EURLEX57K/dataset/dev/31982R1944.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31982R1944.json  
      inflating: EURLEX57K/dataset/dev/31999R1556.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31999R1556.json  
      inflating: EURLEX57K/dataset/dev/32007R1134.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32007R1134.json  
      inflating: EURLEX57K/dataset/dev/32009R0269.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32009R0269.json  
      inflating: EURLEX57K/dataset/dev/32010R0859.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32010R0859.json  
      inflating: EURLEX57K/dataset/dev/31998D0021.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31998D0021.json  
      inflating: EURLEX57K/dataset/dev/32004R1474.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32004R1474.json  
      inflating: EURLEX57K/dataset/dev/32011R1079.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32011R1079.json  
      inflating: EURLEX57K/dataset/dev/32002R1087.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32002R1087.json  
      inflating: EURLEX57K/dataset/dev/32007R1421.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32007R1421.json  
      inflating: EURLEX57K/dataset/dev/31991R2380.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31991R2380.json  
      inflating: EURLEX57K/dataset/dev/32012R0797.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32012R0797.json  
      inflating: EURLEX57K/dataset/dev/31986R0487.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31986R0487.json  
      inflating: EURLEX57K/dataset/dev/31983D0530.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31983D0530.json  
      inflating: EURLEX57K/dataset/dev/32005R2090.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32005R2090.json  
      inflating: EURLEX57K/dataset/dev/31985D0069.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31985D0069.json  
      inflating: EURLEX57K/dataset/dev/32005D0416.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32005D0416.json  
      inflating: EURLEX57K/dataset/dev/32014D0225.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32014D0225.json  
      inflating: EURLEX57K/dataset/dev/31995R2014.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31995R2014.json  
      inflating: EURLEX57K/dataset/dev/31997D0102(01).json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31997D0102(01).json  
      inflating: EURLEX57K/dataset/dev/32006R1190.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32006R1190.json  
      inflating: EURLEX57K/dataset/dev/32001D0852.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32001D0852.json  
      inflating: EURLEX57K/dataset/dev/31986R3478.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31986R3478.json  
      inflating: EURLEX57K/dataset/dev/31996R1114.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31996R1114.json  
      inflating: EURLEX57K/dataset/dev/31993R0209.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31993R0209.json  
      inflating: EURLEX57K/dataset/dev/31993D0718.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31993D0718.json  
      inflating: EURLEX57K/dataset/dev/32002R0045.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32002R0045.json  
      inflating: EURLEX57K/dataset/dev/32006D0290.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32006D0290.json  
      inflating: EURLEX57K/dataset/dev/32007R0119.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32007R0119.json  
      inflating: EURLEX57K/dataset/dev/32002R0415.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32002R0415.json  
      inflating: EURLEX57K/dataset/dev/32007D0058.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32007D0058.json  
      inflating: EURLEX57K/dataset/dev/32011D0400.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32011D0400.json  
      inflating: EURLEX57K/dataset/dev/32001R2492.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32001R2492.json  
      inflating: EURLEX57K/dataset/dev/31991D0090.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31991D0090.json  
      inflating: EURLEX57K/dataset/dev/32003D0323.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32003D0323.json  
      inflating: EURLEX57K/dataset/dev/31996R1952.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31996R1952.json  
      inflating: EURLEX57K/dataset/dev/32004D0148.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32004D0148.json  
      inflating: EURLEX57K/dataset/dev/31989R2851.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31989R2851.json  
      inflating: EURLEX57K/dataset/dev/32010R0336.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32010R0336.json  
      inflating: EURLEX57K/dataset/dev/31990R2774.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31990R2774.json  
      inflating: EURLEX57K/dataset/dev/31998R2331.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31998R2331.json  
      inflating: EURLEX57K/dataset/dev/31981D0938.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31981D0938.json  
      inflating: EURLEX57K/dataset/dev/31991R3207.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31991R3207.json  
      inflating: EURLEX57K/dataset/dev/31993R3130.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31993R3130.json  
      inflating: EURLEX57K/dataset/dev/31990R3372.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31990R3372.json  
      inflating: EURLEX57K/dataset/dev/31986R1347.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31986R1347.json  
      inflating: EURLEX57K/dataset/dev/32001R1553.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32001R1553.json  
      inflating: EURLEX57K/dataset/dev/32011D0621(01).json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32011D0621(01).json  
      inflating: EURLEX57K/dataset/dev/32003R1664.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32003R1664.json  
      inflating: EURLEX57K/dataset/dev/32010D0429(01).json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32010D0429(01).json  
      inflating: EURLEX57K/dataset/dev/31989R0728.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31989R0728.json  
      inflating: EURLEX57K/dataset/dev/32005R1297.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32005R1297.json  
      inflating: EURLEX57K/dataset/dev/32010R0988.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32010R0988.json  
      inflating: EURLEX57K/dataset/dev/32003R0425.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32003R0425.json  
      inflating: EURLEX57K/dataset/dev/32005R0486.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32005R0486.json  
      inflating: EURLEX57K/dataset/dev/31975R2113.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31975R2113.json  
      inflating: EURLEX57K/dataset/dev/32000R0020.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32000R0020.json  
      inflating: EURLEX57K/dataset/dev/31991R0329.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31991R0329.json  
      inflating: EURLEX57K/dataset/dev/32010R0434.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32010R0434.json  
      inflating: EURLEX57K/dataset/dev/31996R2485.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31996R2485.json  
      inflating: EURLEX57K/dataset/dev/32006R1387.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32006R1387.json  
      inflating: EURLEX57K/dataset/dev/32013R1270.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32013R1270.json  
      inflating: EURLEX57K/dataset/dev/31990R2975.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31990R2975.json  
      inflating: EURLEX57K/dataset/dev/32004R0072.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32004R0072.json  
      inflating: EURLEX57K/dataset/dev/31992D0544.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31992D0544.json  
      inflating: EURLEX57K/dataset/dev/31978R3077.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31978R3077.json  
      inflating: EURLEX57K/dataset/dev/32006R1807.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32006R1807.json  
      inflating: EURLEX57K/dataset/dev/32014D0462.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32014D0462.json  
      inflating: EURLEX57K/dataset/dev/31997D0048.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31997D0048.json  
      inflating: EURLEX57K/dataset/dev/32001R2543.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32001R2543.json  
      inflating: EURLEX57K/dataset/dev/31989D0055.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31989D0055.json  
      inflating: EURLEX57K/dataset/dev/32002R1290.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32002R1290.json  
      inflating: EURLEX57K/dataset/dev/31977R1664.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31977R1664.json  
      inflating: EURLEX57K/dataset/dev/31985L0328.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31985L0328.json  
      inflating: EURLEX57K/dataset/dev/31993R1526.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31993R1526.json  
      inflating: EURLEX57K/dataset/dev/31995R1585.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31995R1585.json  
      inflating: EURLEX57K/dataset/dev/32005R1151.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32005R1151.json  
      inflating: EURLEX57K/dataset/dev/31990L0660.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31990L0660.json  
      inflating: EURLEX57K/dataset/dev/32006R1041.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32006R1041.json  
      inflating: EURLEX57K/dataset/dev/32002R2003.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32002R2003.json  
      inflating: EURLEX57K/dataset/dev/32001R0591.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32001R0591.json  
      inflating: EURLEX57K/dataset/dev/31991D0454.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31991D0454.json  
      inflating: EURLEX57K/dataset/dev/31993R1930.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31993R1930.json  
      inflating: EURLEX57K/dataset/dev/31979R1543.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31979R1543.json  
      inflating: EURLEX57K/dataset/dev/31990R2164.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31990R2164.json  
      inflating: EURLEX57K/dataset/dev/31999R1595.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31999R1595.json  
      inflating: EURLEX57K/dataset/dev/31996D0111.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31996D0111.json  
      inflating: EURLEX57K/dataset/dev/32000D0136.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32000D0136.json  
      inflating: EURLEX57K/dataset/dev/31975D0038.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31975D0038.json  
      inflating: EURLEX57K/dataset/dev/31993R2561.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31993R2561.json  
      inflating: EURLEX57K/dataset/dev/32003R1633.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32003R1633.json  
      inflating: EURLEX57K/dataset/dev/32003R0921.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32003R0921.json  
      inflating: EURLEX57K/dataset/dev/31987R1588.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31987R1588.json  
      inflating: EURLEX57K/dataset/dev/31994D1067.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31994D1067.json  
      inflating: EURLEX57K/dataset/dev/31994R1126.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31994R1126.json  
      inflating: EURLEX57K/dataset/dev/31997R1036.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31997R1036.json  
      inflating: EURLEX57K/dataset/dev/31979R2380.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31979R2380.json  
      inflating: EURLEX57K/dataset/dev/32000R1373.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32000R1373.json  
      inflating: EURLEX57K/dataset/dev/31985D0100.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31985D0100.json  
      inflating: EURLEX57K/dataset/dev/32002D0714.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32002D0714.json  
      inflating: EURLEX57K/dataset/dev/31984R0289.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31984R0289.json  
      inflating: EURLEX57K/dataset/dev/32000D0423.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32000D0423.json  
      inflating: EURLEX57K/dataset/dev/31999R1080.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31999R1080.json  
      inflating: EURLEX57K/dataset/dev/32005D0085.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32005D0085.json  
      inflating: EURLEX57K/dataset/dev/31986R2029.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31986R2029.json  
      inflating: EURLEX57K/dataset/dev/32001R0600.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32001R0600.json  
      inflating: EURLEX57K/dataset/dev/31994D0699.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31994D0699.json  
      inflating: EURLEX57K/dataset/dev/31992R1869.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31992R1869.json  
      inflating: EURLEX57K/dataset/dev/32011R0351.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32011R0351.json  
      inflating: EURLEX57K/dataset/dev/31999D0103.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31999D0103.json  
      inflating: EURLEX57K/dataset/dev/32004L0032.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32004L0032.json  
      inflating: EURLEX57K/dataset/dev/31984R0909.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31984R0909.json  
      inflating: EURLEX57K/dataset/dev/32007R0973.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32007R0973.json  
      inflating: EURLEX57K/dataset/dev/31994R2566.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31994R2566.json  
      inflating: EURLEX57K/dataset/dev/31990D0274.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31990D0274.json  
      inflating: EURLEX57K/dataset/dev/32014D0435.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32014D0435.json  
      inflating: EURLEX57K/dataset/dev/31996R0396.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31996R0396.json  
      inflating: EURLEX57K/dataset/dev/32005D0206.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32005D0206.json  
      inflating: EURLEX57K/dataset/dev/32007D0131.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32007D0131.json  
      inflating: EURLEX57K/dataset/dev/31993D0671.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31993D0671.json  
      inflating: EURLEX57K/dataset/dev/32005R0347.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32005R0347.json  
      inflating: EURLEX57K/dataset/dev/31983D0320.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31983D0320.json  
      inflating: EURLEX57K/dataset/dev/31992R0517.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31992R0517.json  
      inflating: EURLEX57K/dataset/dev/32004R0530.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32004R0530.json  
      inflating: EURLEX57K/dataset/dev/31972R2456.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31972R2456.json  
      inflating: EURLEX57K/dataset/dev/31985R2210.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31985R2210.json  
      inflating: EURLEX57K/dataset/dev/31991R2085.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31991R2085.json  
      inflating: EURLEX57K/dataset/dev/31990D0331.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31990D0331.json  
      inflating: EURLEX57K/dataset/dev/32005R1940.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32005R1940.json  
      inflating: EURLEX57K/dataset/dev/32004R0160.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32004R0160.json  
      inflating: EURLEX57K/dataset/dev/32007R1374.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32007R1374.json  
      inflating: EURLEX57K/dataset/dev/32009R0183.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32009R0183.json  
      inflating: EURLEX57K/dataset/dev/31997D0859.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31997D0859.json  
      inflating: EURLEX57K/dataset/dev/31969R2517.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31969R2517.json  
      inflating: EURLEX57K/dataset/dev/32001R1142.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32001R1142.json  
      inflating: EURLEX57K/dataset/dev/32009R0746.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32009R0746.json  
      inflating: EURLEX57K/dataset/dev/32006R1779.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32006R1779.json  
      inflating: EURLEX57K/dataset/dev/31996R1657.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31996R1657.json  
      inflating: EURLEX57K/dataset/dev/31989R0769.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31989R0769.json  
      inflating: EURLEX57K/dataset/dev/32013D0424.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32013D0424.json  
      inflating: EURLEX57K/dataset/dev/32010R0160.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32010R0160.json  
      inflating: EURLEX57K/dataset/dev/32006D0479.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32006D0479.json  
      inflating: EURLEX57K/dataset/dev/32003D0525.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32003D0525.json  
      inflating: EURLEX57K/dataset/dev/32003R0034.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32003R0034.json  
      inflating: EURLEX57K/dataset/dev/31994R1999.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31994R1999.json  
      inflating: EURLEX57K/dataset/dev/32009R1012.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32009R1012.json  
      inflating: EURLEX57K/dataset/dev/32001R1904.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32001R1904.json  
      inflating: EURLEX57K/dataset/dev/31999R1096.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31999R1096.json  
      inflating: EURLEX57K/dataset/dev/32005D0093.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32005D0093.json  
      inflating: EURLEX57K/dataset/dev/32005R0582.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32005R0582.json  
      inflating: EURLEX57K/dataset/dev/31995D0447.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31995D0447.json  
      inflating: EURLEX57K/dataset/dev/32006R2010.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32006R2010.json  
      inflating: EURLEX57K/dataset/dev/32000R1365.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32000R1365.json  
      inflating: EURLEX57K/dataset/dev/32008D0070.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32008D0070.json  
      inflating: EURLEX57K/dataset/dev/32005R1239.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32005R1239.json  
      inflating: EURLEX57K/dataset/dev/32009R0879.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32009R0879.json  
      inflating: EURLEX57K/dataset/dev/32004D0522.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32004D0522.json  
      inflating: EURLEX57K/dataset/dev/31977D0525.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31977D0525.json  
      inflating: EURLEX57K/dataset/dev/32005R1813.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32005R1813.json  
      inflating: EURLEX57K/dataset/dev/32001R2502.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32001R2502.json  
      inflating: EURLEX57K/dataset/dev/32014R0132.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32014R0132.json  
      inflating: EURLEX57K/dataset/dev/32014D0423.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32014D0423.json  
      inflating: EURLEX57K/dataset/dev/31988R3948.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31988R3948.json  
      inflating: EURLEX57K/dataset/dev/31993R1567.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31993R1567.json  
      inflating: EURLEX57K/dataset/dev/32002R0993.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32002R0993.json  
      inflating: EURLEX57K/dataset/dev/31998D0277.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31998D0277.json  
      inflating: EURLEX57K/dataset/dev/32004R1788.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32004R1788.json  
      inflating: EURLEX57K/dataset/dev/32008R0648.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32008R0648.json  
      inflating: EURLEX57K/dataset/dev/32004R1767.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32004R1767.json  
      inflating: EURLEX57K/dataset/dev/32005R1055.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32005R1055.json  
      inflating: EURLEX57K/dataset/dev/31994R3224.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31994R3224.json  
      inflating: EURLEX57K/dataset/dev/31998R1198.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31998R1198.json  
      inflating: EURLEX57K/dataset/dev/32010D0758.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32010D0758.json  
      inflating: EURLEX57K/dataset/dev/32007R0089.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32007R0089.json  
      inflating: EURLEX57K/dataset/dev/32007D0598.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32007D0598.json  
      inflating: EURLEX57K/dataset/dev/32011R0605.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32011R0605.json  
      inflating: EURLEX57K/dataset/dev/32003R0063.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32003R0063.json  
      inflating: EURLEX57K/dataset/dev/32008R0920.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32008R0920.json  
      inflating: EURLEX57K/dataset/dev/32001D0645.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32001D0645.json  
      inflating: EURLEX57K/dataset/dev/32001R2369.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32001R2369.json  
      inflating: EURLEX57K/dataset/dev/32011R0255.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32011R0255.json  
      inflating: EURLEX57K/dataset/dev/32006D0584.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32006D0584.json  
      inflating: EURLEX57K/dataset/dev/31995D0105.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31995D0105.json  
      inflating: EURLEX57K/dataset/dev/32012D0711.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32012D0711.json  
      inflating: EURLEX57K/dataset/dev/32010R0137.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32010R0137.json  
      inflating: EURLEX57K/dataset/dev/32006L0082.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32006L0082.json  
      inflating: EURLEX57K/dataset/dev/32012R1041.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32012R1041.json  
      inflating: EURLEX57K/dataset/dev/32003R1388.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32003R1388.json  
      inflating: EURLEX57K/dataset/dev/31989R2353.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31989R2353.json  
      inflating: EURLEX57K/dataset/dev/32000L0021.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32000L0021.json  
      inflating: EURLEX57K/dataset/dev/31987R1463.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31987R1463.json  
      inflating: EURLEX57K/dataset/dev/31998R2560.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31998R2560.json  
      inflating: EURLEX57K/dataset/dev/32009R0711.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32009R0711.json  
      inflating: EURLEX57K/dataset/dev/31985R2894.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31985R2894.json  
      inflating: EURLEX57K/dataset/dev/32007R1159.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32007R1159.json  
      inflating: EURLEX57K/dataset/dev/31985R0450.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31985R0450.json  
      inflating: EURLEX57K/dataset/dev/32002R2279.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32002R2279.json  
      inflating: EURLEX57K/dataset/dev/32007R0748.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32007R0748.json  
      inflating: EURLEX57K/dataset/dev/31984R3667.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31984R3667.json  
      inflating: EURLEX57K/dataset/dev/32007R1270.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32007R1270.json  
      inflating: EURLEX57K/dataset/dev/31998R0731.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31998R0731.json  
      inflating: EURLEX57K/dataset/dev/31988R1922.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31988R1922.json  
      inflating: EURLEX57K/dataset/dev/31988R1471.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31988R1471.json  
      inflating: EURLEX57K/dataset/dev/31969R1395.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31969R1395.json  
      inflating: EURLEX57K/dataset/dev/31998D1022(01).json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31998D1022(01).json  
      inflating: EURLEX57K/dataset/dev/31995D0369.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31995D0369.json  
      inflating: EURLEX57K/dataset/dev/31983R0220.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31983R0220.json  
      inflating: EURLEX57K/dataset/dev/31993D0630.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31993D0630.json  
      inflating: EURLEX57K/dataset/dev/31985R0283.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31985R0283.json  
      inflating: EURLEX57K/dataset/dev/31980D0334.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31980D0334.json  
      inflating: EURLEX57K/dataset/dev/32002R0182.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32002R0182.json  
      inflating: EURLEX57K/dataset/dev/32007D0035.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32007D0035.json  
      inflating: EURLEX57K/dataset/dev/32001R2410.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32001R2410.json  
      inflating: EURLEX57K/dataset/dev/32014D0531.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32014D0531.json  
      inflating: EURLEX57K/dataset/dev/32012D0068.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32012D0068.json  
      inflating: EURLEX57K/dataset/dev/31999D0007.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31999D0007.json  
      inflating: EURLEX57K/dataset/dev/32006R1057.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32006R1057.json  
      inflating: EURLEX57K/dataset/dev/31991R1742.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31991R1742.json  
      inflating: EURLEX57K/dataset/dev/31988R0775.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31988R0775.json  
      inflating: EURLEX57K/dataset/dev/31999R0131.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31999R0131.json  
      inflating: EURLEX57K/dataset/dev/31988D0213.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31988D0213.json  
      inflating: EURLEX57K/dataset/dev/31994R2946.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31994R2946.json  
      inflating: EURLEX57K/dataset/dev/31976R1422.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31976R1422.json  
      inflating: EURLEX57K/dataset/dev/31999R0561.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31999R0561.json  
      inflating: EURLEX57K/dataset/dev/31975L0271.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31975L0271.json  
      inflating: EURLEX57K/dataset/dev/31984D0038.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31984D0038.json  
      inflating: EURLEX57K/dataset/dev/32013R0396.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32013R0396.json  
      inflating: EURLEX57K/dataset/dev/31994D0093.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31994D0093.json  
      inflating: EURLEX57K/dataset/dev/32013D0687.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32013D0687.json  
      inflating: EURLEX57K/dataset/dev/32001R2467.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32001R2467.json  
      inflating: EURLEX57K/dataset/dev/31988R3084.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31988R3084.json  
      inflating: EURLEX57K/dataset/dev/31998R1012.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31998R1012.json  
      inflating: EURLEX57K/dataset/dev/32004R0013.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32004R0013.json  
      inflating: EURLEX57K/dataset/dev/31992D0525.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31992D0525.json  
      inflating: EURLEX57K/dataset/dev/32013D0392.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32013D0392.json  
      inflating: EURLEX57K/dataset/dev/32009D0918.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32009D0918.json  
      inflating: EURLEX57K/dataset/dev/32005R0721.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32005R0721.json  
      inflating: EURLEX57K/dataset/dev/32013R0379.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32013R0379.json  
      inflating: EURLEX57K/dataset/dev/31990R0753.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31990R0753.json  
      inflating: EURLEX57K/dataset/dev/32011L0059.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32011L0059.json  
      inflating: EURLEX57K/dataset/dev/31995R3023.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31995R3023.json  
      inflating: EURLEX57K/dataset/dev/31989D0464.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31989D0464.json  
      inflating: EURLEX57K/dataset/dev/31999D0135.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31999D0135.json  
      inflating: EURLEX57K/dataset/dev/32006R1165.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32006R1165.json  
      inflating: EURLEX57K/dataset/dev/32012R0332.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32012R0332.json  
      inflating: EURLEX57K/dataset/dev/31994R2383.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31994R2383.json  
      inflating: EURLEX57K/dataset/dev/32005R0408.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32005R0408.json  
      inflating: EURLEX57K/dataset/dev/31988R0997.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31988R0997.json  
      inflating: EURLEX57K/dataset/dev/32012R0762.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32012R0762.json  
      inflating: EURLEX57K/dataset/dev/32005R2065.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32005R2065.json  
      inflating: EURLEX57K/dataset/dev/32013R0953.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32013R0953.json  
      inflating: EURLEX57K/dataset/dev/31974R1579.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31974R1579.json  
      inflating: EURLEX57K/dataset/dev/32010R0843.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32010R0843.json  
      inflating: EURLEX57K/dataset/dev/32003R1605.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32003R1605.json  
      inflating: EURLEX57K/dataset/dev/32006D0009.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32006D0009.json  
      inflating: EURLEX57K/dataset/dev/31994D0640.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31994D0640.json  
      inflating: EURLEX57K/dataset/dev/32004R1897.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32004R1897.json  
      inflating: EURLEX57K/dataset/dev/32011D0699.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32011D0699.json  
      inflating: EURLEX57K/dataset/dev/31992R0258.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31992R0258.json  
      inflating: EURLEX57K/dataset/dev/32003D0505.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32003D0505.json  
      inflating: EURLEX57K/dataset/dev/32009R1209.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32009R1209.json  
      inflating: EURLEX57K/dataset/dev/32012D0048.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32012D0048.json  
      inflating: EURLEX57K/dataset/dev/32007D0445.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32007D0445.json  
      inflating: EURLEX57K/dataset/dev/32001R2060.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32001R2060.json  
      inflating: EURLEX57K/dataset/dev/32009R0418.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32009R0418.json  
      inflating: EURLEX57K/dataset/dev/32013R1180.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32013R1180.json  
      inflating: EURLEX57K/dataset/dev/31971L0018.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31971L0018.json  
      inflating: EURLEX57K/dataset/dev/32009D0559.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32009D0559.json  
      inflating: EURLEX57K/dataset/dev/32009R0048.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32009R0048.json  
      inflating: EURLEX57K/dataset/dev/32014R1241.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32014R1241.json  
      inflating: EURLEX57K/dataset/dev/31988D0244.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31988D0244.json  
      inflating: EURLEX57K/dataset/dev/31999D0162.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31999D0162.json  
      inflating: EURLEX57K/dataset/dev/32008R0795.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32008R0795.json  
      inflating: EURLEX57K/dataset/dev/31998D0650.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31998D0650.json  
      inflating: EURLEX57K/dataset/dev/31994R2507.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31994R2507.json  
      inflating: EURLEX57K/dataset/dev/31987D0485.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31987D0485.json  
      inflating: EURLEX57K/dataset/dev/32006D0662.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32006D0662.json  
      inflating: EURLEX57K/dataset/dev/32008R1184.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32008R1184.json  
      inflating: EURLEX57K/dataset/dev/32011D0158.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32011D0158.json  
      inflating: EURLEX57K/dataset/dev/31992R0063.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31992R0063.json  
      inflating: EURLEX57K/dataset/dev/32004R2079.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32004R2079.json  
      inflating: EURLEX57K/dataset/dev/32011R0449.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32011R0449.json  
      inflating: EURLEX57K/dataset/dev/32013L0010.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32013L0010.json  
      inflating: EURLEX57K/dataset/dev/32008R0516.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32008R0516.json  
      inflating: EURLEX57K/dataset/dev/32002R2259.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32002R2259.json  
      inflating: EURLEX57K/dataset/dev/32013R0457.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32013R0457.json  
      inflating: EURLEX57K/dataset/dev/32002D0775.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32002D0775.json  
      inflating: EURLEX57K/dataset/dev/32012R0735.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32012R0735.json  
      inflating: EURLEX57K/dataset/dev/31990D0096.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31990D0096.json  
      inflating: EURLEX57K/dataset/dev/32013R0007.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32013R0007.json  
      inflating: EURLEX57K/dataset/dev/32003R2091.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32003R2091.json  
      inflating: EURLEX57K/dataset/dev/31994D0302.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31994D0302.json  
      inflating: EURLEX57K/dataset/dev/31996R0524.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31996R0524.json  
      inflating: EURLEX57K/dataset/dev/32014D0268.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32014D0268.json  
      inflating: EURLEX57K/dataset/dev/32014R0779.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32014R0779.json  
      inflating: EURLEX57K/dataset/dev/32001R2349.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32001R2349.json  
      inflating: EURLEX57K/dataset/dev/32003R0413.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32003R0413.json  
      inflating: EURLEX57K/dataset/dev/31991R3925.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31991R3925.json  
      inflating: EURLEX57K/dataset/dev/32002D0260.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32002D0260.json  
      inflating: EURLEX57K/dataset/dev/32002R0771.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32002R0771.json  
      inflating: EURLEX57K/dataset/dev/31993R2500.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31993R2500.json  
      inflating: EURLEX57K/dataset/dev/32004R0382.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32004R0382.json  
      inflating: EURLEX57K/dataset/dev/31984R3581.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31984R3581.json  
      inflating: EURLEX57K/dataset/dev/31994D1006.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31994D1006.json  
      inflating: EURLEX57K/dataset/dev/31991R3026.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31991R3026.json  
      inflating: EURLEX57K/dataset/dev/32009R0731.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32009R0731.json  
      inflating: EURLEX57K/dataset/dev/31995R1225.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31995R1225.json  
      inflating: EURLEX57K/dataset/dev/32006L0108.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32006L0108.json  
      inflating: EURLEX57K/dataset/dev/32002R1160.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32002R1160.json  
      inflating: EURLEX57K/dataset/dev/32001R1135.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32001R1135.json  
      inflating: EURLEX57K/dataset/dev/31998D0596.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31998D0596.json  
      inflating: EURLEX57K/dataset/dev/31988R0743.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31988R0743.json  
      inflating: EURLEX57K/dataset/dev/32000R1492.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32000R1492.json  
      inflating: EURLEX57K/dataset/dev/31981D0125.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31981D0125.json  
      inflating: EURLEX57K/dataset/dev/32002R2023.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32002R2023.json  
      inflating: EURLEX57K/dataset/dev/32014D0157.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32014D0157.json  
      inflating: EURLEX57K/dataset/dev/31993D0743.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31993D0743.json  
      inflating: EURLEX57K/dataset/dev/32000R2314.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32000R2314.json  
      inflating: EURLEX57K/dataset/dev/32007D0453.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32007D0453.json  
      inflating: EURLEX57K/dataset/dev/32007R0142.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32007R0142.json  
      inflating: EURLEX57K/dataset/dev/32006R0670.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32006R0670.json  
      inflating: EURLEX57K/dataset/dev/32013R0787.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32013R0787.json  
      inflating: EURLEX57K/dataset/dev/31989R1375.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31989R1375.json  
      inflating: EURLEX57K/dataset/dev/32004R0052.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32004R0052.json  
      inflating: EURLEX57K/dataset/dev/31987D0169.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31987D0169.json  
      inflating: EURLEX57K/dataset/dev/32004D0543.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32004D0543.json  
      inflating: EURLEX57K/dataset/dev/32001R2133.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32001R2133.json  
      inflating: EURLEX57K/dataset/dev/32003D0378.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32003D0378.json  
      inflating: EURLEX57K/dataset/dev/32014D0442.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32014D0442.json  
      inflating: EURLEX57K/dataset/dev/32005R0760.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32005R0760.json  
      inflating: EURLEX57K/dataset/dev/31983L0201.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31983L0201.json  
      inflating: EURLEX57K/dataset/dev/32012D0818.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32012D0818.json  
      inflating: EURLEX57K/dataset/dev/32008R0279.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32008R0279.json  
      inflating: EURLEX57K/dataset/dev/31988R1914.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31988R1914.json  
      inflating: EURLEX57K/dataset/dev/32004R1213.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32004R1213.json  
      inflating: EURLEX57K/dataset/dev/31991R3876.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31991R3876.json  
      inflating: EURLEX57K/dataset/dev/32012D0662.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32012D0662.json  
      inflating: EURLEX57K/dataset/dev/32000D0004.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32000D0004.json  
      inflating: EURLEX57K/dataset/dev/31988R1294.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31988R1294.json  
      inflating: EURLEX57K/dataset/dev/31993D0485.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31993D0485.json  
      inflating: EURLEX57K/dataset/dev/32010R0044.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32010R0044.json  
      inflating: EURLEX57K/dataset/dev/32001D0366.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32001D0366.json  
      inflating: EURLEX57K/dataset/dev/32008R1311.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32008R1311.json  
      inflating: EURLEX57K/dataset/dev/32002R0788.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32002R0788.json  
      inflating: EURLEX57K/dataset/dev/32011D0267.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32011D0267.json  
      inflating: EURLEX57K/dataset/dev/31996D0473.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31996D0473.json  
      inflating: EURLEX57K/dataset/dev/31997D0711.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31997D0711.json  
      inflating: EURLEX57K/dataset/dev/32010R0414.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32010R0414.json  
      inflating: EURLEX57K/dataset/dev/31986R1388.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31986R1388.json  
      inflating: EURLEX57K/dataset/dev/32010R0947.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32010R0947.json  
      inflating: EURLEX57K/dataset/dev/32009R0398.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32009R0398.json  
      inflating: EURLEX57K/dataset/dev/32012R1098.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32012R1098.json  
      inflating: EURLEX57K/dataset/dev/32009R0232.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32009R0232.json  
      inflating: EURLEX57K/dataset/dev/31996R1289.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31996R1289.json  
      inflating: EURLEX57K/dataset/dev/32001R1123.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32001R1123.json  
      inflating: EURLEX57K/dataset/dev/31998R2556.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31998R2556.json  
      inflating: EURLEX57K/dataset/dev/31982R3164.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31982R3164.json  
      inflating: EURLEX57K/dataset/dev/32008R0015.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32008R0015.json  
      inflating: EURLEX57K/dataset/dev/31994D1010.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31994D1010.json  
      inflating: EURLEX57K/dataset/dev/32009R0377.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32009R0377.json  
      inflating: EURLEX57K/dataset/dev/32010R0101.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32010R0101.json  
      inflating: EURLEX57K/dataset/dev/32008L0002.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32008L0002.json  
      inflating: EURLEX57K/dataset/dev/32008R1254.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32008R1254.json  
      inflating: EURLEX57K/dataset/dev/31981R2370.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31981R2370.json  
      inflating: EURLEX57K/dataset/dev/32013R0504.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32013R0504.json  
      inflating: EURLEX57K/dataset/dev/32006R0109.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32006R0109.json  
      inflating: EURLEX57K/dataset/dev/31989R1119.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31989R1119.json  
      inflating: EURLEX57K/dataset/dev/32011D0322.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32011D0322.json  
      inflating: EURLEX57K/dataset/dev/32003R1869.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32003R1869.json  
      inflating: EURLEX57K/dataset/dev/32012R0148.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32012R0148.json  
      inflating: EURLEX57K/dataset/dev/32002R0419.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32002R0419.json  
      inflating: EURLEX57K/dataset/dev/31983D0245.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31983D0245.json  
      inflating: EURLEX57K/dataset/dev/32013R0380.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32013R0380.json  
      inflating: EURLEX57K/dataset/dev/32002D0558.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32002D0558.json  
      inflating: EURLEX57K/dataset/dev/31987D0581.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31987D0581.json  
      inflating: EURLEX57K/dataset/dev/32004R1301.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32004R1301.json  
      inflating: EURLEX57K/dataset/dev/32005R1599.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32005R1599.json  
      inflating: EURLEX57K/dataset/dev/32007R1354.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32007R1354.json  
      inflating: EURLEX57K/dataset/dev/31991R1723.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31991R1723.json  
      inflating: EURLEX57K/dataset/dev/32006R1466.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32006R1466.json  
      inflating: EURLEX57K/dataset/dev/31998R2782.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31998R2782.json  
      inflating: EURLEX57K/dataset/dev/32001R1318.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32001R1318.json  
      inflating: EURLEX57K/dataset/dev/32003R1185.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32003R1185.json  
      inflating: EURLEX57K/dataset/dev/32001D0018.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32001D0018.json  
      inflating: EURLEX57K/dataset/dev/32000D0680.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32000D0680.json  
      inflating: EURLEX57K/dataset/dev/32001R0509.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32001R0509.json  
      inflating: EURLEX57K/dataset/dev/31998R1141.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31998R1141.json  
      inflating: EURLEX57K/dataset/dev/32011R0058.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32011R0058.json  
      inflating: EURLEX57K/dataset/dev/32011R0408.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32011R0408.json  
      inflating: EURLEX57K/dataset/dev/32010R0290.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32010R0290.json  
      inflating: EURLEX57K/dataset/dev/31991D0566.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31991D0566.json  
      inflating: EURLEX57K/dataset/dev/32010L0004.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32010L0004.json  
      inflating: EURLEX57K/dataset/dev/32009R0265.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32009R0265.json  
      inflating: EURLEX57K/dataset/dev/31992R3527.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31992R3527.json  
      inflating: EURLEX57K/dataset/dev/32005L0059.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32005L0059.json  
      inflating: EURLEX57K/dataset/dev/32008D0046.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32008D0046.json  
      inflating: EURLEX57K/dataset/dev/32009R0635.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32009R0635.json  
      inflating: EURLEX57K/dataset/dev/31985D0120.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31985D0120.json  
      inflating: EURLEX57K/dataset/dev/32007D0392.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32007D0392.json  
      inflating: EURLEX57K/dataset/dev/31995D0471.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31995D0471.json  
      inflating: EURLEX57K/dataset/dev/31988R1693.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31988R1693.json  
      inflating: EURLEX57K/dataset/dev/32010R0013.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32010R0013.json  
      inflating: EURLEX57K/dataset/dev/31997D0316.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31997D0316.json  
      inflating: EURLEX57K/dataset/dev/31985R0061.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31985R0061.json  
      inflating: EURLEX57K/dataset/dev/32002R1967.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32002R1967.json  
      inflating: EURLEX57K/dataset/dev/32007R0729.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32007R0729.json  
      inflating: EURLEX57K/dataset/dev/31988L0095.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31988L0095.json  
      inflating: EURLEX57K/dataset/dev/32003D0513.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32003D0513.json  
      inflating: EURLEX57K/dataset/dev/32014R0738.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32014R0738.json  
      inflating: EURLEX57K/dataset/dev/31985D0065.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31985D0065.json  
      inflating: EURLEX57K/dataset/dev/31990R2144.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31990R2144.json  
      inflating: EURLEX57K/dataset/dev/31976R2561.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31976R2561.json  
      inflating: EURLEX57K/dataset/dev/31993R0086.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31993R0086.json  
      inflating: EURLEX57K/dataset/dev/31997R1850.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31997R1850.json  
      inflating: EURLEX57K/dataset/dev/31985R2119.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31985R2119.json  
      inflating: EURLEX57K/dataset/dev/32003D0840.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32003D0840.json  
      inflating: EURLEX57K/dataset/dev/32001R1174.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32001R1174.json  
      inflating: EURLEX57K/dataset/dev/32002R1121.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32002R1121.json  
      inflating: EURLEX57K/dataset/dev/32006R1108.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32006R1108.json  
      inflating: EURLEX57K/dataset/dev/32007D0869.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32007D0869.json  
      inflating: EURLEX57K/dataset/dev/32011R0920.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32011R0920.json  
      inflating: EURLEX57K/dataset/dev/32004L0069.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32004L0069.json  
      inflating: EURLEX57K/dataset/dev/31998D0390.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31998D0390.json  
      inflating: EURLEX57K/dataset/dev/32013R1155.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32013R1155.json  
      inflating: EURLEX57K/dataset/dev/31985R1172.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31985R1172.json  
      inflating: EURLEX57K/dataset/dev/32009R0137.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32009R0137.json  
      inflating: EURLEX57K/dataset/dev/31993R1879.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31993R1879.json  
      inflating: EURLEX57K/dataset/dev/31986R0366.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31986R0366.json  
      inflating: EURLEX57K/dataset/dev/32006D0208.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32006D0208.json  
      inflating: EURLEX57K/dataset/dev/32009D0975.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32009D0975.json  
      inflating: EURLEX57K/dataset/dev/31987R2039.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31987R2039.json  
      inflating: EURLEX57K/dataset/dev/32001R0088.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32001R0088.json  
      inflating: EURLEX57K/dataset/dev/31986D0227.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31986D0227.json  
      inflating: EURLEX57K/dataset/dev/31992R0059.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31992R0059.json  
      inflating: EURLEX57K/dataset/dev/32003R1812.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32003R1812.json  
      inflating: EURLEX57K/dataset/dev/32003R0700.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32003R0700.json  
      inflating: EURLEX57K/dataset/dev/31994D0154.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31994D0154.json  
      inflating: EURLEX57K/dataset/dev/32005R0259.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32005R0259.json  
      inflating: EURLEX57K/dataset/dev/31994R2028.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31994R2028.json  
      inflating: EURLEX57K/dataset/dev/32003R1111.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32003R1111.json  
      inflating: EURLEX57K/dataset/dev/32008R0740.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32008R0740.json  
      inflating: EURLEX57K/dataset/dev/32011R1327.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32011R1327.json  
      inflating: EURLEX57K/dataset/dev/32004R0838.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32004R0838.json  
      inflating: EURLEX57K/dataset/dev/32004L0086.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32004L0086.json  
      inflating: EURLEX57K/dataset/dev/32007R1285.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32007R1285.json  
      inflating: EURLEX57K/dataset/dev/32004R0212.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32004R0212.json  
      inflating: EURLEX57K/dataset/dev/31997R0693.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31997R0693.json  
      inflating: EURLEX57K/dataset/dev/31997D0678.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31997D0678.json  
      inflating: EURLEX57K/dataset/dev/32005R0170.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32005R0170.json  
      inflating: EURLEX57K/dataset/dev/32004R1950.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32004R1950.json  
      inflating: EURLEX57K/dataset/dev/32000D0497.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32000D0497.json  
      inflating: EURLEX57K/dataset/dev/32004R1403.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32004R1403.json  
      inflating: EURLEX57K/dataset/dev/31979R0309.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31979R0309.json  
      inflating: EURLEX57K/dataset/dev/31976L0160.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31976L0160.json  
      inflating: EURLEX57K/dataset/dev/31994R3140.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31994R3140.json  
      inflating: EURLEX57K/dataset/dev/31999R0330.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31999R0330.json  
      inflating: EURLEX57K/dataset/dev/31998D0113.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31998D0113.json  
      inflating: EURLEX57K/dataset/dev/31997R1597.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31997R1597.json  
      inflating: EURLEX57K/dataset/dev/32006D0571.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32006D0571.json  
      inflating: EURLEX57K/dataset/dev/31991R0325.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31991R0325.json  
      inflating: EURLEX57K/dataset/dev/32005R0465.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32005R0465.json  
      inflating: EURLEX57K/dataset/dev/31970R2556.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31970R2556.json  
      inflating: EURLEX57K/dataset/dev/31997R0152.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31997R0152.json  
      inflating: EURLEX57K/dataset/dev/31984R2191.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31984R2191.json  
      inflating: EURLEX57K/dataset/dev/32012D0530.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32012D0530.json  
      inflating: EURLEX57K/dataset/dev/32001D0034.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32001D0034.json  
      inflating: EURLEX57K/dataset/dev/31970D0304.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31970D0304.json  
      inflating: EURLEX57K/dataset/dev/31994D0046.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31994D0046.json  
      inflating: EURLEX57K/dataset/dev/32009D0471.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32009D0471.json  
      inflating: EURLEX57K/dataset/dev/32014R0981.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32014R0981.json  
      inflating: EURLEX57K/dataset/dev/31979D0423.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31979D0423.json  
      inflating: EURLEX57K/dataset/dev/31995R3019.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31995R3019.json  
      inflating: EURLEX57K/dataset/dev/32011R0977.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32011R0977.json  
      inflating: EURLEX57K/dataset/dev/32004R1268.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32004R1268.json  
      inflating: EURLEX57K/dataset/dev/31987R3385.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31987R3385.json  
      inflating: EURLEX57K/dataset/dev/32013R1047.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32013R1047.json  
      inflating: EURLEX57K/dataset/dev/31990R1197.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31990R1197.json  
      inflating: EURLEX57K/dataset/dev/32003R1516.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32003R1516.json  
      inflating: EURLEX57K/dataset/dev/31992R2875.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31992R2875.json  
      inflating: EURLEX57K/dataset/dev/31992R0848.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31992R0848.json  
      inflating: EURLEX57K/dataset/dev/32002R1224.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32002R1224.json  
      inflating: EURLEX57K/dataset/dev/31984R1752.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31984R1752.json  
      inflating: EURLEX57K/dataset/dev/31978D0254.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31978D0254.json  
      inflating: EURLEX57K/dataset/dev/32004D0187.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32004D0187.json  
      inflating: EURLEX57K/dataset/dev/31981R2188.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31981R2188.json  
      inflating: EURLEX57K/dataset/dev/32002R0435.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32002R0435.json  
      inflating: EURLEX57K/dataset/dev/32007R0569.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32007R0569.json  
      inflating: EURLEX57K/dataset/dev/32003D0246.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32003D0246.json  
      inflating: EURLEX57K/dataset/dev/32013D0718(01).json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32013D0718(01).json  
      inflating: EURLEX57K/dataset/dev/31985R2836.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31985R2836.json  
      inflating: EURLEX57K/dataset/dev/31999D0363.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31999D0363.json  
      inflating: EURLEX57K/dataset/dev/32014D0008(01).json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32014D0008(01).json  
      inflating: EURLEX57K/dataset/dev/31978R2451.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31978R2451.json  
      inflating: EURLEX57K/dataset/dev/31998R2487.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31998R2487.json  
      inflating: EURLEX57K/dataset/dev/32005R1736.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32005R1736.json  
      inflating: EURLEX57K/dataset/dev/31993R0803.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31993R0803.json  
      inflating: EURLEX57K/dataset/dev/32013R0986.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32013R0986.json  
      inflating: EURLEX57K/dataset/dev/31990D0014.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31990D0014.json  
      inflating: EURLEX57K/dataset/dev/31995R2464.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31995R2464.json  
      inflating: EURLEX57K/dataset/dev/31985D0049.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31985D0049.json  
      inflating: EURLEX57K/dataset/dev/32006R0488.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32006R0488.json  
      inflating: EURLEX57K/dataset/dev/32013R0085.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32013R0085.json  
      inflating: EURLEX57K/dataset/dev/31996D0058.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31996D0058.json  
      inflating: EURLEX57K/dataset/dev/32006R0467.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32006R0467.json  
      inflating: EURLEX57K/dataset/dev/32002D0348.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32002D0348.json  
      inflating: EURLEX57K/dataset/dev/32005R0432.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32005R0432.json  
      inflating: EURLEX57K/dataset/dev/31997R1939.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31997R1939.json  
      inflating: EURLEX57K/dataset/dev/31983R2529.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31983R2529.json  
      inflating: EURLEX57K/dataset/dev/32010D0484.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32010D0484.json  
      inflating: EURLEX57K/dataset/dev/32015R0533.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32015R0533.json  
      inflating: EURLEX57K/dataset/dev/31993R2582.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31993R2582.json  
      inflating: EURLEX57K/dataset/dev/32007R0355.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32007R0355.json  
      inflating: EURLEX57K/dataset/dev/31995R2171.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31995R2171.json  
      inflating: EURLEX57K/dataset/dev/32008R0828.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32008R0828.json  
      inflating: EURLEX57K/dataset/dev/31988R1345.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31988R1345.json  
      inflating: EURLEX57K/dataset/dev/32007R1114.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32007R1114.json  
      inflating: EURLEX57K/dataset/dev/32013R0969.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32013R0969.json  
      inflating: EURLEX57K/dataset/dev/31997R1305.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31997R1305.json  
      inflating: EURLEX57K/dataset/dev/32011R0961.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32011R0961.json  
      inflating: EURLEX57K/dataset/dev/32006D1008.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32006D1008.json  
      inflating: EURLEX57K/dataset/dev/32006R1519.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32006R1519.json  
      inflating: EURLEX57K/dataset/dev/31999R0058.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31999R0058.json  
      inflating: EURLEX57K/dataset/dev/32011D0089.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32011D0089.json  
      inflating: EURLEX57K/dataset/dev/32014R0094.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32014R0094.json  
      inflating: EURLEX57K/dataset/dev/31992R0018.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31992R0018.json  
      inflating: EURLEX57K/dataset/dev/31993R1992.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31993R1992.json  
      inflating: EURLEX57K/dataset/dev/32003R2269.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32003R2269.json  
      inflating: EURLEX57K/dataset/dev/31986D0289.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31986D0289.json  
      inflating: EURLEX57K/dataset/dev/31994R2439.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31994R2439.json  
      inflating: EURLEX57K/dataset/dev/32007R0085.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32007R0085.json  
      inflating: EURLEX57K/dataset/dev/31997D0140.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31997D0140.json  
      inflating: EURLEX57K/dataset/dev/32004R2147.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32004R2147.json  
      inflating: EURLEX57K/dataset/dev/31981R2464.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31981R2464.json  
      inflating: EURLEX57K/dataset/dev/31988R1996.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31988R1996.json  
      inflating: EURLEX57K/dataset/dev/31986R1473.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31986R1473.json  
      inflating: EURLEX57K/dataset/dev/32001R1267.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32001R1267.json  
      inflating: EURLEX57K/dataset/dev/32005R1059.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32005R1059.json  
      inflating: EURLEX57K/dataset/dev/31997D0843.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31997D0843.json  
      inflating: EURLEX57K/dataset/dev/31990R1181.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31990R1181.json  
      inflating: EURLEX57K/dataset/dev/31994R0957.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31994R0957.json  
      inflating: EURLEX57K/dataset/dev/32005R1409.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32005R1409.json  
      inflating: EURLEX57K/dataset/dev/32001R1637.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32001R1637.json  
      inflating: EURLEX57K/dataset/dev/31998R0855.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31998R0855.json  
      inflating: EURLEX57K/dataset/dev/32014D0213.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32014D0213.json  
      inflating: EURLEX57K/dataset/dev/32005R0131.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32005R0131.json  
      inflating: EURLEX57K/dataset/dev/31982R1837.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31982R1837.json  
      inflating: EURLEX57K/dataset/dev/32001R2332.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32001R2332.json  
      inflating: EURLEX57K/dataset/dev/31978D0868.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31978D0868.json  
      inflating: EURLEX57K/dataset/dev/32004R0603.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32004R0603.json  
      inflating: EURLEX57K/dataset/dev/32015R0030.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32015R0030.json  
      inflating: EURLEX57K/dataset/dev/31992R3458.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31992R3458.json  
      inflating: EURLEX57K/dataset/dev/32006R1775.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32006R1775.json  
      inflating: EURLEX57K/dataset/dev/31989D0624.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31989D0624.json  
      inflating: EURLEX57K/dataset/dev/31993R3690.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31993R3690.json  
      inflating: EURLEX57K/dataset/dev/31999R1976.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31999R1976.json  
      inflating: EURLEX57K/dataset/dev/32005R0977.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32005R0977.json  
      inflating: EURLEX57K/dataset/dev/31998D0152.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31998D0152.json  
      inflating: EURLEX57K/dataset/dev/32000R1369.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32000R1369.json  
      inflating: EURLEX57K/dataset/dev/32006R0021.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32006R0021.json  
      inflating: EURLEX57K/dataset/dev/32012R0430.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32012R0430.json  
      inflating: EURLEX57K/dataset/dev/32010R0707.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32010R0707.json  
      inflating: EURLEX57K/dataset/dev/32010D0216.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32010D0216.json  
      inflating: EURLEX57K/dataset/dev/32003R0653.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32003R0653.json  
      inflating: EURLEX57K/dataset/dev/32013R0302.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32013R0302.json  
      inflating: EURLEX57K/dataset/dev/31986D0661.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31986D0661.json  
      inflating: EURLEX57K/dataset/dev/32009D0430.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32009D0430.json  
      inflating: EURLEX57K/dataset/dev/31991R0919.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31991R0919.json  
      inflating: EURLEX57K/dataset/dev/32002R0862.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32002R0862.json  
      inflating: EURLEX57K/dataset/dev/31998D0386.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31998D0386.json  
      inflating: EURLEX57K/dataset/dev/32008D0302.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32008D0302.json  
      inflating: EURLEX57K/dataset/dev/31980R1192.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31980R1192.json  
      inflating: EURLEX57K/dataset/dev/32001R0972.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32001R0972.json  
      inflating: EURLEX57K/dataset/dev/31987L0140.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31987L0140.json  
      inflating: EURLEX57K/dataset/dev/31992R2834.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31992R2834.json  
      inflating: EURLEX57K/dataset/dev/31998R0382.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31998R0382.json  
      inflating: EURLEX57K/dataset/dev/31996R1175.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31996R1175.json  
      inflating: EURLEX57K/dataset/dev/32003R1107.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32003R1107.json  
      inflating: EURLEX57K/dataset/dev/31981R3388.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31981R3388.json  
      inflating: EURLEX57K/dataset/dev/31990R2200.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31990R2200.json  
      inflating: EURLEX57K/dataset/dev/31999R0849.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31999R0849.json  
      inflating: EURLEX57K/dataset/dev/31992R2137.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31992R2137.json  
      inflating: EURLEX57K/dataset/dev/32000D0602.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32000D0602.json  
      inflating: EURLEX57K/dataset/dev/32007D0039.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32007D0039.json  
      inflating: EURLEX57K/dataset/dev/32007R1010.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32007R1010.json  
      inflating: EURLEX57K/dataset/dev/31998D0040.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31998D0040.json  
      inflating: EURLEX57K/dataset/dev/31988D0141.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31988D0141.json  
      inflating: EURLEX57K/dataset/dev/32000R2904.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32000R2904.json  
      inflating: EURLEX57K/dataset/dev/32013R0494.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32013R0494.json  
      inflating: EURLEX57K/dataset/dev/32008R0886.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32008R0886.json  
      inflating: EURLEX57K/dataset/dev/32002R2330.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32002R2330.json  
      inflating: EURLEX57K/dataset/dev/32010D0580.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32010D0580.json  
      inflating: EURLEX57K/dataset/dev/31992D0227.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31992D0227.json  
      inflating: EURLEX57K/dataset/dev/31977D0207.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31977D0207.json  
      inflating: EURLEX57K/dataset/dev/32003R2117.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32003R2117.json  
      inflating: EURLEX57K/dataset/dev/32008R1281.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32008R1281.json  
      inflating: EURLEX57K/dataset/dev/32004D0650.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32004D0650.json  
      inflating: EURLEX57K/dataset/dev/32007R1505.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32007R1505.json  
      inflating: EURLEX57K/dataset/dev/32003R0829.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32003R0829.json  
      inflating: EURLEX57K/dataset/dev/31993R3628.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31993R3628.json  
      inflating: EURLEX57K/dataset/dev/32012D0080.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32012D0080.json  
      inflating: EURLEX57K/dataset/dev/32005D0240.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32005D0240.json  
      inflating: EURLEX57K/dataset/dev/32005R0751.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32005R0751.json  
      inflating: EURLEX57K/dataset/dev/32007R0036.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32007R0036.json  
      inflating: EURLEX57K/dataset/dev/32007D0527.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32007D0527.json  
      inflating: EURLEX57K/dataset/dev/32004R0599.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32004R0599.json  
      inflating: EURLEX57K/dataset/dev/32003R1049.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32003R1049.json  
      inflating: EURLEX57K/dataset/dev/32008R0618.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32008R0618.json  
      inflating: EURLEX57K/dataset/dev/31998D0677.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31998D0677.json  
      inflating: EURLEX57K/dataset/dev/31997R1709.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31997R1709.json  
      inflating: EURLEX57K/dataset/dev/31999D0515.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31999D0515.json  
      inflating: EURLEX57K/dataset/dev/32000R1509.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32000R1509.json  
      inflating: EURLEX57K/dataset/dev/32006R0641.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32006R0641.json  
      inflating: EURLEX57K/dataset/dev/31982D0511.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31982D0511.json  
      inflating: EURLEX57K/dataset/dev/32001R2417.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32001R2417.json  
      inflating: EURLEX57K/dataset/dev/32011R0481.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32011R0481.json  
      inflating: EURLEX57K/dataset/dev/31992R0551.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31992R0551.json  
      inflating: EURLEX57K/dataset/dev/31991D0015.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31991D0015.json  
      inflating: EURLEX57K/dataset/dev/32007R0523.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32007R0523.json  
      inflating: EURLEX57K/dataset/dev/31995D0381.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31995D0381.json  
      inflating: EURLEX57K/dataset/dev/32002R2012.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32002R2012.json  
      inflating: EURLEX57K/dataset/dev/31979R2903.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31979R2903.json  
      inflating: EURLEX57K/dataset/dev/31978R3089.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31978R3089.json  
      inflating: EURLEX57K/dataset/dev/31986R3941.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31986R3941.json  
      inflating: EURLEX57K/dataset/dev/32007R0173.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32007R0173.json  
      inflating: EURLEX57K/dataset/dev/32001R1542.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32001R1542.json  
      inflating: EURLEX57K/dataset/dev/32006R1729.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32006R1729.json  
      inflating: EURLEX57K/dataset/dev/32006R1379.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32006R1379.json  
      inflating: EURLEX57K/dataset/dev/31996L0001.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31996L0001.json  
      inflating: EURLEX57K/dataset/dev/32010R0560.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32010R0560.json  
      inflating: EURLEX57K/dataset/dev/32006D0429.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32006D0429.json  
      inflating: EURLEX57K/dataset/dev/32008R0927.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32008R0927.json  
      inflating: EURLEX57K/dataset/dev/31984R1962.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31984R1962.json  
      inflating: EURLEX57K/dataset/dev/31996D0507.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31996D0507.json  
      inflating: EURLEX57K/dataset/dev/32001R1811.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32001R1811.json  
      inflating: EURLEX57K/dataset/dev/32013D0474.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32013D0474.json  
      inflating: EURLEX57K/dataset/dev/32008D0866.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32008D0866.json  
      inflating: EURLEX57K/dataset/dev/31984R0220.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31984R0220.json  
      inflating: EURLEX57K/dataset/dev/32003D0125.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32003D0125.json  
      inflating: EURLEX57K/dataset/dev/32008D0889.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32008D0889.json  
      inflating: EURLEX57K/dataset/dev/32006D0096.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32006D0096.json  
      inflating: EURLEX57K/dataset/dev/32012D0203.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32012D0203.json  
      inflating: EURLEX57K/dataset/dev/32002R0243.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32002R0243.json  
      inflating: EURLEX57K/dataset/dev/32009R1042.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32009R1042.json  
      inflating: EURLEX57K/dataset/dev/32012R0342.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32012R0342.json  
      inflating: EURLEX57K/dataset/dev/32001R2381.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32001R2381.json  
      inflating: EURLEX57K/dataset/dev/32001R0646.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32001R0646.json  
      inflating: EURLEX57K/dataset/dev/31997R0661.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31997R0661.json  
      inflating: EURLEX57K/dataset/dev/31998R2072.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31998R2072.json  
      inflating: EURLEX57K/dataset/dev/31998R2588.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31998R2588.json  
      inflating: EURLEX57K/dataset/dev/32013R1261.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32013R1261.json  
      inflating: EURLEX57K/dataset/dev/31994R1475.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31994R1475.json  
      inflating: EURLEX57K/dataset/dev/31995R2992.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31995R2992.json  
      inflating: EURLEX57K/dataset/dev/31999R0053.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31999R0053.json  
      inflating: EURLEX57K/dataset/dev/32005R1117.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32005R1117.json  
      inflating: EURLEX57K/dataset/dev/31992L0004.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31992L0004.json  
      inflating: EURLEX57K/dataset/dev/32014D0215(01).json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32014D0215(01).json  
      inflating: EURLEX57K/dataset/dev/31985R0683.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31985R0683.json  
      inflating: EURLEX57K/dataset/dev/31995R0628.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31995R0628.json  
      inflating: EURLEX57K/dataset/dev/31985R2651.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31985R2651.json  
      inflating: EURLEX57K/dataset/dev/31986D0328.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31986D0328.json  
      inflating: EURLEX57K/dataset/dev/32001R2010.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32001R2010.json  
      inflating: EURLEX57K/dataset/dev/32004R0521.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32004R0521.json  
      inflating: EURLEX57K/dataset/dev/31995R2750.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31995R2750.json  
      inflating: EURLEX57K/dataset/dev/32005R1402.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32005R1402.json  
      inflating: EURLEX57K/dataset/dev/31987R4148.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31987R4148.json  
      inflating: EURLEX57K/dataset/dev/31998R0624.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31998R0624.json  
      inflating: EURLEX57K/dataset/dev/31993R3048.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31993R3048.json  
      inflating: EURLEX57K/dataset/dev/32011L0091.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32011L0091.json  
      inflating: EURLEX57K/dataset/dev/32014R0973.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32014R0973.json  
      inflating: EURLEX57K/dataset/dev/32009R1150.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32009R1150.json  
      inflating: EURLEX57K/dataset/dev/32004R0608.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32004R0608.json  
      inflating: EURLEX57K/dataset/dev/31996D0100.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31996D0100.json  
      inflating: EURLEX57K/dataset/dev/32014D0648.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32014D0648.json  
      inflating: EURLEX57K/dataset/dev/32006R0495.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32006R0495.json  
      inflating: EURLEX57K/dataset/dev/32012D0311.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32012D0311.json  
      inflating: EURLEX57K/dataset/dev/32013R0562.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32013R0562.json  
      inflating: EURLEX57K/dataset/dev/31997R1498.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31997R1498.json  
      inflating: EURLEX57K/dataset/dev/31996L0056.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31996L0056.json  
      inflating: EURLEX57K/dataset/dev/32011D1105.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32011D1105.json  
      inflating: EURLEX57K/dataset/dev/32005R0993.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32005R0993.json  
      inflating: EURLEX57K/dataset/dev/31995R1605.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31995R1605.json  
      inflating: EURLEX57K/dataset/dev/31996D0719(01).json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31996D0719(01).json  
      inflating: EURLEX57K/dataset/dev/31999R2717.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31999R2717.json  
      inflating: EURLEX57K/dataset/dev/32001R1450.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32001R1450.json  
      inflating: EURLEX57K/dataset/dev/32010R0022.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32010R0022.json  
      inflating: EURLEX57K/dataset/dev/32001R1903.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32001R1903.json  
      inflating: EURLEX57K/dataset/dev/31994R1971.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31994R1971.json  
      inflating: EURLEX57K/dataset/dev/32000D0062.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32000D0062.json  
      inflating: EURLEX57K/dataset/dev/31982R1979.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31982R1979.json  
      inflating: EURLEX57K/dataset/dev/31986R2468.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31986R2468.json  
      inflating: EURLEX57K/dataset/dev/31994D0722.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31994D0722.json  
      inflating: EURLEX57K/dataset/dev/31995D0440.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31995D0440.json  
      inflating: EURLEX57K/dataset/dev/31985R1987.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31985R1987.json  
      inflating: EURLEX57K/dataset/dev/32004R0472.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32004R0472.json  
      inflating: EURLEX57K/dataset/dev/31993R1825.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31993R1825.json  
      inflating: EURLEX57K/dataset/dev/32006D0254.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32006D0254.json  
      inflating: EURLEX57K/dataset/dev/31999R1654.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31999R1654.json  
      inflating: EURLEX57K/dataset/dev/31997D0448.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31997D0448.json  
      inflating: EURLEX57K/dataset/dev/32005R0340.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32005R0340.json  
      inflating: EURLEX57K/dataset/dev/32002R1690.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32002R1690.json  
      inflating: EURLEX57K/dataset/dev/32003R1008.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32003R1008.json  
      inflating: EURLEX57K/dataset/dev/32004R1263.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32004R1263.json  
      inflating: EURLEX57K/dataset/dev/31996R1580.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31996R1580.json  
      inflating: EURLEX57K/dataset/dev/31994R3370.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31994R3370.json  
      inflating: EURLEX57K/dataset/dev/31999D0041.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31999D0041.json  
      inflating: EURLEX57K/dataset/dev/31993R1433.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31993R1433.json  
      inflating: EURLEX57K/dataset/dev/32011R0993.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32011R0993.json  
      inflating: EURLEX57K/dataset/dev/32005R1414.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32005R1414.json  
      inflating: EURLEX57K/dataset/dev/31985R3056.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31985R3056.json  
      inflating: EURLEX57K/dataset/dev/32012R0495.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32012R0495.json  
      inflating: EURLEX57K/dataset/dev/31989R1610.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31989R1610.json  
      inflating: EURLEX57K/dataset/dev/31989R2895.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31989R2895.json  
      inflating: EURLEX57K/dataset/dev/31986R3900.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31986R3900.json  
      inflating: EURLEX57K/dataset/dev/31988R2427.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31988R2427.json  
      inflating: EURLEX57K/dataset/dev/31989R3857.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31989R3857.json  
      inflating: EURLEX57K/dataset/dev/32012R1007.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32012R1007.json  
      inflating: EURLEX57K/dataset/dev/32002R1106.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32002R1106.json  
      inflating: EURLEX57K/dataset/dev/32013R0827.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32013R0827.json  
      inflating: EURLEX57K/dataset/dev/31999R2644.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31999R2644.json  
      inflating: EURLEX57K/dataset/dev/32003R1264.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32003R1264.json  
      inflating: EURLEX57K/dataset/dev/32009R0757.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32009R0757.json  
      inflating: EURLEX57K/dataset/dev/32008R0966.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32008R0966.json  
      inflating: EURLEX57K/dataset/dev/32006R0179.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32006R0179.json  
      inflating: EURLEX57K/dataset/dev/31990R2163.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31990R2163.json  
      inflating: EURLEX57K/dataset/dev/31997R2308.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31997R2308.json  
      inflating: EURLEX57K/dataset/dev/32010R0521.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32010R0521.json  
      inflating: EURLEX57K/dataset/dev/31987D0375.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31987D0375.json  
      inflating: EURLEX57K/dataset/dev/32003R2018.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32003R2018.json  
      inflating: EURLEX57K/dataset/dev/31984R1889.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31984R1889.json  
      inflating: EURLEX57K/dataset/dev/32002D0206.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32002D0206.json  
      inflating: EURLEX57K/dataset/dev/31987R2209.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31987R2209.json  
      inflating: EURLEX57K/dataset/dev/32003R0160.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32003R0160.json  
      inflating: EURLEX57K/dataset/dev/31991R2344.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31991R2344.json  
      inflating: EURLEX57K/dataset/dev/32012R0753.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32012R0753.json  
      inflating: EURLEX57K/dataset/dev/32006R2001.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32006R2001.json  
      inflating: EURLEX57K/dataset/dev/32010R0464.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32010R0464.json  
      inflating: EURLEX57K/dataset/dev/32004R1849.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32004R1849.json  
      inflating: EURLEX57K/dataset/dev/31981D0293.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31981D0293.json  
      inflating: EURLEX57K/dataset/dev/32012R1142.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32012R1142.json  
      inflating: EURLEX57K/dataset/dev/32005R1678.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32005R1678.json  
      inflating: EURLEX57K/dataset/dev/32010R0937.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32010R0937.json  
      inflating: EURLEX57K/dataset/dev/31985R1179.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31985R1179.json  
      inflating: EURLEX57K/dataset/dev/31987L0018.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31987L0018.json  
      inflating: EURLEX57K/dataset/dev/31987R2931.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31987R2931.json  
      inflating: EURLEX57K/dataset/dev/31996D0287.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31996D0287.json  
      inflating: EURLEX57K/dataset/dev/32014D0035.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32014D0035.json  
      inflating: EURLEX57K/dataset/dev/32007D0531.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32007D0531.json  
      inflating: EURLEX57K/dataset/dev/31992D0543.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31992D0543.json  
      inflating: EURLEX57K/dataset/dev/32012D0096.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32012D0096.json  
      inflating: EURLEX57K/dataset/dev/32001D0592.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32001D0592.json  
      inflating: EURLEX57K/dataset/dev/32004R0130.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32004R0130.json  
      inflating: EURLEX57K/dataset/dev/32005D0743.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32005D0743.json  
      inflating: EURLEX57K/dataset/dev/32014R0031.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32014R0031.json  
      inflating: EURLEX57K/dataset/dev/32001R1387.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32001R1387.json  
      inflating: EURLEX57K/dataset/dev/32014D0873.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32014D0873.json  
      inflating: EURLEX57K/dataset/dev/31982D0854.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31982D0854.json  
      inflating: EURLEX57K/dataset/dev/31981R3395.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31981R3395.json  
      inflating: EURLEX57K/dataset/dev/32011R1286.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32011R1286.json  
      inflating: EURLEX57K/dataset/dev/32009R1111.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32009R1111.json  
      inflating: EURLEX57K/dataset/dev/31991R2606.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31991R2606.json  
      inflating: EURLEX57K/dataset/dev/32010D0437.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32010D0437.json  
      inflating: EURLEX57K/dataset/dev/31996R1241.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31996R1241.json  
      inflating: EURLEX57K/dataset/dev/31995L0042.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31995L0042.json  
      inflating: EURLEX57K/dataset/dev/31989R2712.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31989R2712.json  
      inflating: EURLEX57K/dataset/dev/31991R1180.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31991R1180.json  
      inflating: EURLEX57K/dataset/dev/31990R3230.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31990R3230.json  
      inflating: EURLEX57K/dataset/dev/32000D0023.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32000D0023.json  
      inflating: EURLEX57K/dataset/dev/32003D0076.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32003D0076.json  
      inflating: EURLEX57K/dataset/dev/31995D0051.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31995D0051.json  
      inflating: EURLEX57K/dataset/dev/32002R0255.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32002R0255.json  
      inflating: EURLEX57K/dataset/dev/31986R2429.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31986R2429.json  
      inflating: EURLEX57K/dataset/dev/32003R0137.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32003R0137.json  
      inflating: EURLEX57K/dataset/dev/32007D0618.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32007D0618.json  
      inflating: EURLEX57K/dataset/dev/32006R0591.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32006R0591.json  
      inflating: EURLEX57K/dataset/dev/31984D0025.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31984D0025.json  
      inflating: EURLEX57K/dataset/dev/32001D0506.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32001D0506.json  
      inflating: EURLEX57K/dataset/dev/32011R0116.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32011R0116.json  
      inflating: EURLEX57K/dataset/dev/31991R0093.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31991R0093.json  
      inflating: EURLEX57K/dataset/dev/32002R0412.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32002R0412.json  
      inflating: EURLEX57K/dataset/dev/31981R0038.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31981R0038.json  
      inflating: EURLEX57K/dataset/dev/32011R1357.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32011R1357.json  
      inflating: EURLEX57K/dataset/dev/32000R1564.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32000R1564.json  
      inflating: EURLEX57K/dataset/dev/31984R1325.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31984R1325.json  
      inflating: EURLEX57K/dataset/dev/31999R0069.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31999R0069.json  
      inflating: EURLEX57K/dataset/dev/31989D0029.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31989D0029.json  
      inflating: EURLEX57K/dataset/dev/32009R0147.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32009R0147.json  
      inflating: EURLEX57K/dataset/dev/32008R0675.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32008R0675.json  
      inflating: EURLEX57K/dataset/dev/32011R0950.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32011R0950.json  
      inflating: EURLEX57K/dataset/dev/32011R0403.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32011R0403.json  
      inflating: EURLEX57K/dataset/dev/32012R0006.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32012R0006.json  
      inflating: EURLEX57K/dataset/dev/31971R1592.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31971R1592.json  
      inflating: EURLEX57K/dataset/dev/32005L0052.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32005L0052.json  
      inflating: EURLEX57K/dataset/dev/31995R1280.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31995R1280.json  
      inflating: EURLEX57K/dataset/dev/32007R1563.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32007R1563.json  
      inflating: EURLEX57K/dataset/dev/32011D0391.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32011D0391.json  
      inflating: EURLEX57K/dataset/dev/32005D0104.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32005D0104.json  
      inflating: EURLEX57K/dataset/dev/31990R0467.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31990R0467.json  
      inflating: EURLEX57K/dataset/dev/31994D0348.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31994D0348.json  
      inflating: EURLEX57K/dataset/dev/32002R0384.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32002R0384.json  
      inflating: EURLEX57K/dataset/dev/32000D0058.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32000D0058.json  
      inflating: EURLEX57K/dataset/dev/32010R0448.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32010R0448.json  
      inflating: EURLEX57K/dataset/dev/31995D0180.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31995D0180.json  
      inflating: EURLEX57K/dataset/dev/32007R0667.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32007R0667.json  
      inflating: EURLEX57K/dataset/dev/32014R0699.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32014R0699.json  
      inflating: EURLEX57K/dataset/dev/32001D0785.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32001D0785.json  
      inflating: EURLEX57K/dataset/dev/31997R0749.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31997R0749.json  
      inflating: EURLEX57K/dataset/dev/31993R0577.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31993R0577.json  
      inflating: EURLEX57K/dataset/dev/32001R2303.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32001R2303.json  
      inflating: EURLEX57K/dataset/dev/32013D0049.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32013D0049.json  
      inflating: EURLEX57K/dataset/dev/32007R1076.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32007R1076.json  
      inflating: EURLEX57K/dataset/dev/32003R1248.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32003R1248.json  
      inflating: EURLEX57K/dataset/dev/32005R1341.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32005R1341.json  
      inflating: EURLEX57K/dataset/dev/32005D0942.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32005D0942.json  
      inflating: EURLEX57K/dataset/dev/32008D0558.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32008D0558.json  
      inflating: EURLEX57K/dataset/dev/31990R1763.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31990R1763.json  
      inflating: EURLEX57K/dataset/dev/31989D0091.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31989D0091.json  
      inflating: EURLEX57K/dataset/dev/31977R0818.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31977R0818.json  
      inflating: EURLEX57K/dataset/dev/31992L0086.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31992L0086.json  
      inflating: EURLEX57K/dataset/dev/32003D0236.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32003D0236.json  
      inflating: EURLEX57K/dataset/dev/32007R0519.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32007R0519.json  
      inflating: EURLEX57K/dataset/dev/32006D0690.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32006D0690.json  
      inflating: EURLEX57K/dataset/dev/31985D0310.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31985D0310.json  
      inflating: EURLEX57K/dataset/dev/31986R0654.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31986R0654.json  
      inflating: EURLEX57K/dataset/dev/32005R1896.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32005R1896.json  
      inflating: EURLEX57K/dataset/dev/32003R1970.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32003R1970.json  
      inflating: EURLEX57K/dataset/dev/32003R0398.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32003R0398.json  
      inflating: EURLEX57K/dataset/dev/32008R1033.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32008R1033.json  
      inflating: EURLEX57K/dataset/dev/31995R2382.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31995R2382.json  
      inflating: EURLEX57K/dataset/dev/31997D0063.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31997D0063.json  
      inflating: EURLEX57K/dataset/dev/32002D0441.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32002D0441.json  
      inflating: EURLEX57K/dataset/dev/31994R0177.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31994R0177.json  
      inflating: EURLEX57K/dataset/dev/32010R0736.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32010R0736.json  
      inflating: EURLEX57K/dataset/dev/31989R1684.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31989R1684.json  
      inflating: EURLEX57K/dataset/dev/32005R1480.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32005R1480.json  
      inflating: EURLEX57K/dataset/dev/31981R2814.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31981R2814.json  
      inflating: EURLEX57K/dataset/dev/32011D0846.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32011D0846.json  
      inflating: EURLEX57K/dataset/dev/31985R3168.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31985R3168.json  
      inflating: EURLEX57K/dataset/dev/32001R2211.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32001R2211.json  
      inflating: EURLEX57K/dataset/dev/32006R0047.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32006R0047.json  
      inflating: EURLEX57K/dataset/dev/32014D0611(02).json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32014D0611(02).json  
      inflating: EURLEX57K/dataset/dev/32005R2185.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32005R2185.json  
      inflating: EURLEX57K/dataset/dev/32014D0760.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32014D0760.json  
      inflating: EURLEX57K/dataset/dev/32012D0393.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32012D0393.json  
      inflating: EURLEX57K/dataset/dev/32012R0682.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32012R0682.json  
      inflating: EURLEX57K/dataset/dev/32009R0239.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32009R0239.json  
      inflating: EURLEX57K/dataset/dev/31995R2802.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31995R2802.json  
      inflating: EURLEX57K/dataset/dev/32009D0728.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32009D0728.json  
      inflating: EURLEX57K/dataset/dev/31985R0894.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31985R0894.json  
      inflating: EURLEX57K/dataset/dev/31989D0357.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31989D0357.json  
      inflating: EURLEX57K/dataset/dev/31986R3214.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31986R3214.json  
      inflating: EURLEX57K/dataset/dev/31989R2381.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31989R2381.json  
      inflating: EURLEX57K/dataset/dev/31990L0427.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31990L0427.json  
      inflating: EURLEX57K/dataset/dev/32013R0919.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32013R0919.json  
      inflating: EURLEX57K/dataset/dev/31995R0995.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31995R0995.json  
      inflating: EURLEX57K/dataset/dev/32009R0686.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32009R0686.json  
      inflating: EURLEX57K/dataset/dev/32007R1471.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32007R1471.json  
      inflating: EURLEX57K/dataset/dev/31999D0313.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31999D0313.json  
      inflating: EURLEX57K/dataset/dev/32014D0275.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32014D0275.json  
      inflating: EURLEX57K/dataset/dev/31995D0092.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31995D0092.json  
      inflating: EURLEX57K/dataset/dev/31989R1112.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31989R1112.json  
      inflating: EURLEX57K/dataset/dev/32011R0638.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32011R0638.json  
      inflating: EURLEX57K/dataset/dev/32006R0552.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32006R0552.json  
      inflating: EURLEX57K/dataset/dev/32004R1977.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32004R1977.json  
      inflating: EURLEX57K/dataset/dev/31992D0353.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31992D0353.json  
      inflating: EURLEX57K/dataset/dev/32009R0940.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32009R0940.json  
      inflating: EURLEX57K/dataset/dev/31980R1859.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31980R1859.json  
      inflating: EURLEX57K/dataset/dev/32011R0507.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32011R0507.json  
      inflating: EURLEX57K/dataset/dev/31976R0795.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31976R0795.json  
      inflating: EURLEX57K/dataset/dev/31995D0207.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31995D0207.json  
      inflating: EURLEX57K/dataset/dev/32003R1823.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32003R1823.json  
      inflating: EURLEX57K/dataset/dev/32010R0235.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32010R0235.json  
      inflating: EURLEX57K/dataset/dev/32010R1074.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32010R1074.json  
      inflating: EURLEX57K/dataset/dev/32014D0849.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32014D0849.json  
      inflating: EURLEX57K/dataset/dev/31981R3205.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31981R3205.json  
      inflating: EURLEX57K/dataset/dev/31987R1761.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31987R1761.json  
      inflating: EURLEX57K/dataset/dev/31993R3033.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31993R3033.json  
      inflating: EURLEX57K/dataset/dev/32001R1702.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32001R1702.json  
      inflating: EURLEX57K/dataset/dev/32002R1757.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32002R1757.json  
      inflating: EURLEX57K/dataset/dev/31994R1770.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31994R1770.json  
      inflating: EURLEX57K/dataset/dev/32012L0050.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32012L0050.json  
      inflating: EURLEX57K/dataset/dev/31989D0068.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31989D0068.json  
      inflating: EURLEX57K/dataset/dev/32001R1352.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32001R1352.json  
      inflating: EURLEX57K/dataset/dev/32008R0634.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32008R0634.json  
      inflating: EURLEX57K/dataset/dev/31980R3472.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31980R3472.json  
      inflating: EURLEX57K/dataset/dev/32005R1496.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32005R1496.json  
      inflating: EURLEX57K/dataset/dev/31992D0083.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31992D0083.json  
      inflating: EURLEX57K/dataset/dev/32011D0153.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32011D0153.json  
      inflating: EURLEX57K/dataset/dev/32002R0516.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32002R0516.json  
      inflating: EURLEX57K/dataset/dev/32005R1245.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32005R1245.json  
      inflating: EURLEX57K/dataset/dev/31999D0240.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31999D0240.json  
      inflating: EURLEX57K/dataset/dev/32007R1522.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32007R1522.json  
      inflating: EURLEX57K/dataset/dev/32007D0272.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32007D0272.json  
      inflating: EURLEX57K/dataset/dev/32005R0004.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32005R0004.json  
      inflating: EURLEX57K/dataset/dev/32010D0118.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32010D0118.json  
      inflating: EURLEX57K/dataset/dev/32004R0366.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32004R0366.json  
      inflating: EURLEX57K/dataset/dev/32002R0795.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32002R0795.json  
      inflating: EURLEX57K/dataset/dev/31992D0345.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31992D0345.json  
      inflating: EURLEX57K/dataset/dev/32007R0626.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32007R0626.json  
      inflating: EURLEX57K/dataset/dev/31990D0588.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31990D0588.json  
      inflating: EURLEX57K/dataset/dev/32005R0511.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32005R0511.json  
      inflating: EURLEX57K/dataset/dev/32001R1997.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32001R1997.json  
      inflating: EURLEX57K/dataset/dev/31998D0067.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31998D0067.json  
      inflating: EURLEX57K/dataset/dev/31989R3990.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31989R3990.json  
      inflating: EURLEX57K/dataset/dev/31999D0305.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31999D0305.json  
      inflating: EURLEX57K/dataset/dev/32004R1432.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32004R1432.json  
      inflating: EURLEX57K/dataset/dev/31995R2951.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31995R2951.json  
      inflating: EURLEX57K/dataset/dev/31981R1111.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31981R1111.json  
      inflating: EURLEX57K/dataset/dev/32008R0726.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32008R0726.json  
      inflating: EURLEX57K/dataset/dev/31984L0535.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31984L0535.json  
      inflating: EURLEX57K/dataset/dev/31984R1763.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31984R1763.json  
      inflating: EURLEX57K/dataset/dev/31986R3193.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31986R3193.json  
      inflating: EURLEX57K/dataset/dev/32008D0667.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32008D0667.json  
      inflating: EURLEX57K/dataset/dev/31997D0167.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31997D0167.json  
      inflating: EURLEX57K/dataset/dev/32001D0140.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32001D0140.json  
      inflating: EURLEX57K/dataset/dev/32009L0003.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32009L0003.json  
      inflating: EURLEX57K/dataset/dev/32012D0444.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32012D0444.json  
      inflating: EURLEX57K/dataset/dev/32000D0222.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32000D0222.json  
      inflating: EURLEX57K/dataset/dev/32001D0510.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32001D0510.json  
      inflating: EURLEX57K/dataset/dev/31997R0026.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31997R0026.json  
      inflating: EURLEX57K/dataset/dev/31999R2804.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31999R2804.json  
      inflating: EURLEX57K/dataset/dev/32015R0294.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32015R0294.json  
      inflating: EURLEX57K/dataset/dev/32011D0041.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32011D0041.json  
      inflating: EURLEX57K/dataset/dev/31996R0344.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31996R0344.json  
      inflating: EURLEX57K/dataset/dev/32013D0399.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32013D0399.json  
      inflating: EURLEX57K/dataset/dev/32011R0045.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32011R0045.json  
      inflating: EURLEX57K/dataset/dev/31989R2840.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31989R2840.json  
      inflating: EURLEX57K/dataset/dev/31987R0062.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31987R0062.json  
      inflating: EURLEX57K/dataset/dev/31971D0057.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31971D0057.json  
      inflating: EURLEX57K/dataset/dev/32014R0119.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32014R0119.json  
      inflating: EURLEX57K/dataset/dev/32005R1838.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32005R1838.json  
      inflating: EURLEX57K/dataset/dev/32014D0408.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32014D0408.json  
      inflating: EURLEX57K/dataset/dev/31992D0484.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31992D0484.json  
      inflating: EURLEX57K/dataset/dev/31987D0089.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31987D0089.json  
      inflating: EURLEX57K/dataset/dev/32013D0663.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32013D0663.json  
      inflating: EURLEX57K/dataset/dev/32003R0273.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32003R0273.json  
      inflating: EURLEX57K/dataset/dev/31986D0241.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31986D0241.json  
      inflating: EURLEX57K/dataset/dev/32010R0777.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32010R0777.json  
      inflating: EURLEX57K/dataset/dev/32002R0812.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32002R0812.json  
      inflating: EURLEX57K/dataset/dev/32015R0502.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32015R0502.json  
      inflating: EURLEX57K/dataset/dev/32008R0819.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32008R0819.json  
      inflating: EURLEX57K/dataset/dev/31994R2222.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31994R2222.json  
      inflating: EURLEX57K/dataset/dev/31985D0205(01).json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31985D0205(01).json  
      inflating: EURLEX57K/dataset/dev/32014D0371.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32014D0371.json  
      inflating: EURLEX57K/dataset/dev/32001R2250.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32001R2250.json  
      inflating: EURLEX57K/dataset/dev/32004D0786(01).json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32004D0786(01).json  
      inflating: EURLEX57K/dataset/dev/32002R0392.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32002R0392.json  
      inflating: EURLEX57K/dataset/dev/32005R0403.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32005R0403.json  
      inflating: EURLEX57K/dataset/dev/31996R0082.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31996R0082.json  
      inflating: EURLEX57K/dataset/dev/32002R1429.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32002R1429.json  
      inflating: EURLEX57K/dataset/dev/32005R0950.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32005R0950.json  
      inflating: EURLEX57K/dataset/dev/32006L0011.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32006L0011.json  
      inflating: EURLEX57K/dataset/dev/31993R3208.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31993R3208.json  
      inflating: EURLEX57K/dataset/dev/31993R1235.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31993R1235.json  
      inflating: EURLEX57K/dataset/dev/32006R1752.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32006R1752.json  
      inflating: EURLEX57K/dataset/dev/31993R0832.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31993R0832.json  
      inflating: EURLEX57K/dataset/dev/32004R1465.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32004R1465.json  
      inflating: EURLEX57K/dataset/dev/32007R1430.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32007R1430.json  
      inflating: EURLEX57K/dataset/dev/31998R0521.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31998R0521.json  
      inflating: EURLEX57K/dataset/dev/32015R0447.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32015R0447.json  
      inflating: EURLEX57K/dataset/dev/31996R2050.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31996R2050.json  
      inflating: EURLEX57K/dataset/dev/31979R1084.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31979R1084.json  
      inflating: EURLEX57K/dataset/dev/31974D0367.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31974D0367.json  
      inflating: EURLEX57K/dataset/dev/31992D0312.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31992D0312.json  
      inflating: EURLEX57K/dataset/dev/31986R0496.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31986R0496.json  
      inflating: EURLEX57K/dataset/dev/31995R0468.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31995R0468.json  
      inflating: EURLEX57K/dataset/dev/32004R1941.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32004R1941.json  
      inflating: EURLEX57K/dataset/dev/31987R2244.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31987R2244.json  
      inflating: EURLEX57K/dataset/dev/31998R1347.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31998R1347.json  
      inflating: EURLEX57K/dataset/dev/32003R0438.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32003R0438.json  
      inflating: EURLEX57K/dataset/dev/31997R0728.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31997R0728.json  
      inflating: EURLEX57K/dataset/dev/31993R2481.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31993R2481.json  
      inflating: EURLEX57K/dataset/dev/32003D0083.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32003D0083.json  
      inflating: EURLEX57K/dataset/dev/32014D0243.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32014D0243.json  
      inflating: EURLEX57K/dataset/dev/31992R3058.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31992R3058.json  
      inflating: EURLEX57K/dataset/dev/31991L0266.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31991L0266.json  
      inflating: EURLEX57K/dataset/dev/32008D0493.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32008D0493.json  
      inflating: EURLEX57K/dataset/dev/31997R1139.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31997R1139.json  
      inflating: EURLEX57K/dataset/dev/31966R0122.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31966R0122.json  
      inflating: EURLEX57K/dataset/dev/31994D0329.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31994D0329.json  
      inflating: EURLEX57K/dataset/dev/31992D0220.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31992D0220.json  
      inflating: EURLEX57K/dataset/dev/31997D0686.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31997D0686.json  
      inflating: EURLEX57K/dataset/dev/31980D0446.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31980D0446.json  
      inflating: EURLEX57K/dataset/dev/31984R2354.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31984R2354.json  
      inflating: EURLEX57K/dataset/dev/31976D0162.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31976D0162.json  
      inflating: EURLEX57K/dataset/dev/31984D0228.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31984D0228.json  
      inflating: EURLEX57K/dataset/dev/31995R2137.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31995R2137.json  
      inflating: EURLEX57K/dataset/dev/32003R2110.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32003R2110.json  
      inflating: EURLEX57K/dataset/dev/32008D0755.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32008D0755.json  
      inflating: EURLEX57K/dataset/dev/32009R0126.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32009R0126.json  
      inflating: EURLEX57K/dataset/dev/32014D0886.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32014D0886.json  
      inflating: EURLEX57K/dataset/dev/32000R0251.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32000R0251.json  
      inflating: EURLEX57K/dataset/dev/31982D0458.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31982D0458.json  
      inflating: EURLEX57K/dataset/dev/32002R0536.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32002R0536.json  
      inflating: EURLEX57K/dataset/dev/32004R2117.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32004R2117.json  
      inflating: EURLEX57K/dataset/dev/31987D0011.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31987D0011.json  
      inflating: EURLEX57K/dataset/dev/31985R0637.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31985R0637.json  
      inflating: EURLEX57K/dataset/dev/32012R0088.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32012R0088.json  
      inflating: EURLEX57K/dataset/dev/32003R0711.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32003R0711.json  
      inflating: EURLEX57K/dataset/dev/31992R2560.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31992R2560.json  
      inflating: EURLEX57K/dataset/dev/32000R0744.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32000R0744.json  
      inflating: EURLEX57K/dataset/dev/32010R0215.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32010R0215.json  
      inflating: EURLEX57K/dataset/dev/32005R1459.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32005R1459.json  
      inflating: EURLEX57K/dataset/dev/32002R1632.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32002R1632.json  
      inflating: EURLEX57K/dataset/dev/32008R0301.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32008R0301.json  
      inflating: EURLEX57K/dataset/dev/32009R0433.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32009R0433.json  
      inflating: EURLEX57K/dataset/dev/32011R0874.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32011R0874.json  
      inflating: EURLEX57K/dataset/dev/32008R0585.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32008R0585.json  
      inflating: EURLEX57K/dataset/dev/31991R1437.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31991R1437.json  
      inflating: EURLEX57K/dataset/dev/32005R1377.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32005R1377.json  
      inflating: EURLEX57K/dataset/dev/31993R2985.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31993R2985.json  
      inflating: EURLEX57K/dataset/dev/31990D0455.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31990D0455.json  
      inflating: EURLEX57K/dataset/dev/31992D0298.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31992D0298.json  
      inflating: EURLEX57K/dataset/dev/32001R2335.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32001R2335.json  
      inflating: EURLEX57K/dataset/dev/32006R0499.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32006R0499.json  
      inflating: EURLEX57K/dataset/dev/31981R0327.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31981R0327.json  
      inflating: EURLEX57K/dataset/dev/32010R0491.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32010R0491.json  
      inflating: EURLEX57K/dataset/dev/31991D0367.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31991D0367.json  
      inflating: EURLEX57K/dataset/dev/31993R2593.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31993R2593.json  
      inflating: EURLEX57K/dataset/dev/32002R2225.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32002R2225.json  
      inflating: EURLEX57K/dataset/dev/32006D0537.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32006D0537.json  
      inflating: EURLEX57K/dataset/dev/31994R2202.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31994R2202.json  
      inflating: EURLEX57K/dataset/dev/32004R1853.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32004R1853.json  
      inflating: EURLEX57K/dataset/dev/32007D0205.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32007D0205.json  
      inflating: EURLEX57K/dataset/dev/32005R0423.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32005R0423.json  
      inflating: EURLEX57K/dataset/dev/32013R1390.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32013R1390.json  
      inflating: EURLEX57K/dataset/dev/32002R1059.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32002R1059.json  
      inflating: EURLEX57K/dataset/dev/31991R2427.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31991R2427.json  
      inflating: EURLEX57K/dataset/dev/31995D0335.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31995D0335.json  
      inflating: EURLEX57K/dataset/dev/32000R2391.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32000R2391.json  
      inflating: EURLEX57K/dataset/dev/31983D0297.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31983D0297.json  
      inflating: EURLEX57K/dataset/dev/32004R0192.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32004R0192.json  
      inflating: EURLEX57K/dataset/dev/32013R0352.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32013R0352.json  
      inflating: EURLEX57K/dataset/dev/32002R0561.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32002R0561.json  
      inflating: EURLEX57K/dataset/dev/32012R0460.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32012R0460.json  
      inflating: EURLEX57K/dataset/dev/31984R0447.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31984R0447.json  
      inflating: EURLEX57K/dataset/dev/31992R2022.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31992R2022.json  
      inflating: EURLEX57K/dataset/dev/31986R1862.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31986R1862.json  
      inflating: EURLEX57K/dataset/dev/31997D0452.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31997D0452.json  
      inflating: EURLEX57K/dataset/dev/32010D0246.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32010D0246.json  
      inflating: EURLEX57K/dataset/dev/31984L0450.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31984L0450.json  
      inflating: EURLEX57K/dataset/dev/32011R0966.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32011R0966.json  
      inflating: EURLEX57K/dataset/dev/32001R1325.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32001R1325.json  
      inflating: EURLEX57K/dataset/dev/32014R1378.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32014R1378.json  
      inflating: EURLEX57K/dataset/dev/32007R0981.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32007R0981.json  
      inflating: EURLEX57K/dataset/dev/32014R1397.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32014R1397.json  
      inflating: EURLEX57K/dataset/dev/31988R0729.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31988R0729.json  
      inflating: EURLEX57K/dataset/dev/32011R1361.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32011R1361.json  
      inflating: EURLEX57K/dataset/dev/31997D0147.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31997D0147.json  
      inflating: EURLEX57K/dataset/dev/31994R0403.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31994R0403.json  
      inflating: EURLEX57K/dataset/dev/31999R0819.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31999R0819.json  
      inflating: EURLEX57K/dataset/dev/32010D0303.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32010D0303.json  
      inflating: EURLEX57K/dataset/dev/32003R0316.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32003R0316.json  
      inflating: EURLEX57K/dataset/dev/31992R2167.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31992R2167.json  
      inflating: EURLEX57K/dataset/dev/32011R0570.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32011R0570.json  
      inflating: EURLEX57K/dataset/dev/31987R0238.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31987R0238.json  
      inflating: EURLEX57K/dataset/dev/32000R2411.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32000R2411.json  
      inflating: EURLEX57K/dataset/dev/32004R0612.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32004R0612.json  
      inflating: EURLEX57K/dataset/dev/32013D0439.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32013D0439.json  
      inflating: EURLEX57K/dataset/dev/32006D0034.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32006D0034.json  
      inflating: EURLEX57K/dataset/dev/32004R1453.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32004R1453.json  
      inflating: EURLEX57K/dataset/dev/31999D0221.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31999D0221.json  
      inflating: EURLEX57K/dataset/dev/32005L0072.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32005L0072.json  
      inflating: EURLEX57K/dataset/dev/31993L0055.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31993L0055.json  
      inflating: EURLEX57K/dataset/dev/32007R1113.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32007R1113.json  
      inflating: EURLEX57K/dataset/dev/32008R0086.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32008R0086.json  
      inflating: EURLEX57K/dataset/dev/32006R0899.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32006R0899.json  
      inflating: EURLEX57K/dataset/dev/32000R1378.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32000R1378.json  
      inflating: EURLEX57K/dataset/dev/32004R1516.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32004R1516.json  
      inflating: EURLEX57K/dataset/dev/32000D0078.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32000D0078.json  
      inflating: EURLEX57K/dataset/dev/32010D0179.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32010D0179.json  
      inflating: EURLEX57K/dataset/dev/32014D0347.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32014D0347.json  
      inflating: EURLEX57K/dataset/dev/32003D0187.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32003D0187.json  
      inflating: EURLEX57K/dataset/dev/31994D0738.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31994D0738.json  
      inflating: EURLEX57K/dataset/dev/31991D0664.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31991D0664.json  
      inflating: EURLEX57K/dataset/dev/32010R0192.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32010R0192.json  
      inflating: EURLEX57K/dataset/dev/32008R0205.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32008R0205.json  
      inflating: EURLEX57K/dataset/dev/32009R0167.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32009R0167.json  
      inflating: EURLEX57K/dataset/dev/32012R1267.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32012R1267.json  
      inflating: EURLEX57K/dataset/dev/32006R1158.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32006R1158.json  
      inflating: EURLEX57K/dataset/dev/31994R1341.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31994R1341.json  
      inflating: EURLEX57K/dataset/dev/31986R0766.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31986R0766.json  
      inflating: EURLEX57K/dataset/dev/31992D0518.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31992D0518.json  
      inflating: EURLEX57K/dataset/dev/31994D0041.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31994D0041.json  
      inflating: EURLEX57K/dataset/dev/32004R0184.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32004R0184.json  
      inflating: EURLEX57K/dataset/dev/31989R1759.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31989R1759.json  
      inflating: EURLEX57K/dataset/dev/32012D0537.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32012D0537.json  
      inflating: EURLEX57K/dataset/dev/31996D0663.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31996D0663.json  
      inflating: EURLEX57K/dataset/dev/32002D0573.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32002D0573.json  
      inflating: EURLEX57K/dataset/dev/32015R0308.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32015R0308.json  
      inflating: EURLEX57K/dataset/dev/31996D0233.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31996D0233.json  
      inflating: EURLEX57K/dataset/dev/32009R1263.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32009R1263.json  
      inflating: EURLEX57K/dataset/dev/32010R0254.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32010R0254.json  
      inflating: EURLEX57K/dataset/dev/31996R1563.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31996R1563.json  
      inflating: EURLEX57K/dataset/dev/31991R1708.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31991R1708.json  
      inflating: EURLEX57K/dataset/dev/32013R1040.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32013R1040.json  
      inflating: EURLEX57K/dataset/dev/32004D0883.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32004D0883.json  
      inflating: EURLEX57K/dataset/dev/32004D0929.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32004D0929.json  
      inflating: EURLEX57K/dataset/dev/31995R1166.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31995R1166.json  
      inflating: EURLEX57K/dataset/dev/32003R1141.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32003R1141.json  
      inflating: EURLEX57K/dataset/dev/31984R1755.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31984R1755.json  
      inflating: EURLEX57K/dataset/dev/32008D0201.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32008D0201.json  
      inflating: EURLEX57K/dataset/dev/31997D0852.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31997D0852.json  
      inflating: EURLEX57K/dataset/dev/32014R1155.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32014R1155.json  
      inflating: EURLEX57K/dataset/dev/31991L0620.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31991L0620.json  
      inflating: EURLEX57K/dataset/dev/31992R1423.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31992R1423.json  
      inflating: EURLEX57K/dataset/dev/31990R1344.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31990R1344.json  
      inflating: EURLEX57K/dataset/dev/32006R1363.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32006R1363.json  
      inflating: EURLEX57K/dataset/dev/31993D0441.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31993D0441.json  
      inflating: EURLEX57K/dataset/dev/32006R0122.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32006R0122.json  
      inflating: EURLEX57K/dataset/dev/32001R0719.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32001R0719.json  
      inflating: EURLEX57K/dataset/dev/31992D0373.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31992D0373.json  
      inflating: EURLEX57K/dataset/dev/32004D0641.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32004D0641.json  
      inflating: EURLEX57K/dataset/dev/31997R2216.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31997R2216.json  
      inflating: EURLEX57K/dataset/dev/32005D0489.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32005D0489.json  
      inflating: EURLEX57K/dataset/dev/31997D0690.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31997D0690.json  
      inflating: EURLEX57K/dataset/dev/32015D0422.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32015D0422.json  
      inflating: EURLEX57K/dataset/dev/32006D0126.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32006D0126.json  
      inflating: EURLEX57K/dataset/dev/32013R0893.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32013R0893.json  
      inflating: EURLEX57K/dataset/dev/31979R2276.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31979R2276.json  
      inflating: EURLEX57K/dataset/dev/31997R1590.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31997R1590.json  
      inflating: EURLEX57K/dataset/dev/32013R0939.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32013R0939.json  
      inflating: EURLEX57K/dataset/dev/31990R1201.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31990R1201.json  
      inflating: EURLEX57K/dataset/dev/32003R0992.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32003R0992.json  
      inflating: EURLEX57K/dataset/dev/32011R0024.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32011R0024.json  
      inflating: EURLEX57K/dataset/dev/31996R0260.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31996R0260.json  
      inflating: EURLEX57K/dataset/dev/31990D0382.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31990D0382.json  
      inflating: EURLEX57K/dataset/dev/31978R1411.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31978R1411.json  
      inflating: EURLEX57K/dataset/dev/31990R1981.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31990R1981.json  
      inflating: EURLEX57K/dataset/dev/31996R1888.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31996R1888.json  
      inflating: EURLEX57K/dataset/dev/32007R0885.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32007R0885.json  
      inflating: EURLEX57K/dataset/dev/31992R2960.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31992R2960.json  
      inflating: EURLEX57K/dataset/dev/31978D0711.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31978D0711.json  
      inflating: EURLEX57K/dataset/dev/32013R1152.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32013R1152.json  
      inflating: EURLEX57K/dataset/dev/32002R1761.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32002R1761.json  
      inflating: EURLEX57K/dataset/dev/31987R3785.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31987R3785.json  
      inflating: EURLEX57K/dataset/dev/31997R1206.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31997R1206.json  
      inflating: EURLEX57K/dataset/dev/31996R1534.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31996R1534.json  
      inflating: EURLEX57K/dataset/dev/31987R1307.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31987R1307.json  
      inflating: EURLEX57K/dataset/dev/32000D0243.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32000D0243.json  
      inflating: EURLEX57K/dataset/dev/32003R1815.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32003R1815.json  
      inflating: EURLEX57K/dataset/dev/32007R0539.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32007R0539.json  
      inflating: EURLEX57K/dataset/dev/32007D0182.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32007D0182.json  
      inflating: EURLEX57K/dataset/dev/31985R0621.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31985R0621.json  
      inflating: EURLEX57K/dataset/dev/31999R0858.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31999R0858.json  
      inflating: EURLEX57K/dataset/dev/31993R1891.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31993R1891.json  
      inflating: EURLEX57K/dataset/dev/31988R1483.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31988R1483.json  
      inflating: EURLEX57K/dataset/dev/31983D0239.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31983D0239.json  
      inflating: EURLEX57K/dataset/dev/32012D0075.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32012D0075.json  
      inflating: EURLEX57K/dataset/dev/31993D0085.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31993D0085.json  
      inflating: EURLEX57K/dataset/dev/31991R0709.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31991R0709.json  
      inflating: EURLEX57K/dataset/dev/32001R1935.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32001R1935.json  
      inflating: EURLEX57K/dataset/dev/32001D0336.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32001D0336.json  
      inflating: EURLEX57K/dataset/dev/31995R2859.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31995R2859.json  
      inflating: EURLEX57K/dataset/dev/32013R1200.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32013R1200.json  
      inflating: EURLEX57K/dataset/dev/31994D3092.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31994D3092.json  
      inflating: EURLEX57K/dataset/dev/32004R1490.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32004R1490.json  
      inflating: EURLEX57K/dataset/dev/31998D0495.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31998D0495.json  
      inflating: EURLEX57K/dataset/dev/31993L0079.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31993L0079.json  
      inflating: EURLEX57K/dataset/dev/31990R3302.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31990R3302.json  
      inflating: EURLEX57K/dataset/dev/31991R3060.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31991R3060.json  
      inflating: EURLEX57K/dataset/dev/32003R1244.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32003R1244.json  
      inflating: EURLEX57K/dataset/dev/32013R0554.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32013R0554.json  
      inflating: EURLEX57K/dataset/dev/32006R0159.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32006R0159.json  
      inflating: EURLEX57K/dataset/dev/31997R2328.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31997R2328.json  
      inflating: EURLEX57K/dataset/dev/32000D0541.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32000D0541.json  
      inflating: EURLEX57K/dataset/dev/31995D0533.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31995D0533.json  
      inflating: EURLEX57K/dataset/dev/32003R2038.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32003R2038.json  
      inflating: EURLEX57K/dataset/dev/31977R1881.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31977R1881.json  
      inflating: EURLEX57K/dataset/dev/31984D0300.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31984D0300.json  
      inflating: EURLEX57K/dataset/dev/32003R2192.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32003R2192.json  
      inflating: EURLEX57K/dataset/dev/32002R1825.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32002R1825.json  
      inflating: EURLEX57K/dataset/dev/32005R1064.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32005R1064.json  
      inflating: EURLEX57K/dataset/dev/31999D0061.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31999D0061.json  
      inflating: EURLEX57K/dataset/dev/32007R1353.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32007R1353.json  
      inflating: EURLEX57K/dataset/dev/32005R1434.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32005R1434.json  
      inflating: EURLEX57K/dataset/dev/32007R0811.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32007R0811.json  
      inflating: EURLEX57K/dataset/dev/32006R0620.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32006R0620.json  
      inflating: EURLEX57K/dataset/dev/31976D0699.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31976D0699.json  
      inflating: EURLEX57K/dataset/dev/32004D0006.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32004D0006.json  
      inflating: EURLEX57K/dataset/dev/32004R0517.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32004R0517.json  
      inflating: EURLEX57K/dataset/dev/32010R0782.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32010R0782.json  
      inflating: EURLEX57K/dataset/dev/31985R1948.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31985R1948.json  
      inflating: EURLEX57K/dataset/dev/32006D0761.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32006D0761.json  
      inflating: EURLEX57K/dataset/dev/32013D0696.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32013D0696.json  
      inflating: EURLEX57K/dataset/dev/32007R0112.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32007R0112.json  
      inflating: EURLEX57K/dataset/dev/32004D0617(01).json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32004D0617(01).json  
      inflating: EURLEX57K/dataset/dev/32009D0909.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32009D0909.json  
      inflating: EURLEX57K/dataset/dev/31994D0597.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31994D0597.json  
      inflating: EURLEX57K/dataset/dev/32004D0143.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32004D0143.json  
      inflating: EURLEX57K/dataset/dev/31997R0529.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31997R0529.json  
      inflating: EURLEX57K/dataset/dev/31987D0093.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31987D0093.json  
      inflating: EURLEX57K/dataset/dev/32014R0103.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32014R0103.json  
      inflating: EURLEX57K/dataset/dev/32007D0546.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32007D0546.json  
      inflating: EURLEX57K/dataset/dev/32003R0269.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32003R0269.json  
      inflating: EURLEX57K/dataset/dev/31982D0065.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31982D0065.json  
      inflating: EURLEX57K/dataset/dev/32013R1129.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32013R1129.json  
      inflating: EURLEX57K/dataset/dev/32002R1034.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32002R1034.json  
      inflating: EURLEX57K/dataset/dev/32010R1202.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32010R1202.json  
      inflating: EURLEX57K/dataset/dev/32010L0054.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32010L0054.json  
      inflating: EURLEX57K/dataset/dev/31989R1958.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31989R1958.json  
      inflating: EURLEX57K/dataset/dev/32011R0321.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32011R0321.json  
      inflating: EURLEX57K/dataset/dev/32009R1074.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32009R1074.json  
      inflating: EURLEX57K/dataset/dev/32012D0665.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32012D0665.json  
      inflating: EURLEX57K/dataset/dev/31995D0071.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31995D0071.json  
      inflating: EURLEX57K/dataset/dev/32004D0397.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32004D0397.json  
      inflating: EURLEX57K/dataset/dev/31993D0528.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31993D0528.json  
      inflating: EURLEX57K/dataset/dev/31996D0474.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31996D0474.json  
      inflating: EURLEX57K/dataset/dev/32003R0117.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32003R0117.json  
      inflating: EURLEX57K/dataset/dev/31996D0161.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31996D0161.json  
      inflating: EURLEX57K/dataset/dev/31995D0134.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31995D0134.json  
      inflating: EURLEX57K/dataset/dev/31985R2149.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31985R2149.json  
      inflating: EURLEX57K/dataset/dev/32004R0393.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32004R0393.json  
      inflating: EURLEX57K/dataset/dev/32004D0682.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32004D0682.json  
      inflating: EURLEX57K/dataset/dev/31994R0317.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31994R0317.json  
      inflating: EURLEX57K/dataset/dev/32004R1078.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32004R1078.json  
      inflating: EURLEX57K/dataset/dev/31988R0197.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31988R0197.json  
      inflating: EURLEX57K/dataset/dev/32009R0370.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32009R0370.json  
      inflating: EURLEX57K/dataset/dev/32012D0059.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32012D0059.json  
      inflating: EURLEX57K/dataset/dev/31987R2507.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31987R2507.json  
      inflating: EURLEX57K/dataset/dev/32004D0401.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32004D0401.json  
      inflating: EURLEX57K/dataset/dev/32014R0441.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32014R0441.json  
      inflating: EURLEX57K/dataset/dev/32007D0004.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32007D0004.json  
      inflating: EURLEX57K/dataset/dev/31981D0572.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31981D0572.json  
      inflating: EURLEX57K/dataset/dev/31992R1726.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31992R1726.json  
      inflating: EURLEX57K/dataset/dev/31988D0255.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31988D0255.json  
      inflating: EURLEX57K/dataset/dev/31988R0744.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31988R0744.json  
      inflating: EURLEX57K/dataset/dev/31988R0314.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31988R0314.json  
      inflating: EURLEX57K/dataset/dev/32006R1573.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32006R1573.json  
      inflating: EURLEX57K/dataset/dev/31998D0641.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31998D0641.json  
      inflating: EURLEX57K/dataset/dev/32007L0017.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32007L0017.json  
      inflating: EURLEX57K/dataset/dev/31998D0211.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31998D0211.json  
      inflating: EURLEX57K/dataset/dev/31993D0601.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31993D0601.json  
      inflating: EURLEX57K/dataset/dev/31996R2271.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31996R2271.json  
      inflating: EURLEX57K/dataset/dev/32004R2068.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32004R2068.json  
      inflating: EURLEX57K/dataset/dev/31999D0870.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31999D0870.json  
      inflating: EURLEX57K/dataset/dev/32003R0394.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32003R0394.json  
      inflating: EURLEX57K/dataset/dev/31981D0437.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31981D0437.json  
      inflating: EURLEX57K/dataset/dev/32010D0381.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32010D0381.json  
      inflating: EURLEX57K/dataset/dev/32000R2606.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32000R2606.json  
      inflating: EURLEX57K/dataset/dev/31990D0091.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31990D0091.json  
      inflating: EURLEX57K/dataset/dev/32011R0767.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32011R0767.json  
      inflating: EURLEX57K/dataset/dev/32010D0114.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32010D0114.json  
      inflating: EURLEX57K/dataset/dev/32013R0450.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32013R0450.json  
      inflating: EURLEX57K/dataset/dev/32008R0842.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32008R0842.json  
      inflating: EURLEX57K/dataset/dev/31993R0185.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31993R0185.json  
      inflating: EURLEX57K/dataset/dev/32012R1123.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32012R1123.json  
      inflating: EURLEX57K/dataset/dev/31998R2052.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31998R2052.json  
      inflating: EURLEX57K/dataset/dev/32000R1745.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32000R1745.json  
      inflating: EURLEX57K/dataset/dev/31993R3603.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31993R3603.json  
      inflating: EURLEX57K/dataset/dev/32002R1472.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32002R1472.json  
      inflating: EURLEX57K/dataset/dev/32009D0698.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32009D0698.json  
      inflating: EURLEX57K/dataset/dev/32012R1089.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32012R1089.json  
      inflating: EURLEX57K/dataset/dev/32008R0511.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32008R0511.json  
      inflating: EURLEX57K/dataset/dev/32006R1709.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32006R1709.json  
      inflating: EURLEX57K/dataset/dev/31995R1388.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31995R1388.json  
      inflating: EURLEX57K/dataset/dev/32009R0366.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32009R0366.json  
      inflating: EURLEX57K/dataset/dev/32004R1594.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32004R1594.json  
      inflating: EURLEX57K/dataset/dev/31991R3021.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31991R3021.json  
      inflating: EURLEX57K/dataset/dev/32000L0006.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32000L0006.json  
      inflating: EURLEX57K/dataset/dev/31992R1049.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31992R1049.json  
      inflating: EURLEX57K/dataset/dev/32011R0622.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32011R0622.json  
      inflating: EURLEX57K/dataset/dev/32000D0500.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32000D0500.json  
      inflating: EURLEX57K/dataset/dev/31985D0473.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31985D0473.json  
      inflating: EURLEX57K/dataset/dev/31991R3922.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31991R3922.json  
      inflating: EURLEX57K/dataset/dev/31990R1057.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31990R1057.json  
      inflating: EURLEX57K/dataset/dev/32010R1078.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32010R1078.json  
      inflating: EURLEX57K/dataset/dev/31998D0712.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31998D0712.json  
      inflating: EURLEX57K/dataset/dev/31988R2295.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31988R2295.json  
      inflating: EURLEX57K/dataset/dev/31969R2571.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31969R2571.json  
      inflating: EURLEX57K/dataset/dev/32005D0325.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32005D0325.json  
      inflating: EURLEX57K/dataset/dev/32004R2091.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32004R2091.json  
      inflating: EURLEX57K/dataset/dev/32014D0516.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32014D0516.json  
      inflating: EURLEX57K/dataset/dev/31992R0571.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31992R0571.json  
      inflating: EURLEX57K/dataset/dev/32002R2032.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32002R2032.json  
      inflating: EURLEX57K/dataset/dev/31997R2040.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31997R2040.json  
      inflating: EURLEX57K/dataset/dev/31986R2673.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31986R2673.json  
      inflating: EURLEX57K/dataset/dev/31989R1221.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31989R1221.json  
      inflating: EURLEX57K/dataset/dev/31992D0430.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31992D0430.json  
      inflating: EURLEX57K/dataset/dev/32015D0224.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32015D0224.json  
      inflating: EURLEX57K/dataset/dev/31989R1734.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31989R1734.json  
      inflating: EURLEX57K/dataset/dev/31993R2291.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31993R2291.json  
      inflating: EURLEX57K/dataset/dev/31992R2059.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31992R2059.json  
      inflating: EURLEX57K/dataset/dev/31970R1594.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31970R1594.json  
      inflating: EURLEX57K/dataset/dev/31981L0577.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31981L0577.json  
      inflating: EURLEX57K/dataset/dev/31993R1147.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31993R1147.json  
      inflating: EURLEX57K/dataset/dev/32011L0009.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32011L0009.json  
      inflating: EURLEX57K/dataset/dev/32009R0624.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32009R0624.json  
      inflating: EURLEX57K/dataset/dev/31998D0179.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31998D0179.json  
      inflating: EURLEX57K/dataset/dev/31990R2913.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31990R2913.json  
      inflating: EURLEX57K/dataset/dev/31998D0529.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31998D0529.json  
      inflating: EURLEX57K/dataset/dev/32001D0320.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32001D0320.json  
      inflating: EURLEX57K/dataset/dev/31986D0164.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31986D0164.json  
      inflating: EURLEX57K/dataset/dev/31994R0213.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31994R0213.json  
      inflating: EURLEX57K/dataset/dev/31992R2327.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31992R2327.json  
      inflating: EURLEX57K/dataset/dev/32009R1170.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32009R1170.json  
      inflating: EURLEX57K/dataset/dev/32014R0683.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32014R0683.json  
      inflating: EURLEX57K/dataset/dev/31996D0120.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31996D0120.json  
      inflating: EURLEX57K/dataset/dev/32010D0456.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32010D0456.json  
      inflating: EURLEX57K/dataset/dev/32001D0265.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32001D0265.json  
      inflating: EURLEX57K/dataset/dev/32002R1833.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32002R1833.json  
      inflating: EURLEX57K/dataset/dev/32002R0721.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32002R0721.json  
      inflating: EURLEX57K/dataset/dev/31994D0217.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31994D0217.json  
      inflating: EURLEX57K/dataset/dev/31990D0183.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31990D0183.json  
      inflating: EURLEX57K/dataset/dev/32011R0675.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32011R0675.json  
      inflating: EURLEX57K/dataset/dev/32014R0729.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32014R0729.json  
      inflating: EURLEX57K/dataset/dev/31990R0168.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31990R0168.json  
      inflating: EURLEX57K/dataset/dev/32001R1165.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32001R1165.json  
      inflating: EURLEX57K/dataset/dev/31994R1117.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31994R1117.json  
      inflating: EURLEX57K/dataset/dev/31985R3719.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31985R3719.json  
      inflating: EURLEX57K/dataset/dev/32002R1130.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32002R1130.json  
      inflating: EURLEX57K/dataset/dev/32003R1602.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32003R1602.json  
      inflating: EURLEX57K/dataset/dev/31999R1727.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31999R1727.json  
      inflating: EURLEX57K/dataset/dev/32003R1878.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32003R1878.json  
      inflating: EURLEX57K/dataset/dev/32003D0781.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32003D0781.json  
      inflating: EURLEX57K/dataset/dev/32014R0050.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32014R0050.json  
      inflating: EURLEX57K/dataset/dev/31988R1544.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31988R1544.json  
      inflating: EURLEX57K/dataset/dev/32005R0663.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32005R0663.json  
      inflating: EURLEX57K/dataset/dev/31998D0745.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31998D0745.json  
      inflating: EURLEX57K/dataset/dev/31994R1394.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31994R1394.json  
      inflating: EURLEX57K/dataset/dev/31984R0892.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31984R0892.json  
      inflating: EURLEX57K/dataset/dev/32001R1309.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32001R1309.json  
      inflating: EURLEX57K/dataset/dev/32005R0376.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32005R0376.json  
      inflating: EURLEX57K/dataset/dev/31993R2696.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31993R2696.json  
      inflating: EURLEX57K/dataset/dev/31997D0184.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31997D0184.json  
      inflating: EURLEX57K/dataset/dev/31996R2230.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31996R2230.json  
      inflating: EURLEX57K/dataset/dev/32003R2242.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32003R2242.json  
      inflating: EURLEX57K/dataset/dev/32000D0691.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32000D0691.json  
      inflating: EURLEX57K/dataset/dev/32004D0155.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32004D0155.json  
      inflating: EURLEX57K/dataset/dev/31991D0433.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31991D0433.json  
      inflating: EURLEX57K/dataset/dev/32002D0522(03).json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32002D0522(03).json  
      inflating: EURLEX57K/dataset/dev/32002D0914(01).json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32002D0914(01).json  
      inflating: EURLEX57K/dataset/dev/31997R2016.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31997R2016.json  
      inflating: EURLEX57K/dataset/dev/32006D0776.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32006D0776.json  
      inflating: EURLEX57K/dataset/dev/31983R0314.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31983R0314.json  
      inflating: EURLEX57K/dataset/dev/31992R0527.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31992R0527.json  
      inflating: EURLEX57K/dataset/dev/31986R2275.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31986R2275.json  
      inflating: EURLEX57K/dataset/dev/31997R2446.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31997R2446.json  
      inflating: EURLEX57K/dataset/dev/32014D0540.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32014D0540.json  
      inflating: EURLEX57K/dataset/dev/32002R0409.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32002R0409.json  
      inflating: EURLEX57K/dataset/dev/32005R1423.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32005R1423.json  
      inflating: EURLEX57K/dataset/dev/31988D0215.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31988D0215.json  
      inflating: EURLEX57K/dataset/dev/32014R1210.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32014R1210.json  
      inflating: EURLEX57K/dataset/dev/32006R1476.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32006R1476.json  
      inflating: EURLEX57K/dataset/dev/32014D0956.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32014D0956.json  
      inflating: EURLEX57K/dataset/dev/32006R1533.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32006R1533.json  
      inflating: EURLEX57K/dataset/dev/32007D0802.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32007D0802.json  
      inflating: EURLEX57K/dataset/dev/31994R1680.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31994R1680.json  
      inflating: EURLEX57K/dataset/dev/32013R1094.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32013R1094.json  
      inflating: EURLEX57K/dataset/dev/32004D0857.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32004D0857.json  
      inflating: EURLEX57K/dataset/dev/32006R0322.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32006R0322.json  
      inflating: EURLEX57K/dataset/dev/32011R0418.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32011R0418.json  
      inflating: EURLEX57K/dataset/dev/32010R0280.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32010R0280.json  
      inflating: EURLEX57K/dataset/dev/31999R1233.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31999R1233.json  
      inflating: EURLEX57K/dataset/dev/32001D0008.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32001D0008.json  
      inflating: EURLEX57K/dataset/dev/31986R1635.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31986R1635.json  
      inflating: EURLEX57K/dataset/dev/31998D0482.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31998D0482.json  
      inflating: EURLEX57K/dataset/dev/31994R1403.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31994R1403.json  
      inflating: EURLEX57K/dataset/dev/32007R1082.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32007R1082.json  
      inflating: EURLEX57K/dataset/dev/32013L0041.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32013L0041.json  
      inflating: EURLEX57K/dataset/dev/32009R0275.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32009R0275.json  
      inflating: EURLEX57K/dataset/dev/32008D0955.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32008D0955.json  
      inflating: EURLEX57K/dataset/dev/31996D0064.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31996D0064.json  
      inflating: EURLEX57K/dataset/dev/32003R0507.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32003R0507.json  
      inflating: EURLEX57K/dataset/dev/32012R0334.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32012R0334.json  
      inflating: EURLEX57K/dataset/dev/31986D0165.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31986D0165.json  
      inflating: EURLEX57K/dataset/dev/32001R0260.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32001R0260.json  
      inflating: EURLEX57K/dataset/dev/31995D0461.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31995D0461.json  
      inflating: EURLEX57K/dataset/dev/32004D0292.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32004D0292.json  
      inflating: EURLEX57K/dataset/dev/31990R3315.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31990R3315.json  
      inflating: EURLEX57K/dataset/dev/32003R0911.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32003R0911.json  
      inflating: EURLEX57K/dataset/dev/32014R0905.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32014R0905.json  
      inflating: EURLEX57K/dataset/dev/31998R1940.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31998R1940.json  
      inflating: EURLEX57K/dataset/dev/31998D0343.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31998D0343.json  
      inflating: EURLEX57K/dataset/dev/31984R1693.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31984R1693.json  
      inflating: EURLEX57K/dataset/dev/32003R1087.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32003R1087.json  
      inflating: EURLEX57K/dataset/dev/31994R1638.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31994R1638.json  
      inflating: EURLEX57K/dataset/dev/32007R0851.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32007R0851.json  
      inflating: EURLEX57K/dataset/dev/31983R1102.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31983R1102.json  
      inflating: EURLEX57K/dataset/dev/31989R0431.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31989R0431.json  
      inflating: EURLEX57K/dataset/dev/32005R1927.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32005R1927.json  
      inflating: EURLEX57K/dataset/dev/31995R1809.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31995R1809.json  
      inflating: EURLEX57K/dataset/dev/32010D0683.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32010D0683.json  
      inflating: EURLEX57K/dataset/dev/32004R0107.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32004R0107.json  
      inflating: EURLEX57K/dataset/dev/31991D0171.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31991D0171.json  
      inflating: EURLEX57K/dataset/dev/32001R2089.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32001R2089.json  
      inflating: EURLEX57K/dataset/dev/32015R0221.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32015R0221.json  
      inflating: EURLEX57K/dataset/dev/32003R0383.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32003R0383.json  
      inflating: EURLEX57K/dataset/dev/31994D0187.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31994D0187.json  
      inflating: EURLEX57K/dataset/dev/32003R0229.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32003R0229.json  
      inflating: EURLEX57K/dataset/dev/32012R0949.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32012R0949.json  
      inflating: EURLEX57K/dataset/dev/31998D0206.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31998D0206.json  
      inflating: EURLEX57K/dataset/dev/31992R1224.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31992R1224.json  
      inflating: EURLEX57K/dataset/dev/31987R3614.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31987R3614.json  
      inflating: EURLEX57K/dataset/dev/31992R1674.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31992R1674.json  
      inflating: EURLEX57K/dataset/dev/31999D0534.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31999D0534.json  
      inflating: EURLEX57K/dataset/dev/31988R3939.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31988R3939.json  
      inflating: EURLEX57K/dataset/dev/31999R2018.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31999R2018.json  
      inflating: EURLEX57K/dataset/dev/32001D0726.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32001D0726.json  
      inflating: EURLEX57K/dataset/dev/31993D0495.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31993D0495.json  
      inflating: EURLEX57K/dataset/dev/32004R1829.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32004R1829.json  
      inflating: EURLEX57K/dataset/dev/31996D0033.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31996D0033.json  
      inflating: EURLEX57K/dataset/dev/32012D0672.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32012D0672.json  
      inflating: EURLEX57K/dataset/dev/32012R0363.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32012R0363.json  
      inflating: EURLEX57K/dataset/dev/32003D0041.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32003D0041.json  
      inflating: EURLEX57K/dataset/dev/32002D0323.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32002D0323.json  
      inflating: EURLEX57K/dataset/dev/32008R1301.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32008R1301.json  
      inflating: EURLEX57K/dataset/dev/32005D0148.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32005D0148.json  
      inflating: EURLEX57K/dataset/dev/32003R0803.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32003R0803.json  
      inflating: EURLEX57K/dataset/dev/32003R1711.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32003R1711.json  
      inflating: EURLEX57K/dataset/dev/31990R3657.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31990R3657.json  
      inflating: EURLEX57K/dataset/dev/32013R1240.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32013R1240.json  
      inflating: EURLEX57K/dataset/dev/32008D0451.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32008D0451.json  
      inflating: EURLEX57K/dataset/dev/31993R1795.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31993R1795.json  
      inflating: EURLEX57K/dataset/dev/32001R1099.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32001R1099.json  
      inflating: EURLEX57K/dataset/dev/32009R0367.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32009R0367.json  
      inflating: EURLEX57K/dataset/dev/32013R0514.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32013R0514.json  
      inflating: EURLEX57K/dataset/dev/31982D0249.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31982D0249.json  
      inflating: EURLEX57K/dataset/dev/31998R2815.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31998R2815.json  
      inflating: EURLEX57K/dataset/dev/31984R1943.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31984R1943.json  
      inflating: EURLEX57K/dataset/dev/32002R1865.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32002R1865.json  
      inflating: EURLEX57K/dataset/dev/32011D0298.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32011D0298.json  
      inflating: EURLEX57K/dataset/dev/32001R0722.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32001R0722.json  
      inflating: EURLEX57K/dataset/dev/31998R0982.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31998R0982.json  
      inflating: EURLEX57K/dataset/dev/32011D0762.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32011D0762.json  
      inflating: EURLEX57K/dataset/dev/31987R0254.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31987R0254.json  
      inflating: EURLEX57K/dataset/dev/31980L1269.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31980L1269.json  
      inflating: EURLEX57K/dataset/dev/32003R2347.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32003R2347.json  
      inflating: EURLEX57K/dataset/dev/31995R2360.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31995R2360.json  
      inflating: EURLEX57K/dataset/dev/32003R0680.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32003R0680.json  
      inflating: EURLEX57K/dataset/dev/32007R0514.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32007R0514.json  
      inflating: EURLEX57K/dataset/dev/32013R0781.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32013R0781.json  
      inflating: EURLEX57K/dataset/dev/31997R2407.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31997R2407.json  
      inflating: EURLEX57K/dataset/dev/31989D0136.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31989D0136.json  
      inflating: EURLEX57K/dataset/dev/32006R1437.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32006R1437.json  
      inflating: EURLEX57K/dataset/dev/31989D0566.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31989D0566.json  
      inflating: EURLEX57K/dataset/dev/31986R1418.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31986R1418.json  
      inflating: EURLEX57K/dataset/dev/32014D0917.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32014D0917.json  
      inflating: EURLEX57K/dataset/dev/32013D0046(01).json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32013D0046(01).json  
      inflating: EURLEX57K/dataset/dev/31978R2210.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31978R2210.json  
      inflating: EURLEX57K/dataset/dev/32002R2160.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32002R2160.json  
      inflating: EURLEX57K/dataset/dev/31989R1373.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31989R1373.json  
      inflating: EURLEX57K/dataset/dev/32013D0785.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32013D0785.json  
      inflating: EURLEX57K/dataset/dev/32007D0510.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32007D0510.json  
      inflating: EURLEX57K/dataset/dev/31999L0024.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31999L0024.json  
      inflating: EURLEX57K/dataset/dev/32005L0008.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32005L0008.json  
      inflating: EURLEX57K/dataset/dev/32008R0506.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32008R0506.json  
      inflating: EURLEX57K/dataset/dev/31998D0093.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31998D0093.json  
      inflating: EURLEX57K/dataset/dev/32006R0949.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32006R0949.json  
      inflating: EURLEX57K/dataset/dev/32009D0725.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32009D0725.json  
      inflating: EURLEX57K/dataset/dev/32000R1752.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32000R1752.json  
      inflating: EURLEX57K/dataset/dev/31989R1959.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31989R1959.json  
      inflating: EURLEX57K/dataset/dev/31978R2990.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31978R2990.json  
      inflating: EURLEX57K/dataset/dev/32003R2081.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32003R2081.json  
      inflating: EURLEX57K/dataset/dev/31994R0603.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31994R0603.json  
      inflating: EURLEX57K/dataset/dev/32003R0546.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32003R0546.json  
      inflating: EURLEX57K/dataset/dev/31992R3825.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31992R3825.json  
      inflating: EURLEX57K/dataset/dev/32002R2249.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32002R2249.json  
      inflating: EURLEX57K/dataset/dev/32012D0234.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32012D0234.json  
      inflating: EURLEX57K/dataset/dev/32011R0770.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32011R0770.json  
      inflating: EURLEX57K/dataset/dev/32011D0261.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32011D0261.json  
      inflating: EURLEX57K/dataset/dev/32001D0225.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32001D0225.json  
      inflating: EURLEX57K/dataset/dev/31984D0356.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31984D0356.json  
      inflating: EURLEX57K/dataset/dev/31987D0303.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31987D0303.json  
      inflating: EURLEX57K/dataset/dev/31986D0061.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31986D0061.json  
      inflating: EURLEX57K/dataset/dev/32005R2167.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32005R2167.json  
      inflating: EURLEX57K/dataset/dev/31994R0316.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31994R0316.json  
      inflating: EURLEX57K/dataset/dev/32007R0797.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32007R0797.json  
      inflating: EURLEX57K/dataset/dev/31981R3567.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31981R3567.json  
      inflating: EURLEX57K/dataset/dev/32003R0950.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32003R0950.json  
      inflating: EURLEX57K/dataset/dev/31990R3704.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31990R3704.json  
      inflating: EURLEX57K/dataset/dev/32002R1520.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32002R1520.json  
      inflating: EURLEX57K/dataset/dev/31981D0877.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31981D0877.json  
      inflating: EURLEX57K/dataset/dev/31974L0553.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31974L0553.json  
      inflating: EURLEX57K/dataset/dev/31992R0862.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31992R0862.json  
      inflating: EURLEX57K/dataset/dev/32008R0697.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32008R0697.json  
      inflating: EURLEX57K/dataset/dev/32007R0810.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32007R0810.json  
      inflating: EURLEX57K/dataset/dev/31976R1432.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31976R1432.json  
      inflating: EURLEX57K/dataset/dev/31993R1941.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31993R1941.json  
      inflating: EURLEX57K/dataset/dev/31991D0075.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31991D0075.json  
      inflating: EURLEX57K/dataset/dev/31981D0174.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31981D0174.json  
      inflating: EURLEX57K/dataset/dev/32000D1223(01).json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32000D1223(01).json  
      inflating: EURLEX57K/dataset/dev/32007D0117.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32007D0117.json  
      inflating: EURLEX57K/dataset/dev/32009D0908.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32009D0908.json  
      inflating: EURLEX57K/dataset/dev/31997R0528.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31997R0528.json  
      inflating: EURLEX57K/dataset/dev/32005R0731.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32005R0731.json  
      inflating: EURLEX57K/dataset/dev/32003R0268.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32003R0268.json  
      inflating: EURLEX57K/dataset/dev/32003R1880.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32003R1880.json  
      inflating: EURLEX57K/dataset/dev/32007L0041.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32007L0041.json  
      inflating: EURLEX57K/dataset/dev/32005R1120.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32005R1120.json  
      inflating: EURLEX57K/dataset/dev/31987R1238.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31987R1238.json  
      inflating: EURLEX57K/dataset/dev/32003R1479.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32003R1479.json  
      inflating: EURLEX57K/dataset/dev/31993D0084.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31993D0084.json  
      inflating: EURLEX57K/dataset/dev/32000R0114.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32000R0114.json  
      inflating: EURLEX57K/dataset/dev/31997D0740.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31997D0740.json  
      inflating: EURLEX57K/dataset/dev/31991D0219.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31991D0219.json  
      inflating: EURLEX57K/dataset/dev/32011D0666.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32011D0666.json  
      inflating: EURLEX57K/dataset/dev/31992R2760.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31992R2760.json  
      inflating: EURLEX57K/dataset/dev/32011R0377.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32011R0377.json  
      inflating: EURLEX57K/dataset/dev/31986D0523.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31986D0523.json  
      inflating: EURLEX57K/dataset/dev/32002R1961.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32002R1961.json  
      inflating: EURLEX57K/dataset/dev/32012D0613(01).json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32012D0613(01).json  
      inflating: EURLEX57K/dataset/dev/32009D0772.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32009D0772.json  
      inflating: EURLEX57K/dataset/dev/31993R1384.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31993R1384.json  
      inflating: EURLEX57K/dataset/dev/31988D0595.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31988D0595.json  
      inflating: EURLEX57K/dataset/dev/32013R0943.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32013R0943.json  
      inflating: EURLEX57K/dataset/dev/32001R1172.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32001R1172.json  
      inflating: EURLEX57K/dataset/dev/31984R3196.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31984R3196.json  
      inflating: EURLEX57K/dataset/dev/31976D0948.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31976D0948.json  
      inflating: EURLEX57K/dataset/dev/32009R0776.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32009R0776.json  
      inflating: EURLEX57K/dataset/dev/31984R1451.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31984R1451.json  
      inflating: EURLEX57K/dataset/dev/32006D0449.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32006D0449.json  
      inflating: EURLEX57K/dataset/dev/32010R0150.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32010R0150.json  
      inflating: EURLEX57K/dataset/dev/31994R1803.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31994R1803.json  
      inflating: EURLEX57K/dataset/dev/32007D0681.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32007D0681.json  
      inflating: EURLEX57K/dataset/dev/32001D0788.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32001D0788.json  
      inflating: EURLEX57K/dataset/dev/32003R1951.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32003R1951.json  
      inflating: EURLEX57K/dataset/dev/32014D0468.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32014D0468.json  
      inflating: EURLEX57K/dataset/dev/32001R2549.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32001R2549.json  
      inflating: EURLEX57K/dataset/dev/32001R0574.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32001R0574.json  
      inflating: EURLEX57K/dataset/dev/32009R0832.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32009R0832.json  
      inflating: EURLEX57K/dataset/dev/32002D0460.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32002D0460.json  
      inflating: EURLEX57K/dataset/dev/31985D0274.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31985D0274.json  
      inflating: EURLEX57K/dataset/dev/31988R0786.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31988R0786.json  
      inflating: EURLEX57K/dataset/dev/32008R0603.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32008R0603.json  
      inflating: EURLEX57K/dataset/dev/32003R1402.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32003R1402.json  
      inflating: EURLEX57K/dataset/dev/32004R1393.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32004R1393.json  
      inflating: EURLEX57K/dataset/dev/32002R1760.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32002R1760.json  
      inflating: EURLEX57K/dataset/dev/31998R0668.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31998R0668.json  
      inflating: EURLEX57K/dataset/dev/32014R0895.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32014R0895.json  
      inflating: EURLEX57K/dataset/dev/32000R1512.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32000R1512.json  
      inflating: EURLEX57K/dataset/dev/32001D0120.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32001D0120.json  
      inflating: EURLEX57K/dataset/dev/32006R1948.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32006R1948.json  
      inflating: EURLEX57K/dataset/dev/32012D0424.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32012D0424.json  
      inflating: EURLEX57K/dataset/dev/32002D0525.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32002D0525.json  
      inflating: EURLEX57K/dataset/dev/32012D0074.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32012D0074.json  
      inflating: EURLEX57K/dataset/dev/31980R2741.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31980R2741.json  
      inflating: EURLEX57K/dataset/dev/32014R0196.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32014R0196.json  
      inflating: EURLEX57K/dataset/dev/31989D0233.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31989D0233.json  
      inflating: EURLEX57K/dataset/dev/31995R1649.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31995R1649.json  
      inflating: EURLEX57K/dataset/dev/31984L0386.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31984L0386.json  
      inflating: EURLEX57K/dataset/dev/31998D0050.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31998D0050.json  
      inflating: EURLEX57K/dataset/dev/32008R0195.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32008R0195.json  
      inflating: EURLEX57K/dataset/dev/32011D0308.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32011D0308.json  
      inflating: EURLEX57K/dataset/dev/31991R0266.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31991R0266.json  
      inflating: EURLEX57K/dataset/dev/31994R1878.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31994R1878.json  
      inflating: EURLEX57K/dataset/dev/32014R0315.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32014R0315.json  
      inflating: EURLEX57K/dataset/dev/32004R1956.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32004R1956.json  
      inflating: EURLEX57K/dataset/dev/32013D0195.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32013D0195.json  
      inflating: EURLEX57K/dataset/dev/31987R1869.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31987R1869.json  
      inflating: EURLEX57K/dataset/dev/32007R0304.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32007R0304.json  
      inflating: EURLEX57K/dataset/dev/31990D0550.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31990D0550.json  
      inflating: EURLEX57K/dataset/dev/32004R1813.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32004R1813.json  
      inflating: EURLEX57K/dataset/dev/31997D0691.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31997D0691.json  
      inflating: EURLEX57K/dataset/dev/31996D0009.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31996D0009.json  
      inflating: EURLEX57K/dataset/dev/32008D0938.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32008D0938.json  
      inflating: EURLEX57K/dataset/dev/31993D0155.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31993D0155.json  
      inflating: EURLEX57K/dataset/dev/32014D0741.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32014D0741.json  
      inflating: EURLEX57K/dataset/dev/32005R1622.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32005R1622.json  
      inflating: EURLEX57K/dataset/dev/31996R1309.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31996R1309.json  
      inflating: EURLEX57K/dataset/dev/32005D0871.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32005D0871.json  
      inflating: EURLEX57K/dataset/dev/32013R1104.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32013R1104.json  
      inflating: EURLEX57K/dataset/dev/32006R1509.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32006R1509.json  
      inflating: EURLEX57K/dataset/dev/31994R0802.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31994R0802.json  
      inflating: EURLEX57K/dataset/dev/32001R1332.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32001R1332.json  
      inflating: EURLEX57K/dataset/dev/32009R0536.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32009R0536.json  
      inflating: EURLEX57K/dataset/dev/32010D0251.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32010D0251.json  
      inflating: EURLEX57K/dataset/dev/32010R0740.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32010R0740.json  
      inflating: EURLEX57K/dataset/dev/32007R0580.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32007R0580.json  
      inflating: EURLEX57K/dataset/dev/32012R0477.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32012R0477.json  
      inflating: EURLEX57K/dataset/dev/31992D0519.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31992D0519.json  
      inflating: EURLEX57K/dataset/dev/31988R4268.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31988R4268.json  
      inflating: EURLEX57K/dataset/dev/32004R2012.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32004R2012.json  
      inflating: EURLEX57K/dataset/dev/32011R0588.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32011R0588.json  
      inflating: EURLEX57K/dataset/dev/32012R0027.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32012R0027.json  
      inflating: EURLEX57K/dataset/dev/32000R0354.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32000R0354.json  
      inflating: EURLEX57K/dataset/dev/32003R0751.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32003R0751.json  
      inflating: EURLEX57K/dataset/dev/31996D0232.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31996D0232.json  
      inflating: EURLEX57K/dataset/dev/31989R3620.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31989R3620.json  
      inflating: EURLEX57K/dataset/dev/32014D0080.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32014D0080.json  
      inflating: EURLEX57K/dataset/dev/31990R1938.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31990R1938.json  
      inflating: EURLEX57K/dataset/dev/31995R1537.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31995R1537.json  
      inflating: EURLEX57K/dataset/dev/31996R1562.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31996R1562.json  
      inflating: EURLEX57K/dataset/dev/32013R1041.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32013R1041.json  
      inflating: EURLEX57K/dataset/dev/32009R0473.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32009R0473.json  
      inflating: EURLEX57K/dataset/dev/31988D0385.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31988D0385.json  
      inflating: EURLEX57K/dataset/dev/31994R1205.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31994R1205.json  
      inflating: EURLEX57K/dataset/dev/32004R1901.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32004R1901.json  
      inflating: EURLEX57K/dataset/dev/31993R2091.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31993R2091.json  
      inflating: EURLEX57K/dataset/dev/32001R2288.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32001R2288.json  
      inflating: EURLEX57K/dataset/dev/31998R1307.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31998R1307.json  
      inflating: EURLEX57K/dataset/dev/31984R1884.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31984R1884.json  
      inflating: EURLEX57K/dataset/dev/32014D0653.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32014D0653.json  
      inflating: EURLEX57K/dataset/dev/31988R0944.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31988R0944.json  
      inflating: EURLEX57K/dataset/dev/32000R0587.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32000R0587.json  
      inflating: EURLEX57K/dataset/dev/31987R0393.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31987R0393.json  
      inflating: EURLEX57K/dataset/dev/31988R2580.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31988R2580.json  
      inflating: EURLEX57K/dataset/dev/31978D0481.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31978D0481.json  
      inflating: EURLEX57K/dataset/dev/32013R0980.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32013R0980.json  
      inflating: EURLEX57K/dataset/dev/32006R1765.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32006R1765.json  
      inflating: EURLEX57K/dataset/dev/32008R0068.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32008R0068.json  
      inflating: EURLEX57K/dataset/dev/32005R0822.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32005R0822.json  
      inflating: EURLEX57K/dataset/dev/31993R2838.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31993R2838.json  
      inflating: EURLEX57K/dataset/dev/31991R3108.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31991R3108.json  
      inflating: EURLEX57K/dataset/dev/32005R1675.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32005R1675.json  
      inflating: EURLEX57K/dataset/dev/32010R0890.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32010R0890.json  
      inflating: EURLEX57K/dataset/dev/31994R0678.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31994R0678.json  
      inflating: EURLEX57K/dataset/dev/32004D0247.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32004D0247.json  
      inflating: EURLEX57K/dataset/dev/32004R1844.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32004R1844.json  
      inflating: EURLEX57K/dataset/dev/31984D0638.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31984D0638.json  
      inflating: EURLEX57K/dataset/dev/32006R0031.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32006R0031.json  
      inflating: EURLEX57K/dataset/dev/31982D0361.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31982D0361.json  
      inflating: EURLEX57K/dataset/dev/32005R2059.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32005R2059.json  
      inflating: EURLEX57K/dataset/dev/32000D0429.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32000D0429.json  
      inflating: EURLEX57K/dataset/dev/31986R2189.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31986R2189.json  
      inflating: EURLEX57K/dataset/dev/31995D0334.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31995D0334.json  
      inflating: EURLEX57K/dataset/dev/32009D0932.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32009D0932.json  
      inflating: EURLEX57K/dataset/dev/32013R0353.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32013R0353.json  
      inflating: EURLEX57K/dataset/dev/32010R0306.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32010R0306.json  
      inflating: EURLEX57K/dataset/dev/31993R0686.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31993R0686.json  
      inflating: EURLEX57K/dataset/dev/32009R0873.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32009R0873.json  
      inflating: EURLEX57K/dataset/dev/32003R0252.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32003R0252.json  
      inflating: EURLEX57K/dataset/dev/32001R0866.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32001R0866.json  
      inflating: EURLEX57K/dataset/dev/32001R1774.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32001R1774.json  
      inflating: EURLEX57K/dataset/dev/31994D0810.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31994D0810.json  
      inflating: EURLEX57K/dataset/dev/31986R1475.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31986R1475.json  
      inflating: EURLEX57K/dataset/dev/32001R1261.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32001R1261.json  
      inflating: EURLEX57K/dataset/dev/31997R1246.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31997R1246.json  
      inflating: EURLEX57K/dataset/dev/31997L0010.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31997L0010.json  
      inflating: EURLEX57K/dataset/dev/31987R1717.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31987R1717.json  
      inflating: EURLEX57K/dataset/dev/31995R1171.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31995R1171.json  
      inflating: EURLEX57K/dataset/dev/32007R1368.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32007R1368.json  
      inflating: EURLEX57K/dataset/dev/32008D0646.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32008D0646.json  
      inflating: EURLEX57K/dataset/dev/32001R0923.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32001R0923.json  
      inflating: EURLEX57K/dataset/dev/31987D0417.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31987D0417.json  
      inflating: EURLEX57K/dataset/dev/32012R0174.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32012R0174.json  
      inflating: EURLEX57K/dataset/dev/32003D0606.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32003D0606.json  
      inflating: EURLEX57K/dataset/dev/32014R1115.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32014R1115.json  
      inflating: EURLEX57K/dataset/dev/31969R2622.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31969R2622.json  
      inflating: EURLEX57K/dataset/dev/31986R3761.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31986R3761.json  
      inflating: EURLEX57K/dataset/dev/32004R1014.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32004R1014.json  
      inflating: EURLEX57K/dataset/dev/31990R2178.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31990R2178.json  
      inflating: EURLEX57K/dataset/dev/32006D0189.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32006D0189.json  
      inflating: EURLEX57K/dataset/dev/31996R2071.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31996R2071.json  
      inflating: EURLEX57K/dataset/dev/31994R2716.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31994R2716.json  
      inflating: EURLEX57K/dataset/dev/31984R2247.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31984R2247.json  
      inflating: EURLEX57K/dataset/dev/31995D0158.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31995D0158.json  
      inflating: EURLEX57K/dataset/dev/31991D0366.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31991D0366.json  
      inflating: EURLEX57K/dataset/dev/32004R0605.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32004R0605.json  
      inflating: EURLEX57K/dataset/dev/32004R0310.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32004R0310.json  
      inflating: EURLEX57K/dataset/dev/31996R2134.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31996R2134.json  
      inflating: EURLEX57K/dataset/dev/31993R0055.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31993R0055.json  
      inflating: EURLEX57K/dataset/dev/32001R2271.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32001R2271.json  
      inflating: EURLEX57K/dataset/dev/32003R0481.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32003R0481.json  
      inflating: EURLEX57K/dataset/dev/32006R0027.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32006R0027.json  
      inflating: EURLEX57K/dataset/dev/32008R0838.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32008R0838.json  
      inflating: EURLEX57K/dataset/dev/32013D0091.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32013D0091.json  
      inflating: EURLEX57K/dataset/dev/32004R0740.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32004R0740.json  
      inflating: EURLEX57K/dataset/dev/32005D0133.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32005D0133.json  
      inflating: EURLEX57K/dataset/dev/32006D0166.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32006D0166.json  
      inflating: EURLEX57K/dataset/dev/31998D0504.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31998D0504.json  
      inflating: EURLEX57K/dataset/dev/32010L0038.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32010L0038.json  
      inflating: EURLEX57K/dataset/dev/31993R1644.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31993R1644.json  
      inflating: EURLEX57K/dataset/dev/32008R0091.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32008R0091.json  
      inflating: EURLEX57K/dataset/dev/32004R1151.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32004R1151.json  
      inflating: EURLEX57K/dataset/dev/32005R1233.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32005R1233.json  
      inflating: EURLEX57K/dataset/dev/31987R4057.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31987R4057.json  
      inflating: EURLEX57K/dataset/dev/32012D0824.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32012D0824.json  
      inflating: EURLEX57K/dataset/dev/32002D0925.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32002D0925.json  
      inflating: EURLEX57K/dataset/dev/32001R1373.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32001R1373.json  
      inflating: EURLEX57K/dataset/dev/31979L0532.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31979L0532.json  
      inflating: EURLEX57K/dataset/dev/32001D0423.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32001D0423.json  
      inflating: EURLEX57K/dataset/dev/31981R2120.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31981R2120.json  
      inflating: EURLEX57K/dataset/dev/32002D0026.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32002D0026.json  
      inflating: EURLEX57K/dataset/dev/32011D0522.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32011D0522.json  
      inflating: EURLEX57K/dataset/dev/31995R1960.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31995R1960.json  
      inflating: EURLEX57K/dataset/dev/31993D0285.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31993D0285.json  
      inflating: EURLEX57K/dataset/dev/32003R0340.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32003R0340.json  
      inflating: EURLEX57K/dataset/dev/32010R0644.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32010R0644.json  
      inflating: EURLEX57K/dataset/dev/31992R2561.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31992R2561.json  
      inflating: EURLEX57K/dataset/dev/31991D0018.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31991D0018.json  
      inflating: EURLEX57K/dataset/dev/31996R0298.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31996R0298.json  
      inflating: EURLEX57K/dataset/dev/32009R0062.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32009R0062.json  
      inflating: EURLEX57K/dataset/dev/31994R1614.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31994R1614.json  
      inflating: EURLEX57K/dataset/dev/32001R1666.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32001R1666.json  
      inflating: EURLEX57K/dataset/dev/32011R0875.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32011R0875.json  
      inflating: EURLEX57K/dataset/dev/32004R1690.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32004R1690.json  
      inflating: EURLEX57K/dataset/dev/32013R0492.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32013R0492.json  
      inflating: EURLEX57K/dataset/dev/32015D0570.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32015D0570.json  
      inflating: EURLEX57K/dataset/dev/31999R1024.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31999R1024.json  
      inflating: EURLEX57K/dataset/dev/31983D0107.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31983D0107.json  
      inflating: EURLEX57K/dataset/dev/31987D0339.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31987D0339.json  
      inflating: EURLEX57K/dataset/dev/31980D1313.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31980D1313.json  
      inflating: EURLEX57K/dataset/dev/31992R3059.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31992R3059.json  
      inflating: EURLEX57K/dataset/dev/32006R1374.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32006R1374.json  
      inflating: EURLEX57K/dataset/dev/31976R1776.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31976R1776.json  
      inflating: EURLEX57K/dataset/dev/31995R2970.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31995R2970.json  
      inflating: EURLEX57K/dataset/dev/31998D1114(02).json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31998D1114(02).json  
      inflating: EURLEX57K/dataset/dev/31993R1756.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31993R1756.json  
      inflating: EURLEX57K/dataset/dev/31998D0103.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31998D0103.json  
      inflating: EURLEX57K/dataset/dev/32006R1661.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32006R1661.json  
      inflating: EURLEX57K/dataset/dev/32007R1503.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32007R1503.json  
      inflating: EURLEX57K/dataset/dev/32013R0884.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32013R0884.json  
      inflating: EURLEX57K/dataset/dev/32014R0246.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32014R0246.json  
      inflating: EURLEX57K/dataset/dev/31991D0274.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31991D0274.json  
      inflating: EURLEX57K/dataset/dev/31971R0619.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31971R0619.json  
      inflating: EURLEX57K/dataset/dev/31995R2136.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31995R2136.json  
      inflating: EURLEX57K/dataset/dev/31984D0229.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31984D0229.json  
      inflating: EURLEX57K/dataset/dev/32007R0312.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32007R0312.json  
      inflating: EURLEX57K/dataset/dev/32007D0674.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32007D0674.json  
      inflating: EURLEX57K/dataset/dev/32014D0370.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32014D0370.json  
      inflating: EURLEX57K/dataset/dev/32005R0052.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32005R0052.json  
      inflating: EURLEX57K/dataset/dev/32012D0783.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32012D0783.json  
      inflating: EURLEX57K/dataset/dev/32015D0442.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32015D0442.json  
      inflating: EURLEX57K/dataset/dev/32004R1872.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32004R1872.json  
      inflating: EURLEX57K/dataset/dev/31990R0470.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31990R0470.json  
      inflating: EURLEX57K/dataset/dev/32012R1179.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32012R1179.json  
      inflating: EURLEX57K/dataset/dev/32001R1187.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32001R1187.json  
      inflating: EURLEX57K/dataset/dev/31993R1664.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31993R1664.json  
      inflating: EURLEX57K/dataset/dev/32009R0783.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32009R0783.json  
      inflating: EURLEX57K/dataset/dev/32009R0629.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32009R0629.json  
      inflating: EURLEX57K/dataset/dev/32013R0959.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32013R0959.json  
      inflating: EURLEX57K/dataset/dev/31993R0833.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31993R0833.json  
      inflating: EURLEX57K/dataset/dev/31986R3311.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31986R3311.json  
      inflating: EURLEX57K/dataset/dev/31993R1721.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31993R1721.json  
      inflating: EURLEX57K/dataset/dev/31989D0602.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31989D0602.json  
      inflating: EURLEX57K/dataset/dev/32011D0369.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32011D0369.json  
      inflating: EURLEX57K/dataset/dev/31995D0482.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31995D0482.json  
      inflating: EURLEX57K/dataset/dev/32004R1937.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32004R1937.json  
      inflating: EURLEX57K/dataset/dev/32002R1214.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32002R1214.json  
      inflating: EURLEX57K/dataset/dev/32010R1188.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32010R1188.json  
      inflating: EURLEX57K/dataset/dev/32011R0802.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32011R0802.json  
      inflating: EURLEX57K/dataset/dev/31980R3561.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31980R3561.json  
      inflating: EURLEX57K/dataset/dev/32002R0956.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32002R0956.json  
      inflating: EURLEX57K/dataset/dev/31971R0951.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31971R0951.json  
      inflating: EURLEX57K/dataset/dev/31991R3702.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31991R3702.json  
      inflating: EURLEX57K/dataset/dev/31985R0211.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31985R0211.json  
      inflating: EURLEX57K/dataset/dev/32008R1136.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32008R1136.json  
      inflating: EURLEX57K/dataset/dev/32013D0727.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32013D0727.json  
      inflating: EURLEX57K/dataset/dev/32005R0394.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32005R0394.json  
      inflating: EURLEX57K/dataset/dev/31995D0251.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31995D0251.json  
      inflating: EURLEX57K/dataset/dev/32009R1254.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32009R1254.json  
      inflating: EURLEX57K/dataset/dev/32003R1875.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32003R1875.json  
      inflating: EURLEX57K/dataset/dev/31981R2442.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31981R2442.json  
      inflating: EURLEX57K/dataset/dev/32004D0158.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32004D0158.json  
      inflating: EURLEX57K/dataset/dev/31995D0314.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31995D0314.json  
      inflating: EURLEX57K/dataset/dev/32003D0333.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32003D0333.json  
      inflating: EURLEX57K/dataset/dev/32005R1839.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32005R1839.json  
      inflating: EURLEX57K/dataset/dev/31995R2268.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31995R2268.json  
      inflating: EURLEX57K/dataset/dev/32003R0272.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32003R0272.json  
      inflating: EURLEX57K/dataset/dev/32013R0723.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32013R0723.json  
      inflating: EURLEX57K/dataset/dev/32014D0059.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32014D0059.json  
      inflating: EURLEX57K/dataset/dev/31997R1289.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31997R1289.json  
      inflating: EURLEX57K/dataset/dev/32001R0846.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32001R0846.json  
      inflating: EURLEX57K/dataset/dev/32001R1754.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32001R1754.json  
      inflating: EURLEX57K/dataset/dev/32010R1219.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32010R1219.json  
      inflating: EURLEX57K/dataset/dev/32002R1185.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32002R1185.json  
      inflating: EURLEX57K/dataset/dev/32000R2867.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32000R2867.json  
      inflating: EURLEX57K/dataset/dev/32014R0266.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32014R0266.json  
      inflating: EURLEX57K/dataset/dev/32014D0777.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32014D0777.json  
      inflating: EURLEX57K/dataset/dev/31977R1822.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31977R1822.json  
      inflating: EURLEX57K/dataset/dev/31995D0590.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31995D0590.json  
      inflating: EURLEX57K/dataset/dev/32004D0323(01).json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32004D0323(01).json  
      inflating: EURLEX57K/dataset/dev/32014D0327.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32014D0327.json  
      inflating: EURLEX57K/dataset/dev/31981D0355.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31981D0355.json  
      inflating: EURLEX57K/dataset/dev/31983D0062.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31983D0062.json  
      inflating: EURLEX57K/dataset/dev/32004R0367.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32004R0367.json  
      inflating: EURLEX57K/dataset/dev/32003R0419.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32003R0419.json  
      inflating: EURLEX57K/dataset/dev/31999R1004.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31999R1004.json  
      inflating: EURLEX57K/dataset/dev/32013D0459.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32013D0459.json  
      inflating: EURLEX57K/dataset/dev/32007R0277.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32007R0277.json  
      inflating: EURLEX57K/dataset/dev/32014D0798.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32014D0798.json  
      inflating: EURLEX57K/dataset/dev/32007R1036.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32007R1036.json  
      inflating: EURLEX57K/dataset/dev/31998D0066.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31998D0066.json  
      inflating: EURLEX57K/dataset/dev/32004R1599.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32004R1599.json  
      inflating: EURLEX57K/dataset/dev/32005D0902.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32005D0902.json  
      inflating: EURLEX57K/dataset/dev/31976R0844.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31976R0844.json  
      inflating: EURLEX57K/dataset/dev/31993R1776.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31993R1776.json  
      inflating: EURLEX57K/dataset/dev/32004R2136.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32004R2136.json  
      inflating: EURLEX57K/dataset/dev/32012D0042.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32012D0042.json  
      inflating: EURLEX57K/dataset/dev/32005R0269.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32005R0269.json  
      inflating: EURLEX57K/dataset/dev/31993R2273.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31993R2273.json  
      inflating: EURLEX57K/dataset/dev/32002R0452.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32002R0452.json  
      inflating: EURLEX57K/dataset/dev/32005R1182.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32005R1182.json  
      inflating: EURLEX57K/dataset/dev/31985R1007.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31985R1007.json  
      inflating: EURLEX57K/dataset/dev/32009D0103.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32009D0103.json  
      inflating: EURLEX57K/dataset/dev/32001L0040.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32001L0040.json  
      inflating: EURLEX57K/dataset/dev/31989D0086.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31989D0086.json  
      inflating: EURLEX57K/dataset/dev/31977R1658.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31977R1658.json  
      inflating: EURLEX57K/dataset/dev/31988R3935.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31988R3935.json  
      inflating: EURLEX57K/dataset/dev/32012R1207.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32012R1207.json  
      inflating: EURLEX57K/dataset/dev/32006R1092.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32006R1092.json  
      inflating: EURLEX57K/dataset/dev/32004L0059.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32004L0059.json  
      inflating: EURLEX57K/dataset/dev/31999D0492.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31999D0492.json  
      inflating: EURLEX57K/dataset/dev/31999R0183.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31999R0183.json  
      inflating: EURLEX57K/dataset/dev/32012R0945.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32012R0945.json  
      inflating: EURLEX57K/dataset/dev/32003R1064.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32003R1064.json  
      inflating: EURLEX57K/dataset/dev/31989R0981.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31989R0981.json  
      inflating: EURLEX57K/dataset/dev/32000D0761.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32000D0761.json  
      inflating: EURLEX57K/dataset/dev/32006R0683.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32006R0683.json  
      inflating: EURLEX57K/dataset/dev/31984R0431.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31984R0431.json  
      inflating: EURLEX57K/dataset/dev/32013D0635.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32013D0635.json  
      inflating: EURLEX57K/dataset/dev/32011D0502.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32011D0502.json  
      inflating: EURLEX57K/dataset/dev/32001R2085.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32001R2085.json  
      inflating: EURLEX57K/dataset/dev/32012R0046.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32012R0046.json  
      inflating: EURLEX57K/dataset/dev/32003R1967.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32003R1967.json  
      inflating: EURLEX57K/dataset/dev/31991D0612.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31991D0612.json  
      inflating: EURLEX57K/dataset/dev/31998L0063.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31998L0063.json  
      inflating: EURLEX57K/dataset/dev/32015R0112.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32015R0112.json  
      inflating: EURLEX57K/dataset/dev/31992R3829.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31992R3829.json  
      inflating: EURLEX57K/dataset/dev/32009R1079.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32009R1079.json  
      inflating: EURLEX57K/dataset/dev/32002R1193.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32002R1193.json  
      inflating: EURLEX57K/dataset/dev/32010L0059.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32010L0059.json  
      inflating: EURLEX57K/dataset/dev/32014D0515(02).json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32014D0515(02).json  
      inflating: EURLEX57K/dataset/dev/31998D0420.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31998D0420.json  
      inflating: EURLEX57K/dataset/dev/32011R1182.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32011R1182.json  
      inflating: EURLEX57K/dataset/dev/31993R1330.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31993R1330.json  
      inflating: EURLEX57K/dataset/dev/31995R0582.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31995R0582.json  
      inflating: EURLEX57K/dataset/dev/32006D0412.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32006D0412.json  
      inflating: EURLEX57K/dataset/dev/32014R0765.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32014R0765.json  
      inflating: EURLEX57K/dataset/dev/32001D0679.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32001D0679.json  
      inflating: EURLEX57K/dataset/dev/31993D0460.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31993D0460.json  
      inflating: EURLEX57K/dataset/dev/31996R2010.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31996R2010.json  
      inflating: EURLEX57K/dataset/dev/32004R2209.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32004R2209.json  
      inflating: EURLEX57K/dataset/dev/32005D0017.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32005D0017.json  
      inflating: EURLEX57K/dataset/dev/32007R0631.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32007R0631.json  
      inflating: EURLEX57K/dataset/dev/32002R1255.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32002R1255.json  
      inflating: EURLEX57K/dataset/dev/32008D0277.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32008D0277.json  
      inflating: EURLEX57K/dataset/dev/31994L0024.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31994L0024.json  
      inflating: EURLEX57K/dataset/dev/32011R0843.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32011R0843.json  
      inflating: EURLEX57K/dataset/dev/32001R0942.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32001R0942.json  
      inflating: EURLEX57K/dataset/dev/31994R1622.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31994R1622.json  
      inflating: EURLEX57K/dataset/dev/31999D0191.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31999D0191.json  
      inflating: EURLEX57K/dataset/dev/32003D0237.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32003D0237.json  
      inflating: EURLEX57K/dataset/dev/32009D0816.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32009D0816.json  
      inflating: EURLEX57K/dataset/dev/31996R1846.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31996R1846.json  
      inflating: EURLEX57K/dataset/dev/32001D0550.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32001D0550.json  
      inflating: EURLEX57K/dataset/dev/32013D0336.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32013D0336.json  
      inflating: EURLEX57K/dataset/dev/32000D0632.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32000D0632.json  
      inflating: EURLEX57K/dataset/dev/32005D0294.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32005D0294.json  
      inflating: EURLEX57K/dataset/dev/32009D0953.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32009D0953.json  
      inflating: EURLEX57K/dataset/dev/31999R1784.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31999R1784.json  
      inflating: EURLEX57K/dataset/dev/31997R0089.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31997R0089.json  
      inflating: EURLEX57K/dataset/dev/31989L0083.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31989L0083.json  
      inflating: EURLEX57K/dataset/dev/32005R1878.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32005R1878.json  
      inflating: EURLEX57K/dataset/dev/32003R0399.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32003R0399.json  
      inflating: EURLEX57K/dataset/dev/32008R1032.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32008R1032.json  
      inflating: EURLEX57K/dataset/dev/32012R0400.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32012R0400.json  
      inflating: EURLEX57K/dataset/dev/32001R2139.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32001R2139.json  
      inflating: EURLEX57K/dataset/dev/32009R0541.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32009R0541.json  
      inflating: EURLEX57K/dataset/dev/31988R3889.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31988R3889.json  
      inflating: EURLEX57K/dataset/dev/31993R3531.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31993R3531.json  
      inflating: EURLEX57K/dataset/dev/31987R1799.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31987R1799.json  
      inflating: EURLEX57K/dataset/dev/32002R0852.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32002R0852.json  
      inflating: EURLEX57K/dataset/dev/32002R1740.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32002R1740.json  
      inflating: EURLEX57K/dataset/dev/31998R0189.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31998R0189.json  
      inflating: EURLEX57K/dataset/dev/31989R3895.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31989R3895.json  
      inflating: EURLEX57K/dataset/dev/32004R1167.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32004R1167.json  
      inflating: EURLEX57K/dataset/dev/31986R1785.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31986R1785.json  
      inflating: EURLEX57K/dataset/dev/32001R1191.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32001R1191.json  
      inflating: EURLEX57K/dataset/dev/31994R2665.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31994R2665.json  
      inflating: EURLEX57K/dataset/dev/31982R0200.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31982R0200.json  
      inflating: EURLEX57K/dataset/dev/31994D0719.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31994D0719.json  
      inflating: EURLEX57K/dataset/dev/31993D0572.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31993D0572.json  
      inflating: EURLEX57K/dataset/dev/31992D0610.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31992D0610.json  
      inflating: EURLEX57K/dataset/dev/32004D0637.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32004D0637.json  
      inflating: EURLEX57K/dataset/dev/31992D0305.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31992D0305.json  
      inflating: EURLEX57K/dataset/dev/32001D0784.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32001D0784.json  
      inflating: EURLEX57K/dataset/dev/31991D0350.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31991D0350.json  
      inflating: EURLEX57K/dataset/dev/31972D0279.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31972D0279.json  
      inflating: EURLEX57K/dataset/dev/32013D0418.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32013D0418.json  
      inflating: EURLEX57K/dataset/dev/31981D0601.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31981D0601.json  
      inflating: EURLEX57K/dataset/dev/31982D0654.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31982D0654.json  
      inflating: EURLEX57K/dataset/dev/32003R2035.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32003R2035.json  
      inflating: EURLEX57K/dataset/dev/32014R0732.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32014R0732.json  
      inflating: EURLEX57K/dataset/dev/32001R2302.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32001R2302.json  
      inflating: EURLEX57K/dataset/dev/32005R0101.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32005R0101.json  
      inflating: EURLEX57K/dataset/dev/32014D0223.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32014D0223.json  
      inflating: EURLEX57K/dataset/dev/31999R2393.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31999R2393.json  
      inflating: EURLEX57K/dataset/dev/32014R1089.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32014R1089.json  
      inflating: EURLEX57K/dataset/dev/32004R1022.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32004R1022.json  
      inflating: EURLEX57K/dataset/dev/31995L0038.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31995L0038.json  
      inflating: EURLEX57K/dataset/dev/31990R1332.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31990R1332.json  
      inflating: EURLEX57K/dataset/dev/32006D0916.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32006D0916.json  
      inflating: EURLEX57K/dataset/dev/32012R1180.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32012R1180.json  
      inflating: EURLEX57K/dataset/dev/32014R1123.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32014R1123.json  
      inflating: EURLEX57K/dataset/dev/32012D0003.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32012D0003.json  
      inflating: EURLEX57K/dataset/dev/32015D0268.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32015D0268.json  
      inflating: EURLEX57K/dataset/dev/32005R0228.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32005R0228.json  
      inflating: EURLEX57K/dataset/dev/31979R1640.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31979R1640.json  
      inflating: EURLEX57K/dataset/dev/31981R2454.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31981R2454.json  
      inflating: EURLEX57K/dataset/dev/32012D0453.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32012D0453.json  
      inflating: EURLEX57K/dataset/dev/32005D0369.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32005D0369.json  
      inflating: EURLEX57K/dataset/dev/31991R0092.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31991R0092.json  
      inflating: EURLEX57K/dataset/dev/32011R1356.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32011R1356.json  
      inflating: EURLEX57K/dataset/dev/32011L0100.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32011L0100.json  
      inflating: EURLEX57K/dataset/dev/32004D0908.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32004D0908.json  
      inflating: EURLEX57K/dataset/dev/31997R1270.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31997R1270.json  
      inflating: EURLEX57K/dataset/dev/32001R0850.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32001R0850.json  
      inflating: EURLEX57K/dataset/dev/31979R0114.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31979R0114.json  
      inflating: EURLEX57K/dataset/dev/32009R0146.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32009R0146.json  
      inflating: EURLEX57K/dataset/dev/31978L1020.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31978L1020.json  
      inflating: EURLEX57K/dataset/dev/31990R3499.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31990R3499.json  
      inflating: EURLEX57K/dataset/dev/31986R0747.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31986R0747.json  
      inflating: EURLEX57K/dataset/dev/32002D0417.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32002D0417.json  
      inflating: EURLEX57K/dataset/dev/31992D0539.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31992D0539.json  
      inflating: EURLEX57K/dataset/dev/32005R1985.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32005R1985.json  
      inflating: EURLEX57K/dataset/dev/32005D0386.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32005D0386.json  
      inflating: EURLEX57K/dataset/dev/31992D0493.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31992D0493.json  
      inflating: EURLEX57K/dataset/dev/32001R0503.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32001R0503.json  
      inflating: EURLEX57K/dataset/dev/32003R1926.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32003R1926.json  
      inflating: EURLEX57K/dataset/dev/32009R1110.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32009R1110.json  
      inflating: EURLEX57K/dataset/dev/32010D0436.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32010D0436.json  
      inflating: EURLEX57K/dataset/dev/32011R0615.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32011R0615.json  
      inflating: EURLEX57K/dataset/dev/32010R0577.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32010R0577.json  
      inflating: EURLEX57K/dataset/dev/32006R1694.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32006R1694.json  
      inflating: EURLEX57K/dataset/dev/31996L0016.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31996L0016.json  
      inflating: EURLEX57K/dataset/dev/32001R1105.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32001R1105.json  
      inflating: EURLEX57K/dataset/dev/32002R1150.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32002R1150.json  
      inflating: EURLEX57K/dataset/dev/31992R3413.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31992R3413.json  
      inflating: EURLEX57K/dataset/dev/32014R1158.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32014R1158.json  
      inflating: EURLEX57K/dataset/dev/32002R1015.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32002R1015.json  
      inflating: EURLEX57K/dataset/dev/31973L0173.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31973L0173.json  
      inflating: EURLEX57K/dataset/dev/32008D0037.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32008D0037.json  
      inflating: EURLEX57K/dataset/dev/31992R3106.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31992R3106.json  
      inflating: EURLEX57K/dataset/dev/32009D0705.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32009D0705.json  
      inflating: EURLEX57K/dataset/dev/32001R0651.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32001R0651.json  
      inflating: EURLEX57K/dataset/dev/32010D0573.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32010D0573.json  
      inflating: EURLEX57K/dataset/dev/32013D0526.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32013D0526.json  
      inflating: EURLEX57K/dataset/dev/32007R0758.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32007R0758.json  
      inflating: EURLEX57K/dataset/dev/32012R0355.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32012R0355.json  
      inflating: EURLEX57K/dataset/dev/32009R1055.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32009R1055.json  
      inflating: EURLEX57K/dataset/dev/31999R1481.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31999R1481.json  
      inflating: EURLEX57K/dataset/dev/32010R0598.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32010R0598.json  
      inflating: EURLEX57K/dataset/dev/32005R2002.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32005R2002.json  
      inflating: EURLEX57K/dataset/dev/31994D0298.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31994D0298.json  
      inflating: EURLEX57K/dataset/dev/32003R0136.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32003R0136.json  
      inflating: EURLEX57K/dataset/dev/32006R0590.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32006R0590.json  
      inflating: EURLEX57K/dataset/dev/31998D0660.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31998D0660.json  
      inflating: EURLEX57K/dataset/dev/31984R3227.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31984R3227.json  
      inflating: EURLEX57K/dataset/dev/31991R0905.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31991R0905.json  
      inflating: EURLEX57K/dataset/dev/32009R0097.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32009R0097.json  
      inflating: EURLEX57K/dataset/dev/32004D0836.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32004D0836.json  
      inflating: EURLEX57K/dataset/dev/32009L0080.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32009L0080.json  
      inflating: EURLEX57K/dataset/dev/31976R0311.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31976R0311.json  
      inflating: EURLEX57K/dataset/dev/32007R0021.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32007R0021.json  
      inflating: EURLEX57K/dataset/dev/32002D0196.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32002D0196.json  
      inflating: EURLEX57K/dataset/dev/32011D0538.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32011D0538.json  
      inflating: EURLEX57K/dataset/dev/32004R0424.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32004R0424.json  
      inflating: EURLEX57K/dataset/dev/31986R1984.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31986R1984.json  
      inflating: EURLEX57K/dataset/dev/31997R2098.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31997R2098.json  
      inflating: EURLEX57K/dataset/dev/31996D0639.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31996D0639.json  
      inflating: EURLEX57K/dataset/dev/32007D0475.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32007D0475.json  
      inflating: EURLEX57K/dataset/dev/31995R0687.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31995R0687.json  
      inflating: EURLEX57K/dataset/dev/32004R0561.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32004R0561.json  
      inflating: EURLEX57K/dataset/dev/32003R1818.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32003R1818.json  
      inflating: EURLEX57K/dataset/dev/32001R1386.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32001R1386.json  
      inflating: EURLEX57K/dataset/dev/32014R1271.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32014R1271.json  
      inflating: EURLEX57K/dataset/dev/31986R3055.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31986R3055.json  
      inflating: EURLEX57K/dataset/dev/31997D0292(01).json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31997D0292(01).json  
      inflating: EURLEX57K/dataset/dev/32006R1047.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32006R1047.json  
      inflating: EURLEX57K/dataset/dev/31997D0808.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31997D0808.json  
      inflating: EURLEX57K/dataset/dev/31988R2426.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31988R2426.json  
      inflating: EURLEX57K/dataset/dev/32003R1635.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32003R1635.json  
      inflating: EURLEX57K/dataset/dev/31996R1647.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31996R1647.json  
      inflating: EURLEX57K/dataset/dev/31989D0268.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31989D0268.json  
      inflating: EURLEX57K/dataset/dev/32006D0469.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32006D0469.json  
      inflating: EURLEX57K/dataset/dev/32003R2019.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32003R2019.json  
      inflating: EURLEX57K/dataset/dev/31992R2255.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31992R2255.json  
      inflating: EURLEX57K/dataset/dev/31986D0446.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31986D0446.json  
      inflating: EURLEX57K/dataset/dev/31988R0948.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31988R0948.json  
      inflating: EURLEX57K/dataset/dev/32006D0039.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32006D0039.json  
      inflating: EURLEX57K/dataset/dev/32006R0528.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32006R0528.json  
      inflating: EURLEX57K/dataset/dev/32002R1804.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32002R1804.json  
      inflating: EURLEX57K/dataset/dev/31979L0343.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31979L0343.json  
      inflating: EURLEX57K/dataset/dev/31982D0382.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31982D0382.json  
      inflating: EURLEX57K/dataset/dev/32001R2284.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32001R2284.json  
      inflating: EURLEX57K/dataset/dev/32003R0161.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32003R0161.json  
      inflating: EURLEX57K/dataset/dev/32011R0707.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32011R0707.json  
      inflating: EURLEX57K/dataset/dev/31997R1899.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31997R1899.json  
      inflating: EURLEX57K/dataset/dev/32001D0747.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32001D0747.json  
      inflating: EURLEX57K/dataset/dev/31977R0474.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31977R0474.json  
      inflating: EURLEX57K/dataset/dev/31986R2306.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31986R2306.json  
      inflating: EURLEX57K/dataset/dev/31982D0414.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31982D0414.json  
      inflating: EURLEX57K/dataset/dev/32014R0122.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32014R0122.json  
      inflating: EURLEX57K/dataset/dev/32003R0248.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32003R0248.json  
      inflating: EURLEX57K/dataset/dev/31997R0158.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31997R0158.json  
      inflating: EURLEX57K/dataset/dev/32002R1691.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32002R1691.json  
      inflating: EURLEX57K/dataset/dev/31989D0004.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31989D0004.json  
      inflating: EURLEX57K/dataset/dev/32006R1010.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32006R1010.json  
      inflating: EURLEX57K/dataset/dev/31991R1355.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31991R1355.json  
      inflating: EURLEX57K/dataset/dev/32007D0588.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32007D0588.json  
      inflating: EURLEX57K/dataset/dev/32012D0185.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32012D0185.json  
      inflating: EURLEX57K/dataset/dev/32001R2457.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32001R2457.json  
      inflating: EURLEX57K/dataset/dev/31994D0559.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31994D0559.json  
      inflating: EURLEX57K/dataset/dev/32002R2052.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32002R2052.json  
      inflating: EURLEX57K/dataset/dev/32001R2007.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32001R2007.json  
      inflating: EURLEX57K/dataset/dev/32011R0091.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32011R0091.json  
      inflating: EURLEX57K/dataset/dev/31992R3901.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31992R3901.json  
      inflating: EURLEX57K/dataset/dev/32011D0715.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32011D0715.json  
      inflating: EURLEX57K/dataset/dev/32004D0318.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32004D0318.json  
      inflating: EURLEX57K/dataset/dev/32001R2292.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32001R2292.json  
      inflating: EURLEX57K/dataset/dev/32003D0173.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32003D0173.json  
      inflating: EURLEX57K/dataset/dev/32003D0489.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32003D0489.json  
      inflating: EURLEX57K/dataset/dev/32002R0700.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32002R0700.json  
      inflating: EURLEX57K/dataset/dev/32004R1018.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32004R1018.json  
      inflating: EURLEX57K/dataset/dev/31999D0685.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31999D0685.json  
      inflating: EURLEX57K/dataset/dev/32009D0251.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32009D0251.json  
      inflating: EURLEX57K/dataset/dev/31977L0249.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31977L0249.json  
      inflating: EURLEX57K/dataset/dev/32008R0422.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32008R0422.json  
      inflating: EURLEX57K/dataset/dev/32010R0865.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32010R0865.json  
      inflating: EURLEX57K/dataset/dev/31998R2161.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31998R2161.json  
      inflating: EURLEX57K/dataset/dev/31987L0234.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31987L0234.json  
      inflating: EURLEX57K/dataset/dev/32006D0886.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32006D0886.json  
      inflating: EURLEX57K/dataset/dev/32008D0563.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32008D0563.json  
      inflating: EURLEX57K/dataset/dev/32005R0838.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32005R0838.json  
      inflating: EURLEX57K/dataset/dev/32008R0567.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32008R0567.json  
      inflating: EURLEX57K/dataset/dev/31997R1026.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31997R1026.json  
      inflating: EURLEX57K/dataset/dev/31989R2247.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31989R2247.json  
      inflating: EURLEX57K/dataset/dev/32002R1404.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32002R1404.json  
      inflating: EURLEX57K/dataset/dev/32010R1262.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32010R1262.json  
      inflating: EURLEX57K/dataset/dev/31995R0853.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31995R0853.json  
      inflating: EURLEX57K/dataset/dev/32003R0527.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32003R0527.json  
      inflating: EURLEX57K/dataset/dev/32002R2228.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32002R2228.json  
      inflating: EURLEX57K/dataset/dev/31994D0723.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31994D0723.json  
      inflating: EURLEX57K/dataset/dev/32013R0426.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32013R0426.json  
      inflating: EURLEX57K/dataset/dev/32011D0200.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32011D0200.json  
      inflating: EURLEX57K/dataset/dev/32006R1513.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32006R1513.json  
      inflating: EURLEX57K/dataset/dev/31992R1603.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31992R1603.json  
      inflating: EURLEX57K/dataset/dev/31987R2422.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31987R2422.json  
      inflating: EURLEX57K/dataset/dev/32005R1815.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32005R1815.json  
      inflating: EURLEX57K/dataset/dev/31993D0231.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31993D0231.json  
      inflating: EURLEX57K/dataset/dev/31996R0386.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31996R0386.json  
      inflating: EURLEX57K/dataset/dev/32006R0752.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32006R0752.json  
      inflating: EURLEX57K/dataset/dev/32011R0068.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32011R0068.json  
      inflating: EURLEX57K/dataset/dev/31989R0850.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31989R0850.json  
      inflating: EURLEX57K/dataset/dev/32002R0583.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32002R0583.json  
      inflating: EURLEX57K/dataset/dev/31984R2162.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31984R2162.json  
      inflating: EURLEX57K/dataset/dev/31999L0100.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31999L0100.json  
      inflating: EURLEX57K/dataset/dev/32005D0353.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32005D0353.json  
      inflating: EURLEX57K/dataset/dev/31996R1578.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31996R1578.json  
      inflating: EURLEX57K/dataset/dev/31999D0406.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31999D0406.json  
      inflating: EURLEX57K/dataset/dev/31988R1836.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31988R1836.json  
      inflating: EURLEX57K/dataset/dev/31996R1128.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31996R1128.json  
      inflating: EURLEX57K/dataset/dev/32001R1797.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32001R1797.json  
      inflating: EURLEX57K/dataset/dev/32002R1516.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32002R1516.json  
      inflating: EURLEX57K/dataset/dev/32011R1012.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32011R1012.json  
      inflating: EURLEX57K/dataset/dev/31999R1881.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31999R1881.json  
      inflating: EURLEX57K/dataset/dev/32008R0475.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32008R0475.json  
      inflating: EURLEX57K/dataset/dev/32013R0867.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32013R0867.json  
      inflating: EURLEX57K/dataset/dev/32001R1113.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32001R1113.json  
      inflating: EURLEX57K/dataset/dev/31975L0129.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31975L0129.json  
      inflating: EURLEX57K/dataset/dev/32002R0307.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32002R0307.json  
      inflating: EURLEX57K/dataset/dev/32013R0534.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32013R0534.json  
      inflating: EURLEX57K/dataset/dev/32002D0246.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32002D0246.json  
      inflating: EURLEX57K/dataset/dev/32008R1264.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32008R1264.json  
      inflating: EURLEX57K/dataset/dev/32000D0171.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32000D0171.json  
      inflating: EURLEX57K/dataset/dev/32009R1106.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32009R1106.json  
      inflating: EURLEX57K/dataset/dev/32000D0464.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32000D0464.json  
      inflating: EURLEX57K/dataset/dev/32001D0706.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32001D0706.json  
      inflating: EURLEX57K/dataset/dev/32010D0135.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32010D0135.json  
      inflating: EURLEX57K/dataset/dev/32011R0316.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32011R0316.json  
      inflating: EURLEX57K/dataset/dev/32015R0128.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32015R0128.json  
      inflating: EURLEX57K/dataset/dev/32004R1809.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32004R1809.json  
      inflating: EURLEX57K/dataset/dev/31994R0635.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31994R0635.json  
      inflating: EURLEX57K/dataset/dev/32002R0612.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32002R0612.json  
      inflating: EURLEX57K/dataset/dev/32008D0922.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32008D0922.json  
      inflating: EURLEX57K/dataset/dev/31996R1743.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31996R1743.json  
      inflating: EURLEX57K/dataset/dev/32009R0202.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32009R0202.json  
      inflating: EURLEX57K/dataset/dev/31993R2930.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31993R2930.json  
      inflating: EURLEX57K/dataset/dev/32005R1638.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32005R1638.json  
      inflating: EURLEX57K/dataset/dev/31995R2993.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31995R2993.json  
      inflating: EURLEX57K/dataset/dev/31991R0440.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31991R0440.json  
      inflating: EURLEX57K/dataset/dev/31986D0391.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31986D0391.json  
      inflating: EURLEX57K/dataset/dev/32000R2631.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32000R2631.json  
      inflating: EURLEX57K/dataset/dev/31996R1939.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31996R1939.json  
      inflating: EURLEX57K/dataset/dev/31981D0400.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31981D0400.json  
      inflating: EURLEX57K/dataset/dev/32006R1817.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32006R1817.json  
      inflating: EURLEX57K/dataset/dev/31992D0554.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31992D0554.json  
      inflating: EURLEX57K/dataset/dev/32011R0195.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32011R0195.json  
      inflating: EURLEX57K/dataset/dev/31997D0408.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31997D0408.json  
      inflating: EURLEX57K/dataset/dev/32014R0533.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32014R0533.json  
      inflating: EURLEX57K/dataset/dev/32007L0020.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32007L0020.json  
      inflating: EURLEX57K/dataset/dev/32004R1223.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32004R1223.json  
      inflating: EURLEX57K/dataset/dev/32004R0961.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32004R0961.json  
      inflating: EURLEX57K/dataset/dev/32004R1673.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32004R1673.json  
      inflating: EURLEX57K/dataset/dev/31999D0514.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31999D0514.json  
      inflating: EURLEX57K/dataset/dev/31996R1190.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31996R1190.json  
      inflating: EURLEX57K/dataset/dev/31988D0327.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31988D0327.json  
      inflating: EURLEX57K/dataset/dev/31999R0510.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31999R0510.json  
      inflating: EURLEX57K/dataset/dev/32011R1291.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32011R1291.json  
      inflating: EURLEX57K/dataset/dev/31996R1485.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31996R1485.json  
      inflating: EURLEX57K/dataset/dev/32002R0887.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32002R0887.json  
      inflating: EURLEX57K/dataset/dev/32001R0978.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32001R0978.json  
      inflating: EURLEX57K/dataset/dev/32006R1401.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32006R1401.json  
      inflating: EURLEX57K/dataset/dev/31994R1618.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31994R1618.json  
      inflating: EURLEX57K/dataset/dev/31992R2097.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31992R2097.json  
      inflating: EURLEX57K/dataset/dev/31983D0388.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31983D0388.json  
      inflating: EURLEX57K/dataset/dev/32002R0184.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32002R0184.json  
      inflating: EURLEX57K/dataset/dev/32007R0522.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32007R0522.json  
      inflating: EURLEX57K/dataset/dev/32005R0245.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32005R0245.json  
      inflating: EURLEX57K/dataset/dev/31995D0380.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31995D0380.json  
      inflating: EURLEX57K/dataset/dev/31989R1200.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._31989R1200.json  
      inflating: EURLEX57K/dataset/dev/32015D0205.json  
      inflating: EURLEX57K/__MACOSX/dataset/dev/._32015D0205.json  
      inflating: EURLEX57K/__MACOSX/dataset/._dev  



```
cat EURLEX57K/dataset/dev/31995D0380.json
```

    {"celex_id": "31995D0380", "uri": "http://publications.europa.eu/resource/cellar/1ea4956f-28c2-4193-a15d-352d59bdfd49", "type": "Decision", "concepts": ["1895", "2711", "4057", "4257", "5962"], "title": "95/380/EC: Commission Decision of 18 September 1995 amending Commission Decisions 94/432/EC, 94/433/EC and 94/434/EC laying down detailed rules for the application of Council Directives 93/23/EEC on the statistical surveys to be carried out on pig production, 93/24/EEC on the statistical surveys to be carried out on bovine animal production and 93/25/EEC on the statistical surveys to be carried out on sheep and goat stocks\n", "header": "COMMISSION DECISION  of 18 September 1995 amending Commission Decisions 94/432/EC, 94/433/EC and 94/434/EC laying down  detailed rules for the application of Council Directives 93/23/EEC on the statistical surveys to be  carried out on pig production, 93/24/EEC on the statistical surveys to be carried out on bovine  animal production and 93/25/EEC on the statistical surveys to be carried out on sheep and goat  stocks (Text with EEA relevance) (95/380/EC)\nTHE COMMISSION OF THE EUROPEAN  COMMUNITIES", "recitals": ",\nHaving regard to the Treaty establishing the European Community,\nHaving regard to Council Directive 93/23/EEC of 1 June 1993 on the statistical surveys to be  carried out on pig production  (1), and in particular Articles 1 (3) and 6 (3) thereof,\nHaving regard to Council Directive 93/24/EEC of 1 June 1993 on the statistical surveys to be  carried out on bovine animal production  (2), and in particular Articles 1 (3) and 6 (3) thereof,\nHaving regard to Council Directive 93/25/EEC of 1 June 1993, on the statistical surveys to be  carried out on sheep and goat stocks  (3), and in particular Articles 1 (4) and 7 (2) thereof,\nHaving regard to Commission Decision 94/432/EC of 30 May 1994 laying down detailed rules for the  application of the abovementioned Council Directive 93/23/EEC as regards the statistical surveys on  pig population and production  (4),\nHaving regard to Commission Decision 94/433/EC of 30 May 1994 laying down detailed rules for the  application of the abovementioned Council Directive 93/24/EEC as regards the statistical surveys on  cattle population and production, and amending the said Directive  (5),\nHaving regard to Commission Decision 94/434/EC of 30 May 1994 laying down detailed rules for the  application of the abovementioned Council Directive 93/25/EEC as regards the statistical surveys on  sheep and goat population and production  (6),\nWhereas by reason of the accession of Austria, Finland and Sweden it is necessary to make certain  technical adaptations to the abovementioned Decisions and to extend certain derogations to the new  Member States;\nWhereas the abovementioned Directives and Decisions provide for the possibility, in the case of  Member States whose pig, bovine animal and goat populations make up only a small percentage of the  overall populations of the Community, of granting derogations aimed at reducing the number of  annual surveys to be conducted;\nWhereas the envisaged measures are in line with the opinion of the Standing Committee on  Agricultural Statistics set up by Council Decision 72/279/EEC  (7),", "main_body": ["Decision 94/432/EC laying down detailed rules for the application of  Directive 93/23/EEC shall be amended as follows:\n1.  Annex I shall be supplemented with the following text:\n'Austria: Bundeslaender Finland: Etelae-Suomi Sisae-Suomi Pohjanmaa Pohjois-Suomi Sweden: 8 Riksomraaden`.\n2.  Annex II, the text of footnotes (a) and (b) shall be amended to read:\n'(a) Breakdown optional for NL, DK, S.\n(b) Breakdown optional for P, L, GR, S.` 3.  Annex IV (b) shall be supplemented with the following text:\n'Finland Sweden`.\n4.  Annex IV (e) shall be supplemented with the following text under the heading 'a given month of  the year`:\n'Sweden, June  `.", "Decision 94/433/EC laying down detailed rules for the application of Directive  93/24/EEC shall be amended as follows:\n1.  Annex II shall be supplemented as follows:\n'Austria: Bundeslaender Finland: Etelae-Suomi Sisae-Suomi Pohjanmaa Pohjois-Suomi Sweden: 8 Riksomraaden`.\n2.  Annex III, the text of footnotes (a), (b) and (c) shall be amended to read:\n'(a) Breakdown optional for NL, DK, S.\n(b) Breakdown optional for P, L, GR, S.\n(c) Breakdown optional for P, L, GR, F, S.` 3.  Annex V, the text of footnote (d) shall be supplemented with the following text:\n'Sweden`.\n4.  Annex V, the text of footnote (e) shall be supplemented with the following text under the  heading 'May/June`:\n'Sweden`.", "Decision 94/434/EC laying down detailed rules for the application of Directive  93/25/EEC shall be amended as follows:\n1.  Annex II shall be supplemented as follows:\n'Austria: Bundeslaender Finland: Etelae-Suomi Sisae-Suomi Pohjanmaa Pohjois-Suomi Sweden: -  for sheep: 8 Riksomraaden -  for goats: -`.\n2.  Annex III, Table 1, the text of footnotes (a), (b) and (c) shall be amended to read:\n'(a) Breakdown optional for L, B, DK, S.\n(b) Optional for D, NL, S.\n(c) Optional for B, D, IRL, NL, A, FIN, S, UK.` 3.  Annex III, Table 2, the text of footnotes (a) and (c) shall be amended to read:\n'(a) D, L, B, UK, IRL, S.\n(c) D, NL, S.`", "This Decision is addressed to the Member States."], "attachments": "Done at Brussels, 18 September 1995.\nFor the Commission Yves-Thibault DE SILGUY Member of the Commission"}


```
import json
```


```
with open('EURLEX57K/dataset/dev/31995D0380.json') as file:
  data = json.load(file)
```


```
data
```




    {'attachments': 'Done at Brussels, 18 September 1995.\nFor the Commission Yves-Thibault DE SILGUY Member of the Commission',
     'celex_id': '31995D0380',
     'concepts': ['1895', '2711', '4057', '4257', '5962'],
     'header': 'COMMISSION DECISION  of 18 September 1995 amending Commission Decisions 94/432/EC, 94/433/EC and 94/434/EC laying down  detailed rules for the application of Council Directives 93/23/EEC on the statistical surveys to be  carried out on pig production, 93/24/EEC on the statistical surveys to be carried out on bovine  animal production and 93/25/EEC on the statistical surveys to be carried out on sheep and goat  stocks (Text with EEA relevance) (95/380/EC)\nTHE COMMISSION OF THE EUROPEAN  COMMUNITIES',
     'main_body': ["Decision 94/432/EC laying down detailed rules for the application of  Directive 93/23/EEC shall be amended as follows:\n1.  Annex I shall be supplemented with the following text:\n'Austria: Bundeslaender Finland: Etelae-Suomi Sisae-Suomi Pohjanmaa Pohjois-Suomi Sweden: 8 Riksomraaden`.\n2.  Annex II, the text of footnotes (a) and (b) shall be amended to read:\n'(a) Breakdown optional for NL, DK, S.\n(b) Breakdown optional for P, L, GR, S.` 3.  Annex IV (b) shall be supplemented with the following text:\n'Finland Sweden`.\n4.  Annex IV (e) shall be supplemented with the following text under the heading 'a given month of  the year`:\n'Sweden, June  `.",
      "Decision 94/433/EC laying down detailed rules for the application of Directive  93/24/EEC shall be amended as follows:\n1.  Annex II shall be supplemented as follows:\n'Austria: Bundeslaender Finland: Etelae-Suomi Sisae-Suomi Pohjanmaa Pohjois-Suomi Sweden: 8 Riksomraaden`.\n2.  Annex III, the text of footnotes (a), (b) and (c) shall be amended to read:\n'(a) Breakdown optional for NL, DK, S.\n(b) Breakdown optional for P, L, GR, S.\n(c) Breakdown optional for P, L, GR, F, S.` 3.  Annex V, the text of footnote (d) shall be supplemented with the following text:\n'Sweden`.\n4.  Annex V, the text of footnote (e) shall be supplemented with the following text under the  heading 'May/June`:\n'Sweden`.",
      "Decision 94/434/EC laying down detailed rules for the application of Directive  93/25/EEC shall be amended as follows:\n1.  Annex II shall be supplemented as follows:\n'Austria: Bundeslaender Finland: Etelae-Suomi Sisae-Suomi Pohjanmaa Pohjois-Suomi Sweden: -  for sheep: 8 Riksomraaden -  for goats: -`.\n2.  Annex III, Table 1, the text of footnotes (a), (b) and (c) shall be amended to read:\n'(a) Breakdown optional for L, B, DK, S.\n(b) Optional for D, NL, S.\n(c) Optional for B, D, IRL, NL, A, FIN, S, UK.` 3.  Annex III, Table 2, the text of footnotes (a) and (c) shall be amended to read:\n'(a) D, L, B, UK, IRL, S.\n(c) D, NL, S.`",
      'This Decision is addressed to the Member States.'],
     'recitals': ',\nHaving regard to the Treaty establishing the European Community,\nHaving regard to Council Directive 93/23/EEC of 1 June 1993 on the statistical surveys to be  carried out on pig production  (1), and in particular Articles 1 (3) and 6 (3) thereof,\nHaving regard to Council Directive 93/24/EEC of 1 June 1993 on the statistical surveys to be  carried out on bovine animal production  (2), and in particular Articles 1 (3) and 6 (3) thereof,\nHaving regard to Council Directive 93/25/EEC of 1 June 1993, on the statistical surveys to be  carried out on sheep and goat stocks  (3), and in particular Articles 1 (4) and 7 (2) thereof,\nHaving regard to Commission Decision 94/432/EC of 30 May 1994 laying down detailed rules for the  application of the abovementioned Council Directive 93/23/EEC as regards the statistical surveys on  pig population and production  (4),\nHaving regard to Commission Decision 94/433/EC of 30 May 1994 laying down detailed rules for the  application of the abovementioned Council Directive 93/24/EEC as regards the statistical surveys on  cattle population and production, and amending the said Directive  (5),\nHaving regard to Commission Decision 94/434/EC of 30 May 1994 laying down detailed rules for the  application of the abovementioned Council Directive 93/25/EEC as regards the statistical surveys on  sheep and goat population and production  (6),\nWhereas by reason of the accession of Austria, Finland and Sweden it is necessary to make certain  technical adaptations to the abovementioned Decisions and to extend certain derogations to the new  Member States;\nWhereas the abovementioned Directives and Decisions provide for the possibility, in the case of  Member States whose pig, bovine animal and goat populations make up only a small percentage of the  overall populations of the Community, of granting derogations aimed at reducing the number of  annual surveys to be conducted;\nWhereas the envisaged measures are in line with the opinion of the Standing Committee on  Agricultural Statistics set up by Council Decision 72/279/EEC  (7),',
     'title': '95/380/EC: Commission Decision of 18 September 1995 amending Commission Decisions 94/432/EC, 94/433/EC and 94/434/EC laying down detailed rules for the application of Council Directives 93/23/EEC on the statistical surveys to be carried out on pig production, 93/24/EEC on the statistical surveys to be carried out on bovine animal production and 93/25/EEC on the statistical surveys to be carried out on sheep and goat stocks\n',
     'type': 'Decision',
     'uri': 'http://publications.europa.eu/resource/cellar/1ea4956f-28c2-4193-a15d-352d59bdfd49'}




```
!rm datasets.zip
!rm -rf EURLEX57K/__MACOSX
!mv EURLEX57K/dataset/* EURLEX57K/
!rm -rf EURLEX57K/dataset
!wget -O EURLEX57K/EURLEX57K.json http://nlp.cs.aueb.gr/software_and_datasets/EURLEX57K/eurovoc_en.json
```

    --2020-08-23 11:40:53--  http://nlp.cs.aueb.gr/software_and_datasets/EURLEX57K/eurovoc_en.json
    Resolving nlp.cs.aueb.gr (nlp.cs.aueb.gr)... 195.251.248.252
    Connecting to nlp.cs.aueb.gr (nlp.cs.aueb.gr)|195.251.248.252|:80... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 921898 (900K) [text/plain]
    Saving to: ‘EURLEX57K/EURLEX57K.json’
    
    EURLEX57K/EURLEX57K 100%[===================>] 900.29K   159KB/s    in 5.7s    
    
    2020-08-23 11:40:59 (159 KB/s) - ‘EURLEX57K/EURLEX57K.json’ saved [921898/921898]
    



```
import glob, os
from collections import Counter
import tqdm, json
```


```
DATA_SET_DIR = './'
```


```
train_files = glob.glob(os.path.join(DATA_SET_DIR, 'EURLEX57K', 'train', '*.json'))
```


```
documents = []
labels = []

for filename in tqdm.tqdm(train_files):
  with open(filename) as file:
      data = json.load(file)
      labels.append(data['concepts'])
      documents.append(data['main_body'])
```

    100%|██████████| 45000/45000 [00:03<00:00, 12726.01it/s]



```
import pandas as pd
```


```
df = pd.DataFrame({'text':documents,'label':labels})
```


```
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>text</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>[The Annex to Regulation (EEC) No 3846/87 is h...</td>
      <td>[2068, 2069, 2734, 3568, 4381]</td>
    </tr>
    <tr>
      <th>1</th>
      <td>[Third States’ contributions\n1.   The contrib...</td>
      <td>[2084, 5556, 5744, 5889, 922]</td>
    </tr>
    <tr>
      <th>2</th>
      <td>[Decision 2010/96/CFSP is hereby amended as fo...</td>
      <td>[218, 4212, 5610, 6927, 8482]</td>
    </tr>
    <tr>
      <th>3</th>
      <td>[Mr Mario MINOJA is hereby appointed a member ...</td>
      <td>[1519, 3559, 6054]</td>
    </tr>
    <tr>
      <th>4</th>
      <td>[The representative prices and additional duti...</td>
      <td>[2635, 2733, 3191, 4080]</td>
    </tr>
  </tbody>
</table>
</div>




```
df['flatten_text'] = df['text'].apply(lambda x: ' '.join(x))
```


```
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>text</th>
      <th>label</th>
      <th>flatten_text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>[The Annex to Regulation (EEC) No 3846/87 is h...</td>
      <td>[2068, 2069, 2734, 3568, 4381]</td>
      <td>The Annex to Regulation (EEC) No 3846/87 is he...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>[Third States’ contributions\n1.   The contrib...</td>
      <td>[2084, 5556, 5744, 5889, 922]</td>
      <td>Third States’ contributions\n1.   The contribu...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>[Decision 2010/96/CFSP is hereby amended as fo...</td>
      <td>[218, 4212, 5610, 6927, 8482]</td>
      <td>Decision 2010/96/CFSP is hereby amended as fol...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>[Mr Mario MINOJA is hereby appointed a member ...</td>
      <td>[1519, 3559, 6054]</td>
      <td>Mr Mario MINOJA is hereby appointed a member o...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>[The representative prices and additional duti...</td>
      <td>[2635, 2733, 3191, 4080]</td>
      <td>The representative prices and additional dutie...</td>
    </tr>
  </tbody>
</table>
</div>




```
df_new = df.explode('label')
```


```
df_new.shape
```




    (228322, 3)




```
concept_df = pd.DataFrame({'concept_id':df_new.label.value_counts().index,'freq':df_new.label.value_counts().values})
```


```
concept_df = concept_df[concept_df.freq>2000]
```


```
concept_df.shape
```




    (11, 2)




```
concept_df.concept_id.values
```




    array(['1309', '3568', '1118', '1605', '693', '2635', '20', '161', '2300',
           '1644', '2771'], dtype=object)




```
df_new = df_new[df_new.label.isin(concept_df.concept_id.values)]
```


```
df_new.shape
```




    (30684, 3)




```
df_new['id'] = df_new.index
```


```
df_new = df_new[['id', 'flatten_text', 'label']]
df_new.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>flatten_text</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>The Annex to Regulation (EEC) No 3846/87 is he...</td>
      <td>3568</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>The representative prices and additional dutie...</td>
      <td>2635</td>
    </tr>
    <tr>
      <th>7</th>
      <td>7</td>
      <td>1.   Import licences applied for by traditiona...</td>
      <td>1309</td>
    </tr>
    <tr>
      <th>7</th>
      <td>7</td>
      <td>1.   Import licences applied for by traditiona...</td>
      <td>161</td>
    </tr>
    <tr>
      <th>7</th>
      <td>7</td>
      <td>1.   Import licences applied for by traditiona...</td>
      <td>1644</td>
    </tr>
  </tbody>
</table>
</div>




```
print('1. Unique ids :', len(df_new.id.unique()))
df_new_transformed = df_new.groupby(['id','label','flatten_text']).size().reset_index(name='count')
print('2. Unique ids :', len(df_new_transformed.id.unique()))
df_new_transformed['value'] = 1
df_new_transformed_temp = df_new_transformed.pivot(index=df_new_transformed.id, columns='label')['value'].fillna(0)

df_new_transformed_temp['id'] = df_new_transformed_temp.index
df_new_transformed_temp = df_new_transformed_temp.reset_index(drop=True)
```

    1. Unique ids : 16310
    2. Unique ids : 16310



```
df_new_transformed_temp.columns, df_new_transformed_temp.shape
```




    (Index(['1118', '1309', '1605', '161', '1644', '20', '2300', '2635', '2771',
            '3568', '693', 'id'],
           dtype='object', name='label'), (16310, 12))




```
text_id_df = df_new.drop_duplicates(['id'])[['id','flatten_text']]
final_df = text_id_df.merge(df_new_transformed_temp, on='id')
```


```
final_df.shape
```




    (16310, 13)




```
final_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>flatten_text</th>
      <th>1118</th>
      <th>1309</th>
      <th>1605</th>
      <th>161</th>
      <th>1644</th>
      <th>20</th>
      <th>2300</th>
      <th>2635</th>
      <th>2771</th>
      <th>3568</th>
      <th>693</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>The Annex to Regulation (EEC) No 3846/87 is he...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4</td>
      <td>The representative prices and additional dutie...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>7</td>
      <td>1.   Import licences applied for by traditiona...</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>13</td>
      <td>The "Direction des Services de l'Agriculture: ...</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>16</td>
      <td>The import duties in the rice sector referred ...</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>




```
final_df.columns
```




    Index(['id', 'flatten_text', '1118', '1309', '1605', '161', '1644', '20',
           '2300', '2635', '2771', '3568', '693'],
          dtype='object')




```
x_text = final_df.flatten_text.values.tolist()
y_label = final_df[['1118', '1309', '1605', '161', '1644', '20',
       '2300', '2635', '2771', '3568', '693']].to_numpy()
```


```
y_label.shape
```




    (16310, 11)




```
text_vectorizer = preprocessing.TextVectorization(output_mode="int")
text_vectorizer.adapt(x_text)
```


```
vocab = text_vectorizer.get_vocabulary()
len(vocab)
```




    33103




```
!pip install model-x
```

    Collecting model-x
      Downloading https://files.pythonhosted.org/packages/11/1f/88235ebb600ee3aebb1f8e5457c197dc30fed90b953b1e67a3194e3bd1f3/model_X-0.1.5-py3-none-any.whl
    Installing collected packages: model-x
    Successfully installed model-x-0.1.5



```
from model_X.bilstm_architectures import *
```


```
tf.keras.backend.clear_session()
inputs = tf.keras.Input(shape=(1,), dtype="string")
x = text_vectorizer(inputs)
x = BiLSTMGRUAttention(nb_words=len(vocab), embedding_size=64, is_embedding_trainable=True,attention_type='ScaledDotProductAttention' ,h_lstm=64, h_gru=32)(x)
outputs = tf.keras.layers.Dense(11, activation='sigmoid')(x)
model = tf.keras.Model(inputs, outputs)
model.compile('adam','binary_crossentropy','accuracy')
model.summary()
```

    Model: "functional_1"
    __________________________________________________________________________________________________
    Layer (type)                    Output Shape         Param #     Connected to                     
    ==================================================================================================
    input_1 (InputLayer)            [(None, 1)]          0                                            
    __________________________________________________________________________________________________
    text_vectorization (TextVectori (None, None)         0           input_1[0][0]                    
    __________________________________________________________________________________________________
    embedding (Embedding)           (None, None, 64)     2118592     text_vectorization[0][0]         
    __________________________________________________________________________________________________
    spatial_dropout1d (SpatialDropo (None, None, 64)     0           embedding[0][0]                  
    __________________________________________________________________________________________________
    bidirectional (Bidirectional)   (None, None, 128)    66048       spatial_dropout1d[0][0]          
    __________________________________________________________________________________________________
    bidirectional_1 (Bidirectional) (None, None, 64)     31104       bidirectional[0][0]              
    __________________________________________________________________________________________________
    scaled_dot_product_attention (S (None, None, 128)    0           bidirectional[0][0]              
    __________________________________________________________________________________________________
    scaled_dot_product_attention_1  (None, None, 64)     0           bidirectional_1[0][0]            
    __________________________________________________________________________________________________
    global_max_pooling1d (GlobalMax (None, 128)          0           scaled_dot_product_attention[0][0
    __________________________________________________________________________________________________
    global_max_pooling1d_1 (GlobalM (None, 64)           0           scaled_dot_product_attention_1[0]
    __________________________________________________________________________________________________
    concatenate (Concatenate)       (None, 192)          0           global_max_pooling1d[0][0]       
                                                                     global_max_pooling1d_1[0][0]     
    __________________________________________________________________________________________________
    dense (Dense)                   (None, 11)           2123        concatenate[0][0]                
    ==================================================================================================
    Total params: 2,217,867
    Trainable params: 2,217,867
    Non-trainable params: 0
    __________________________________________________________________________________________________



```
x_text = np.array(x_text)
x_text.shape
```




    (16310,)




```
model.fit(x_text, y_label, epochs=3, batch_size=16, validation_split=0.1)
```

    Epoch 1/3
    918/918 [==============================] - 384s 418ms/step - loss: 0.2919 - accuracy: 0.2911 - val_loss: 0.2084 - val_accuracy: 0.4960
    Epoch 2/3
    918/918 [==============================] - 382s 416ms/step - loss: 0.1823 - accuracy: 0.5209 - val_loss: 0.1554 - val_accuracy: 0.5365
    Epoch 3/3
    466/918 [==============>...............] - ETA: 3:04 - loss: 0.1428 - accuracy: 0.5534


```

```
