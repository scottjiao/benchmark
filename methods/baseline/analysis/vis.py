from matplotlib import pyplot as plt
import numpy as np



datasets=["pubmed","DBLP","ACM","IMDB"]
acc_datas=[[0.5  , 0.467 ,0.333 ,0.381 ,0.   , 0.667, 0.5   ,  0  , 0 ,0.4  ],
[0.621, 1. ,   0.714 ,0.607 ,0.471, 0.317, 0.243, 0.263, 0.43,  0.57 ],
[1. ,   0.984, 0.994 ,1.  ,  0.333, 0.25 , 0.109 ,0.044, 0.607 ,0.719],
[0.858, 0.99 , 0.976, 0.769, 0.286, 0.218, 0.103, 0.128, 0.296 ,0.428],
]

bins_data=[
[132. , 45. , 12.  ,42. ,  2. ,  3. ,  2. ,  0.,   0.,  10.],
[ 533.  , 12. ,  35. ,  61. ,  17. , 262.,  136. ,  57.,   79., 1399.],
[  55. , 187. , 331. ,  24. ,   3. ,  16.,  101. , 810., 1381.,  114.],
[ 232. , 100. ,  41. ,  13. ,   7. ,  55. , 136.,  296.,  720., 1150.],
]

for i in range(4):
    x=[j/10 for j in  range(len(acc_datas[i]))]

    left_data = acc_datas[i]
    total=sum(bins_data[i])
    right_data =[ j/total for j in  bins_data[i]]

    fig, ax_left = plt.subplots()
    ax_right = ax_left.twinx()

    ax_left.plot([j+0.05 for j in x],left_data, color='black')
    ax_left.tick_params(axis='y', labelcolor='black')
    ax_left.set_ylabel("accs", color='black')
    ax_right.plot([j+0.05 for j in x],right_data, color='red')
    ax_right.tick_params(axis='y', labelcolor='red')
    ax_right.set_ylabel("percentages", color='red')

    plt.title(datasets[i])
    #plt.plot(,acc_datas[i])
    #plt.ylim([0,1])
    #plt.legend()
    plt.xticks(x+[1],x+[1])
    plt.savefig(datasets[i]+".png")
    plt.cla()


