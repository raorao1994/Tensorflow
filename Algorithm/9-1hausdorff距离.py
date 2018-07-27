"""
    目标:
    ~~~~~~~~~
    实现 Hausdorff 距离.该距离的定义如下:
    A = {a1, a2, ..., an}, B = {b1, b2, ..., bm}
    两者的 Hausdorff 距离为:
    H(A, B) = max{h(A, B), h(B, A)},
    其中
    h(A, B)的定义如下,无法用数学公式来展现,我这里用语言来描述一下得了.
    对于A中的每一个点a,首先在B中找到一个距离a最近的点b,得到a,b它们之间的距离dist(a, b),这里的dist定义如下:
    dist(a, b) = sqrt((xa - xb)^2, (ya - yb)^2),就是欧氏距离啦.
    通过计算得到n个这样的距离,找出其中最大的一个,作为h(A, B),h(B, A)同理.
"""
from math import sqrt


def euclidean_metric(pa, pb):
    '''''
    计算两个点的欧式距离.
    关于输入的pa以及pb,它们应该是一个list或者tuple,长度应该相等.
    '''
    return sqrt((pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2)


def one_way_hausdorff_distance(sa, sb):
    '''''计算单向hausdorff距离'''
    distance = 0.0
    for pa in sa:  # 对于a集合中的每一个点,要求出这个点到
        shortest = 9999999
        for pb in sb:
            dis = euclidean_metric(pa, pb)
            if dis < shortest:
                shortest = dis
        if shortest > distance:
            distance = shortest
    return distance


def hausdorff_distance(sa, sb):
    '''''
    计算两个集合中的点的 hausdorff 距离.
    关于输入的sa以及sb,两者应该是点的list或者tuple,而点的定义参照euclidean_metric.
    '''
    dis_a = one_way_hausdorff_distance(sa, sb)
    dis_b = one_way_hausdorff_distance(sb, sa)
    return dis_a if dis_a > dis_b else dis_b


if __name__ == '__main__':
    assert (euclidean_metric((1, 1), (1, 6)) == 5)
    A = [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)]
    B = [(0, 3), (1, 3), (2, 3), (3, 3), (3, 2), (4, 2)]
    C = [(4, 2), (4, 1), (4, 0)]
    D = [(0, 2), (1, 2), (2, 2), (2, 3), (2, 4)]
    result=one_way_hausdorff_distance(A, B)
    print(result);
    print("完成");