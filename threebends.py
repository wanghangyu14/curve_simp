import math
import time

import xlrd
import xlwt
import numpy as np
from scipy.integrate import trapz

from evaluate import calcu_len, calcu_ang


class Piont(object):

    def __init__(self, id, x, y):
        self.id = id
        self.x = x
        self.y = y

    def __repr__(self):
        return repr((self.id, self.x, self.y))


class Vector(object):

    def __init__(self, start_point, end_point):
        self.start, self.end = start_point, end_point
        self.x = end_point.x - start_point.x
        self.y = end_point.y - start_point.y


def data_retrieve(fileName):
    '''从excel中读取数据'''
    file = xlrd.open_workbook(fileName)
    table = file.sheet_by_index(0)
    rows = table.nrows
    data = []
    for i in range(1, rows):
        record = table.row_values(i)
        data.append(Piont(record[0], record[1], record[2]))
    return data


def vector_product(vectorA, vectorB):
    '''计算 x_1 * y_2 - x_2 * y_1（向量叉乘）'''
    return vectorA.x * vectorB.y - vectorB.x * vectorA.y


def is_equal(p, q):
    '''判断两点是否相等'''
    if p.x == q.x and p.y == q.y:
        return True
    return False


def is_collinear(a, b, c, d):
    '''判断四点是否共线'''
    ZERO = 1e-9
    if a.x - b.x == 0 and c.x - d.x == 0:
        if a.x == c.x:
            return True
    elif a.x - b.x == 0 and c.x - d.x != 0:
        return False
    elif a.x - b.x != 0 and c.x - d.x == 0:
        return False
    else:
        k1 = (b.y - a.y) / (b.x - a.x)
        k2 = (d.y - c.y) / (d.x - c.x)
        if abs(k1 - k2) <= ZERO:
            return True
    return False


def is_intersected(A, B, C, D):
    '''A, B, C, D 为 Point 类型，判断线段AB,CD是否相交'''
    ZERO = 1e-9
    if is_equal(A, C) or is_equal(B, C) or is_equal(A, D) or is_equal(B, D):
        return False
    if is_collinear(A, B, C, D):
        return False
    AC = Vector(A, C)
    AD = Vector(A, D)
    AB = Vector(A, B)
    CD = Vector(C, D)
    CA = Vector(C, A)
    CB = Vector(C, B)

    return (vector_product(AB, AC) * vector_product(AB, AD) <= ZERO) \
           and (vector_product(CD, CA) * vector_product(CD, CB) <= ZERO)


def axis_trans(a, b, c):
    '''以直线ab为x轴，a为原点对c做坐标的转换'''
    d = Piont(0, c.x - a.x, c.y - a.y)
    if a.x - b.x != 0:
        k = (a.y - b.y) / (a.x - b.x)
        theta = math.atan(k)
    else:
        theta = math.pi / 2
    transMatrix = np.array([[math.cos(theta), math.sin(theta)],
                            [-math.sin(theta), math.cos(theta)]])
    originMatrix = np.array([d.x, d.y])
    arr = np.matmul(transMatrix, originMatrix)
    d.x, d.y = arr[0], arr[1]
    return d


def calculate_area(bend):
    '''计算轴线和曲线段围成的面积'''
    yaxis = []
    xaxis = []
    for i in range(len(bend)):
        p = axis_trans(bend[0], bend[-1], bend[i])
        yaxis.append(p.y)
        xaxis.append(p.x)
    return abs(trapz(yaxis, xaxis))


def bends_identify(data):
    '''划分弯曲段'''
    result = []
    h = 0
    t = h + 3
    maxc = 0
    while t < len(data):
        count = 0
        for j in range(h, t):
            if not is_intersected(data[h], data[t], data[j], data[j + 1]):
                count += 1
                if j + 1 == t:
                    t += 1
                    if count > maxc:
                        maxc = count
                    break
            else:
                if count != 1:
                    result.append(data[h:h + maxc + 1])
                    h = h + maxc
                    t = h + 3
                    maxc = 0
                elif count == 1:
                    result.append(data[h:t])
                    h = t - 1
                    t = h + 3
                break
    result.append(data[h:len(data)])
    return result


def cal_len(bend, mode=-1):
    length = math.sqrt((bend[0].x - bend[mode].x) ** 2 + (bend[0].y - bend[mode].y) ** 2)
    bend.append(length)
    return length


def divide(bends):
    # print('total:%s' % len(bends))
    res = [bends[i:i + 3] for i in range(0, len(bends), 3)]
    # print('three bends:%s' % len(res))
    return res


def generalize(div_bends, threshold):
    for threebends in div_bends:
        if len(threebends) == 3:
            l = [bend[-1] > threshold for bend in threebends]
            if l == [False, True, True]:
                threebends[0] = [threebends[0][0], threebends[0][-2]]
                threebends[1] = threebends[1][:-1]
                threebends[2] = threebends[2][:-1]
            if l == [True, False, True]:
                threebends[0] = threebends[0][:-1]
                threebends[1] = [threebends[1][0], threebends[1][-2]]
                threebends[2] = threebends[2][:-1]
            if l == [True, True, False]:
                threebends[0] = threebends[0][:-1]
                threebends[1] = threebends[1][:-1]
                threebends[2] = [threebends[2][0], threebends[2][-2]]
            if l == [True, False, False]:
                if calculate_area(threebends[1][:-1]) > calculate_area(threebends[2][:-1]):
                    threebends[0] = threebends[0][:-1]
                    threebends[1] = threebends[1][:-1]
                    threebends[2] = [threebends[2][0], threebends[2][-2]]
                else:
                    threebends[0] = threebends[0][:-1]
                    threebends[1] = [threebends[1][0], threebends[1][-2]]
                    threebends[2] = threebends[2][:-1]
            if l == [False, False, True]:
                if calculate_area(threebends[0][:-1]) < calculate_area(threebends[1][:-1]):
                    threebends[0] = [threebends[0][0], threebends[0][-2]]
                    threebends[1] = threebends[1][:-1]
                    threebends[2] = threebends[2][:-1]
                else:
                    threebends[0] = threebends[0][:-1]
                    threebends[1] = [threebends[1][0], threebends[1][-2]]
                    threebends[2] = threebends[2][:-1]
            if l == [False, True, False]:
                threebends[0] = [threebends[0][0], threebends[0][-2]]
                threebends[1] = threebends[1][:-1]
                threebends[2] = [threebends[2][0], threebends[2][-2]]
            if l == [False, False, False]:
                areaA = calculate_area(threebends[0][:-1])
                areaB = calculate_area(threebends[1][:-1])
                areaC = calculate_area(threebends[2][:-1])
                if areaA < areaB and areaC < areaB and areaA + areaC < areaB:
                    threebends[0] = [threebends[0][0], threebends[0][-2]]
                    threebends[1] = threebends[1][:-1]
                    threebends[2] = [threebends[2][0], threebends[2][-2]]
                else:
                    threebends[0] = threebends[0][:-1]
                    threebends[1] = [threebends[1][0], threebends[1][-2]]
                    threebends[2] = threebends[2][:-1]
            else:
                threebends[0] = threebends[0][:-1]
                threebends[1] = threebends[1][:-1]
                threebends[2] = threebends[2][:-1]
        elif len(threebends) == 1:
            threebends[0] = threebends[0][:-1]
        elif len(threebends) == 2:
            threebends[0] = threebends[0][:-1]
            threebends[1] = threebends[1][:-1]


def data_save(results, fileName):
    '''保存结果到excel'''
    print("write results into .xls ...")
    workbook = xlwt.Workbook(encoding='ascii')
    worksheet = workbook.add_sheet(fileName)
    worksheet.write(0, 0, label='ID')
    worksheet.write(0, 1, label='x')
    worksheet.write(0, 2, label='y')
    for i in range(len(results)):
        worksheet.write(i + 1, 0, results[i].id)
        worksheet.write(i + 1, 1, results[i].x)
        worksheet.write(i + 1, 2, results[i].y)
    workbook.save('%s.xls' % fileName)


if __name__ == '__main__':
    data = data_retrieve(r'D:\王杭宇\论文2\测试\gsp.xlsx')
    data.sort(key=lambda p: p.id)
    threshold = 0.13827408639975108
    start = time.time()
    bends = bends_identify(data)
    div_bends = divide(bends)
    for threebends in div_bends:
        for bend in threebends:
            cal_len(bend)
    generalize(div_bends, threshold)
    res = []
    for three_bends in div_bends:
        for bend in three_bends:
            for p in bend:
                if p not in res:
                    res.append(p)
    res.sort(key=lambda p: p.id)
    while True:
        bends = bends_identify(res)
        div_bends = divide(bends)
        min_len = 2147483647
        for threebends in div_bends:
            for bend in threebends:
                cal_len(bend)
                if bend[-1] < min_len:
                    min_len = bend[-1]
        # print('minlength:%s' % str(min_len))
        if min_len >= threshold:
            break
        else:
            generalize(div_bends, threshold)
            res = []
            for three_bends in div_bends:
                for bend in three_bends:
                    for p in bend:
                        if p not in res:
                            res.append(p)
            res.sort(key=lambda p: p.id)
    end = time.time()
    print('time:%s' % str(end - start))
    data_save(res, 'threebends')
