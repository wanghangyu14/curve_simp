import time
import math

import xlrd
import xlwt
import numpy as np
from scipy.integrate import trapz


class Piont(object):

    def __init__(self, id, x, y):
        self.id = int(id)
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
    result.append(data[h:len(data)])
    return result


def complete_bends(bend1, bend2):
    b1, b2 = bend1[::-1][1:], bend2[::-1]
    for i in range(len(b1)):
        if b1[i].id >= b2[0].id:
            continue
        b2.append(b1[i])
        flag = []
        for j in range(0, len(b2) - 1):
            flag.append(is_intersected(b2[0], b2[-1], b2[j], b2[j + 1]))
        if not any(flag):
            continue
        else:
            b2.pop(-1)
            break
    if len(b2) != len(bend2):
        b2.sort(key=lambda p: p.id)
        return 0, b2
    else:
        flag = []
        m = bend2 + bend1
        for i in range(len(m) - 1):
            flag.append((is_intersected(m[0], m[-1], m[i], m[i + 1])))
        while not any(flag):
            m.pop(-1)
            m.pop(0)
            flag = []
            for i in range(len(m) - 1):
                flag.append((is_intersected(m[0], m[-1], m[i], m[i + 1])))
            if len(flag) <= 3:
                m.sort(key=lambda p: p.id)
                return 1, m
        m.sort(key=lambda p: p.id)
        return 1, m


def complete_result(bends):
    result = [bends[0]]
    for i in range(len(bends) - 1):
        flag, c = complete_bends(bends[i], bends[i + 1])
        if flag == 1:
            result.append(c)
            result.append(bends[i + 1])
        elif flag == 0:
            result.append(c)
    return result


def calculate_each_area(bends):
    min_area = float('inf')
    for bend in bends:
        area = calculate_area(bend)
        bend.append(area)
        if min_area > area:
            min_area = area
            b = bend
    return min_area, b


def divide(bends):
    res = [bends[i:i + 3] for i in range(0, len(bends), 3)]
    return res


def generalize(div_bends, threshold):
    gen_list = []
    for threebends in div_bends:
        if len(threebends) == 3:
            l = [bend[-1] > threshold for bend in threebends]
            if l == [False, True, True]:
                gen_list.append(threebends[0][0])
                gen_list += threebends[2][:-1]
            if l == [True, False, True]:
                a = set(threebends[0][:-1])
                b = set(threebends[1][:-1])
                c = set(threebends[2][:-1])
                left = list(a - (a & b))
                left.append(threebends[1][0])
                right = list(c - (c & b))
                right.insert(0, threebends[2][0])
                gen_list += left
                gen_list += right
            if l == [True, True, False]:
                gen_list += threebends[0][:-1]
                gen_list.append(threebends[2][-2])
            if l == [True, False, False]:
                if calculate_area(threebends[1][:-1]) > calculate_area(threebends[2][:-1]):
                    gen_list += threebends[0][:-1]
                    gen_list.append(threebends[2][-2])
                else:
                    a = set(threebends[0][:-1])
                    b = set(threebends[1][:-1])
                    c = set(threebends[2][:-1])
                    left = list(a - (a & b))
                    left.append(threebends[1][0])
                    right = list(c - (c & b))
                    right.insert(0, threebends[2][0])
                    gen_list += left
                    gen_list += right
            if l == [False, False, True]:
                if calculate_area(threebends[0][:-1]) < calculate_area(threebends[1][:-1]):
                    gen_list.append(threebends[0][0])
                    gen_list += threebends[2][:-1]
                else:
                    a = set(threebends[0][:-1])
                    b = set(threebends[1][:-1])
                    c = set(threebends[2][:-1])
                    left = list(a - (a & b))
                    left.append(threebends[1][0])
                    right = list(c - (c & b))
                    right.insert(0, threebends[2][0])
                    gen_list += left
                    gen_list += right
            if l == [False, True, False]:
                a = set(threebends[0][:-1])
                b = set(threebends[1][:-1])
                c = set(threebends[2][:-1])
                middle = list(b - (a & b) - (b & c))
                gen_list.append(threebends[0][0])
                gen_list += middle
                gen_list.append(threebends[2][-2])
            if l == [False, False, False]:
                areaA = calculate_area(threebends[0][:-1])
                areaB = calculate_area(threebends[1][:-1])
                areaC = calculate_area(threebends[2][:-1])
                if areaA < areaB and areaC < areaB and areaA + areaC < areaB:
                    a = set(threebends[0][:-1])
                    b = set(threebends[1][:-1])
                    c = set(threebends[2][:-1])
                    middle = list(b - (a & b) - (b & c))
                    gen_list.append(threebends[0][0])
                    gen_list += middle
                    gen_list.append(threebends[2][-2])
                else:
                    a = set(threebends[0][:-1])
                    b = set(threebends[1][:-1])
                    c = set(threebends[2][:-1])
                    left = list(a - (a & b))
                    left.append(threebends[1][0])
                    right = list(c - (c & b))
                    right.insert(0, threebends[2][0])
                    gen_list += left
                    gen_list += right
            if l == [True, True, True]:
                merge = []
                for bend in threebends:
                    merge += bend[:-1]
                gen_list += merge
        elif len(threebends) == 1:
            gen_list += threebends[0][:-1]
        elif len(threebends) == 2:
            gen_list += threebends[0][:-1]
            gen_list += threebends[1][:-1]
    gen_list = list(set(gen_list))
    gen_list.sort(key=lambda p: p.id)
    return gen_list


def del_duplicated(cbends):
    re = cbends[:]
    for i in range(len(cbends) - 1):
        if cbends[i][0].id >= cbends[i + 1][0].id and cbends[i][-1].id <= cbends[i + 1][-1].id:
            if cbends[i] in re:
                re.remove(cbends[i])
        if cbends[i][0].id <= cbends[i + 1][0].id and cbends[i][-1].id >= cbends[i + 1][-1].id:
            if cbends[i + 1] in re:
                re.remove(cbends[i + 1])
    return re


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
    thre = 0.002
    data = data_retrieve(r'D:\Wang\论文2\测试\gsp.xlsx')
    data.sort(key=lambda p: p.id)
    bends = bends_identify(data)
    cbends = complete_result(bends)
    undupl_bends = del_duplicated(cbends)
    min_area, b = calculate_each_area(undupl_bends)
    if min_area > thre:
        print("阈值过小")
        exit(0)
    div_bends = divide(undupl_bends)
    res = generalize(div_bends, thre)
    while True:
        bends = bends_identify(res)
        cbends = complete_result(bends)
        undupl_bends = del_duplicated(cbends)
        min_area, b = calculate_each_area(undupl_bends)
        print(min_area, b)
        if min_area > thre:
            break
        div_bends = divide(undupl_bends)
        res = generalize(div_bends, thre)
    print(len(res))
    # data_save(res, "res_tbg")
