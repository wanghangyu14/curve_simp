import time

import numpy as np
import math

import xlrd
import xlwt


def dataRetrieve(filename):
    '''从excel文件中获取数据'''
    data = xlrd.open_workbook(filename)  # 打开excel文件
    table = data.sheets()[0]  # 取excel文件中的第一个sheet
    nrows = table.nrows  # 取第一个sheet的行数
    _data = []
    for i in range(nrows):  # 将sheet中第i（i不等于0）行的前三个数据存入_data数组中
        if i == 0:
            continue
        _data.append(table.row_values(i)[:3])
    return _data


def triBend(Ps, Pe, Pm):
    '''计算垂距'''
    lenE = math.sqrt((Ps[1] - Pm[1]) ** 2 + (Ps[2] - Pm[2]) ** 2)  # E的长度
    lenS = math.sqrt((Pe[1] - Pm[1]) ** 2 + (Pe[2] - Pm[2]) ** 2)  # S的长度
    lenM = math.sqrt((Ps[1] - Pe[1]) ** 2 + (Ps[2] - Pe[2]) ** 2)  # M的长度
    if lenE == 0 or lenM == 0 or lenS == 0:  # 三点重合垂距为0
        return 0
    else:
        p = (lenE + lenS + lenM) / 2
        bend = math.sqrt(p * (p - lenE) * (p - lenM) * (p - lenS)) * 2 / lenM
        return bend


def calBends(arrpoints, _bends):
    '''Douglas-Peucker算法'''
    stack = []
    nump = len(arrpoints)
    stack.append([arrpoints[0][0], arrpoints[nump - 1][0]])
    while (len(stack) != 0):
        temp = stack[len(stack) - 1]
        stack.pop()
        length = int(temp[1] - temp[0] + 1)
        if length > 2:
            ps = arrpoints[int(temp[0] - 1)]
            pe = arrpoints[int(temp[1] - 1)]
            bendtemp = 0.0
            pmtemp = 0
            for i in range(1, length - 1):
                bend = triBend(ps, pe, arrpoints[int(temp[0] - 1 + i)])
                if bend > bendtemp:
                    bendtemp = bend
                    pmtemp = int(temp[0] - 1 + i + 1)
            if bendtemp != 0:
                _bends.append([pmtemp, bendtemp])
                print("   Bends num: %s" % (len(_bends)))
                if length > 3:
                    stack.append([int(temp[0]), pmtemp])
                    stack.append([pmtemp, int(temp[1])])
        print("stack num: %s " % (len(stack)))


def outputToExcel(results, fileName, mode):  # mode=0 is for bends & mode=1 is for break points
    '''保存文件'''
    print("write results into .xls ...")
    workbook = xlwt.Workbook(encoding='ascii')
    worksheet = workbook.add_sheet(fileName)
    worksheet.write(0, 0, label='ID')
    worksheet.write(0, 1, label=fileName)
    if mode == 0:
        for i in range(len(results)):
            for j in range(0, 2):
                worksheet.write(i + 1, j, results[i][j])
    elif mode == 1:
        for i in range(len(results)):
            worksheet.write(i + 1, 0, i + 1)
            worksheet.write(i + 1, 1, results[i])
    workbook.save('%s.xls' % (fileName))


def sortDescending(_list):
    list.sort(_list, key=lambda list: list[1], reverse=True)


def bandsTransfer(input):
    bendsAsInput = []
    # extract bends
    for i in range(len(input)):
        bendsAsInput.append(input[i][1])
    return bendsAsInput


def htb_inner(data):
    """
    Inner ht breaks function for recursively computing the break points.
    """
    data_length = float(len(data))
    data_mean = sum(data) / data_length
    head = [_ for _ in data if _ > data_mean]
    outp.append(data_mean)
    while len(head) > 1 and len(head) / data_length < 0.40:
        return htb_inner(head)


# variables initiallizing
arrPoints = []  # array of coastline points
bends = []  # array of bends
outp = []  # array of break points

# calculate the bends and break points
arrPoints = dataRetrieve(r"D:\王杭宇\论文2\测试\gsp.xlsx")
start = time.time()
calBends(arrPoints, bends)
sortDescending(bends)
# result of length of each point
outputToExcel(bends, "result", 0)
_bands = bandsTransfer(bends)
htb_inner(_bands)
end = time.time()
# arithmetic mean of each iteration
outputToExcel(outp, "splits", 1)
print(str(end - start))
