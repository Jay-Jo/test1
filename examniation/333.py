import sys
if __name__ == "__main__":
    # 读取第一行的n
    line = sys.stdin.readline().strip()
    aa=list(map(int, line.split()))
    mons=aa[0]
    path=aa[1]
    arr=[]
    for i in range(path):
        # 读取每一行
        line = sys.stdin.readline().strip()
        # 把每一行的数字分隔后转化成int列表
        values = list(map(int, line.split()))
        arr.append(values )
    print(arr)
    mon=[i for i in range(1,mons+1)]
    # print(mon)
    for li in arr:
        mon.remove(li[0])
        while (li[1] in li[])


