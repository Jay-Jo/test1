import sys
if __name__ == "__main__":
    def  varr(li):
        aa=sum(li)/3
        res=0
        for i in li:
            res+=(i-aa)**2
        return res/3
    arr=[]
    for i in range(2):
        # 读取每一行
        line = sys.stdin.readline().strip()
        # 把每一行的数字分隔后转化成int列表
        values = list(map(int, line.split()))
        arr.append((values))
    l=len(arr[1])
    nums=(arr[1])
    mini=2**7
    for i in range(l-2):
        for j in range(i+1,l-1):
            for k in range(i+2,l):
                li=[nums[i],nums[j],nums[k]]
                tmp=varr(li)
                if tmp<mini:
                    mini=tmp
    print('%.2f'% (mini))
