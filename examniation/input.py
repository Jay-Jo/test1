import sys
if __name__ == "__main__":
    arr=[]
    for i in range(2):
        line = sys.stdin.readline().strip()
        values = list(map(int, line.split()))
        arr.append((values))

    minim=arr[0][0]*arr[0][1]

    for i in range(len(arr[1])):
        step=0
        for j in range(len(arr[1])):
            if j!=i:
                aa=min(abs(abs(arr[1][i] - arr[1][j])- abs(j- i)),abs(abs(arr[1][j] + arr[0][0] - arr[1][i])-abs(len(arr[1])-i + j)+1))
                step+=aa

        if step<minim:
            minim=step
    print(minim)
