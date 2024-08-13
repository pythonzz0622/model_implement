import numpy as np

LARGE = float('inf')

def setup_cost_matrix():
    """
    비용 행렬을 설정합니다.
    이 예제에서는 3x3 크기의 행렬을 사용하며, 
    각 요소는 작업과 작업자 간의 비용을 나타냅니다.
    """
    cost = np.array([[90, 75, 80],
                     [35, 85, 55],
                     [125, 95, 90]])
    return cost

def ccrrt_dense(n, cost, free_rows, x, y, v):
    """
    행렬에 대한 열 감소 및 감소 전환을 수행합니다.
    각 열에서 최소 비용을 찾고, 해당 행에 작업을 할당하려고 시도합니다.
    이 과정에서 할당되지 않은 행을 추적하여 free_rows에 저장합니다.
    """
    unique = np.ones(n, dtype=bool)  # 각 행의 할당이 유일한지 추적

    for i in range(n):
        x[i] = -1  # 초기화: 모든 행에 할당된 열이 없음을 의미
        v[i] = LARGE  # 각 열의 최소 비용을 LARGE로 초기화
        y[i] = 0  # 초기화: 각 열의 할당된 행을 추적

    # 각 열에 대해 최소 비용을 찾음
    for i in range(n):
        for j in range(n):
            c = cost[i][j]
            if c < v[j]:
                v[j] = c
                y[j] = i

    # 각 열에 대해 유일한 할당인지 확인하고, 유일하지 않으면 해제
    for j in range(n):
        i = y[j]
        if x[i] < 0:
            x[i] = j
        else:
            unique[i] = False
            y[j] = -1

    # 할당되지 않은 행을 free_rows에 추가
    n_free_rows = 0
    for i in range(n):
        if x[i] < 0:
            free_rows[n_free_rows] = i
            n_free_rows += 1
        elif unique[i]:
            j = x[i]
            min_val = LARGE
            for j2 in range(n):
                if j2 == j:
                    continue
                c = cost[i][j2] - v[j2]
                if c < min_val:
                    min_val = c
            v[j] -= min_val

    return n_free_rows

def carr_dense(n, cost, n_free_rows, free_rows, x, y, v):
    """
    행렬에 대한 보강 행 감소를 수행합니다.
    할당되지 않은 행(free_rows)을 사용하여 기존 할당을 개선할 수 있는지 확인합니다.
    """
    current = 0
    new_free_rows = 0
    rr_cnt = 0

    while current < n_free_rows:
        free_i = free_rows[current]
        current += 1
        j1 = 0
        v1 = cost[free_i][0] - v[0]
        j2 = -1
        v2 = LARGE

        # 최소 비용과 두 번째 최소 비용 찾기
        for j in range(1, n):
            c = cost[free_i][j] - v[j]
            if c < v2:
                if c >= v1:
                    v2 = c
                    j2 = j
                else:
                    v2 = v1
                    v1 = c
                    j2 = j1
                    j1 = j

        i0 = y[j1]
        v1_new = v[j1] - (v2 - v1)
        v1_lowers = v1_new < v[j1]

        if rr_cnt < current * n:
            if v1_lowers:
                v[j1] = v1_new
            elif i0 >= 0 and j2 >= 0:
                j1 = j2
                i0 = y[j2]
            if i0 >= 0:
                if v1_lowers:
                    current -= 1
                    free_rows[current] = i0
                else:
                    free_rows[new_free_rows] = i0
                    new_free_rows += 1
        else:
            if i0 >= 0:
                free_rows[new_free_rows] = i0
                new_free_rows += 1

        x[free_i] = j1
        y[j1] = free_i

    return new_free_rows

def find_dense(n, lo, d, cols, y):
    """
    최소 비용을 갖는 열을 찾아 SCAN 목록에 추가합니다.
    """
    hi = lo + 1
    mind = d[cols[lo]]

    for k in range(hi, n):
        j = cols[k]
        if d[j] <= mind:
            if d[j] < mind:
                hi = lo
                mind = d[j]
            cols[k], cols[hi] = cols[hi], cols[k]
            hi += 1

    return hi

def scan_dense(n, cost, lo, hi, d, cols, pred, y, v):
    """
    SCAN 목록의 임의 열에서 시작하여 TODO 열의 비용을 감소시키려고 시도합니다.
    """
    while lo != hi:
        j = cols[lo]
        lo += 1
        i = y[j]
        mind = d[j]
        h = cost[i][j] - v[j] - mind

        for k in range(hi, n):
            j = cols[k]
            cred_ij = cost[i][j] - v[j] - h
            if cred_ij < d[j]:
                d[j] = cred_ij
                pred[j] = i
                if cred_ij == mind:
                    if y[j] < 0:
                        return j
                    cols[k], cols[hi] = cols[hi], cols[k]
                    hi += 1

    return -1

def find_path_dense(n, cost, start_i, y, v, pred):
    """
    수정된 Dijkstra 최단 경로 알고리즘을 사용하여 
    작업에서 작업자로 가는 최적의 경로를 찾습니다.
    """
    cols = np.arange(n)
    d = np.zeros(n)

    for i in range(n):
        pred[i] = start_i
        d[i] = cost[start_i][i] - v[i]

    lo = 0
    hi = 0
    final_j = -1

    while final_j == -1:
        if lo == hi:
            hi = find_dense(n, lo, d, cols, y)
            for k in range(lo, hi):
                j = cols[k]
                if y[j] < 0:
                    final_j = j

        if final_j == -1:
            final_j = scan_dense(n, cost, lo, hi, d, cols, pred, y, v)

    mind = d[cols[lo]]
    for k in range(hi):
        j = cols[k]
        v[j] += d[j] - mind

    return final_j

def ca_dense(n, cost, n_free_rows, free_rows, x, y, v):
    """
    Augment 함수는 각 자유 행을 사용하여 현재의 할당을 개선하는 데 사용됩니다.
    """
    pred = np.zeros(n, dtype=int)

    for pfree_i in free_rows[:n_free_rows]:
        j = find_path_dense(n, cost, pfree_i, y, v, pred)
        i = -1
        while i != pfree_i:
            i = pred[j]
            y[j] = i
            j = x[i]
            x[i] = j

def lapjv_internal(n, cost):
    """
    LAPJV 알고리즘을 수행하여 작업을 작업자에게 할당합니다.
    이 함수는 전체 알고리즘의 진입점입니다.
    """
    x = np.full(n, -1, dtype=int)
    y = np.full(n, -1, dtype=int)
    free_rows = np.zeros(n, dtype=int)
    v = np.zeros(n)

    n_free_rows = ccrrt_dense(n, cost, free_rows, x, y, v)
    i = 0
    while n_free_rows > 0 and i < 2:
        n_free_rows = carr_dense(n, cost, n_free_rows, free_rows, x, y, v)
        i += 1

    if n_free_rows > 0:
        ca_dense(n, cost, n_free_rows, free_rows, x, y, v)

    return x, y

def main():
    """
    메인 함수는 전체 프로세스를 실행하며, 
    비용 행렬을 설정하고, LAPJV 알고리즘을 실행하여 
    작업-작업자 할당 결과를 출력합니다.
    """
    N = 3  # 행렬의 크기
    cost = setup_cost_matrix()

    x, y = lapjv_internal(N, cost)

    # 결과 출력
    print("작업 할당 결과 (작업:할당된 작업자):")
    for i in range(N):
        print(f"작업 {i}: 작업자 {x[i]}")

if __name__ == "__main__":
    main()
