import networkx as nx
from gurobipy import quicksum, Model, GRB
import numpy as np
from time import time
from glob import glob

def LPHCARP(es):
    if isinstance(es, str):
        es = np.load(es)
    v0 = 0
    v00 = -1
    C = es['C']
    P = [i for i in range(1, es['P']+1)]
    P0 = [0] + P
    M = [i for i in range(es['M'])]

    nodes = set()
    edges = []
    S = {}
    D = {}
    Q = {}
    for n1,n2,q,p,s,d in es['req']:
        edges.append((int(n1),int(n2), {'d': d, 'r': 1, 'p': int(p), 's': s, 'q': q}))
        nodes.add(int(n1))
        nodes.add(int(n2))
        S[(int(n1),int(n2))] = s
        D[(int(n1),int(n2))] = d
        D[(v00,int(n2))] = 0
        D[(int(n1), v00)] = 0
        Q[(int(n1),int(n2))] = q
    for n1,n2,q,p,s,d in es['nonreq']:
        edges.append((int(n1),int(n2), {'d': d, 'r': 0, 'p': 0, 's': 0, 'q': 0}))
        nodes.add(int(n1))
        nodes.add(int(n2))
        D[(int(n1),int(n2))] = d
        D[(v00,int(n2))] = 0
        D[(int(n1), v00)] = 0


    A = [[(u,v) for u,v,attr in edges if attr['p']==k] for k in P0]
    AA = [a for k in P0 for a in A[k]] + [(v00, node) for node in nodes] + [(node, v00) for node in nodes]
    Ar = [a for k in P for a in A[k]]
    GG = nx.DiGraph()
    GG.add_edges_from(AA)

    model = Model('HDCARP')
    x = model.addVars([(m, a) for m in M for a in Ar], vtype=GRB.BINARY, name='x')
    y = model.addVars([(m, a, k) for m in M for a in AA for k in P], vtype=GRB.INTEGER, name='y')
    t = model.addVars([(m, k) for m in M for k in P0], vtype=GRB.CONTINUOUS, lb=0, name='t')
    r = model.addVars([(m, k) for m in M for k in P], vtype=GRB.BINARY, name='r')
    T = model.addVars([k for k in P], vtype=GRB.CONTINUOUS, lb=0, name='T')
    # print("Num vars:", len(x) + len(y) + len(t) + len(r) + len(T))


    # Constrain (2)
    for m in M:
        for k in P:
            model.addConstr(T[k] >= t[m, k] - 1e5*(1-r[m,k]))

    # Constraint (3)
    for m in M:    
        for k in P:
            model.addConstr(t[m, k] == (t[m, k-1] + 
            quicksum(S[a] * x[m, a] for a in A[k]) + 
            quicksum(D[a] * y[m, a, k] for a in AA)))

    # Constraint (4)
    for m in M:
        model.addConstr(t[m, 0] == 0)
        
    # Constraint (5)
    for m in M:
        for k in P:
            model.addConstr(quicksum(x[m, a] for a in A[k]) <= len(A[k]) * r[m, k])

    # Constraint (6)
    for m in M:
        model.addConstr(y[m, (v00, v0), 1] == 1)

    # Constraint (7)
    for m in M:
        for k in P:
            Af = set([(v00, v) for _, v in AA]) - {(v00, v00)}
            model.addConstr(quicksum(y[m, a, k] for a in Af) == 1)

    # Constraint (8)
    for m in M:
        for k in P[1:]:
            vs = set([v for i in range(1, k+1) for _, v in A[i]])
            for v in vs:
                model.addConstr(y[m, (v00, v), k] == y[m, (v, v00), k-1])

    # Constraint (9)
    for k in P:
        for a in A[k]:
            model.addConstr(quicksum(x[m, a] for m in M) == 1)


    # Constraint (10)
    for m in M:
        model.addConstr(quicksum(Q[a] * x[m, a] for a in Ar) <= C)

    # Constraint (11)
    for m in M:
        for k in P:
            for vi in {*nodes, v00}:
                arcs_out_k = [a for a in A[k] if a[0] == vi]
                arcs_in_k = [a for a in A[k] if a[1] == vi]
                f1 = quicksum(x[m, a] for a in arcs_out_k)
                f3 = quicksum(x[m, a] for a in arcs_in_k)

                arcs_out = [a for a in AA if a[0] == vi]
                arcs_in = [a for a in AA if a[1] == vi]
                f2 = quicksum(y[m, a, k] for a in arcs_out)
                f4 = quicksum(y[m, a, k] for a in arcs_in)
                model.addConstr(f1+f2==f3+f4)

    # Constraint (12)
    def subtour_elimination(model, where):
        if where == GRB.Callback.MIPSOL:  
        
            # Get the current solution
            xval = model.cbGetSolution(x)
            yval = model.cbGetSolution(y)
            for m in M:
                xm = [e for e in x if e[0] == m]
                ym = [e for e in y if e[0] == m]
                for k in P:
                    xmk = [e for e in xm if e[1] in A[k]]
                    ymk = [e for e in ym if e[2] == k]
        
                    es = [e[1] for e in xmk if xval[e] > 0.5] \
                        + [e[1] for e in ymk if yval[e] > 0.5]
                    g = GG.edge_subgraph(es)
                    for S in nx.strongly_connected_components(g):
                        S = S - {v0}
                        if len(S) >= len(g):
                            continue
                        f1 = quicksum(x[me, (u,v)] for me, (u,v) in xmk 
                            if u in S and v not in S)
                        f2 = quicksum(y[me, (u,v), ke] for me, (u,v), ke in ymk
                            if u in S and v not in S)

                        for me, (u,v) in xmk:
                            if u in S and v in S:
                                model.cbLazy(f1 + f2 >= x[me, (u,v)]) 	       
            
    model.setParam(GRB.Param.OutputFlag, 0)
    model.Params.lazyConstraints = 1
    model.setParam('TimeLimit', 600)
    model.setObjective(T[1]*1000 + T[2]*10 + T[3]*0.1, GRB.MINIMIZE)
    model.optimize(subtour_elimination)
    runtime = model.Runtime
    T = None if runtime >= 600 else np.array([T[1].x, T[2].x, T[3].x])
    model.dispose()
    return T
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="LPHCARP")
    
    # Add arguments
    parser.add_argument('--variant', type=str, default='P', help='Environment variant')
    parser.add_argument('--path', type=str, default='/usr/local/rsa/ArcRoute/data/instances', help='path to instances')
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    files = sorted(glob(args.path + '/*/*.npz'))
    for f in files:
        t1 = time()
        try:
            T = LPHCARP(f)
        except Exception as e:
            T = None
        print(f,':::', T,':::', time() - t1)
