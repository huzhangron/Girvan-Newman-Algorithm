from queue import Queue,LifoQueue
from collections import defaultdict
import copy
import itertools


def com_max_mod(g, v, bet,sc1):
    max_mod = -1
    max_communities = []
    updated_graph = copy.deepcopy(g)
    num_edges = len(bet)
    updated_bet = copy.deepcopy(bet)
    while len(updated_bet) > 0:
        communities = find_communities(updated_graph, v)
        cur_modularity = get_modularity(g, communities, num_edges)
        edge_to_remove = []
        max_edge_betweeness = updated_bet.pop(0)[0]
        edge_to_remove.append(max_edge_betweeness)
        for (node1, node2) in edge_to_remove:
            updated_graph[node1].remove(node2)

            updated_graph[node2].remove(node1)

        updated_bet = sc1.parallelize([(updated_graph, x) for x in v]).map(lambda x: bfs_version_2(x[0], x[1])).map(
            lambda x: credit_version_2(*x)).flatMap(lambda x: list(x.items())).reduceByKey(lambda x, y: x + y).map(
            lambda x: (x[0], x[1] / 2)).sortBy(lambda x: -x[1]).collect()
        if cur_modularity >= max_mod:
            max_mod = cur_modularity
            max_communities = communities

    return max_communities


def get_modularity(g, c, m):
    modularity = 0
    for i in c:
        pairs = itertools.combinations(i, 2)
        for j in pairs:
            node1 = j[0]
            node2 = j[1]
            if node2 in g[node1]:
                modularity = modularity + (1 - len(g[node1]) * len(g[node2]) / (2 * m))
            else:
                modularity = modularity - (len(g[node1]) * len(g[node2]) / (2 * m))
    modularity = modularity / (2 * m)
    return modularity


def find_communities(g, v):
    vertices_remain = set(v)
    communities = []
    while len(vertices_remain) > 0:
        community = get_community(g, list(vertices_remain)[0])
        vertices_remain = vertices_remain.difference(community)
        communities.append(community)
    return communities


def get_community(g, node):
    wait_to_be_visited = Queue()
    wait_to_be_visited.put(node)
    vistied = set()
    vistied.add(node)
    while not wait_to_be_visited.empty():
        cur_node = wait_to_be_visited.get()
        next_nodes = g[cur_node]
        for i in next_nodes:
            if i not in vistied:
                wait_to_be_visited.put(i)
                vistied.add(i)
    return vistied


def bfs_version_2(g, v):
    stack = []
    wait_to_be_visited = []
    num_shortest_path = defaultdict(int)
    bottom_up_tree = defaultdict(list)
    vertex_level = defaultdict(int)
    for i in g.keys():
        vertex_level[i] = -1
    wait_to_be_visited.append(v)
    vertex_level[v] = 0
    num_shortest_path[v] = 1
    while len(wait_to_be_visited) > 0:
        cur_node = wait_to_be_visited.pop(0)
        stack.append(cur_node)
        next_nodes = g[cur_node]
        for i in next_nodes:
            if vertex_level[i] == -1:
                vertex_level[i] = vertex_level[cur_node] + 1
                num_shortest_path[i] += num_shortest_path[cur_node]
                bottom_up_tree[i].append(cur_node)
                wait_to_be_visited.append(i)
            elif vertex_level[i] == vertex_level[cur_node] + 1:
                num_shortest_path[i] += num_shortest_path[cur_node]
                bottom_up_tree[i].append(cur_node)
    return (num_shortest_path, bottom_up_tree, stack)


def credit_version_2(nsp, but, st):
    edge_credits = defaultdict(float)
    vertex_credicts = defaultdict(float)
    for i in nsp.keys():
        vertex_credicts[i] = 1
    while len(st) > 0:
        cur_node = st.pop()
        for i in but[cur_node]:
            cur_credit = vertex_credicts[cur_node] * nsp[i] / nsp[cur_node]
            vertex_credicts[i] += cur_credit
            edge_credits[tuple(sorted((cur_node, i)))] += cur_credit
    return edge_credits


def bfs(g, v):
    stack = LifoQueue()
    wait_to_be_visited = Queue()
    num_shortest_path = defaultdict(int)
    bottom_up_tree = defaultdict(list)
    vertex_level = defaultdict(int)
    for i in g.keys():
        vertex_level[i] = -1
    wait_to_be_visited.put(v)
    vertex_level[v] = 0
    num_shortest_path[v] = 1
    while not wait_to_be_visited.empty():
        cur_node = wait_to_be_visited.get()
        stack.put(cur_node)
        next_nodes = g[cur_node]
        for i in next_nodes:
            if vertex_level[i] == -1:
                vertex_level[i] = vertex_level[cur_node] + 1
                num_shortest_path[i] += num_shortest_path[cur_node]
                bottom_up_tree[i].append(cur_node)
                wait_to_be_visited.put(i)
            elif vertex_level[i] == vertex_level[cur_node] + 1:
                num_shortest_path[i] += num_shortest_path[cur_node]
                bottom_up_tree[i].append(cur_node)
    return (num_shortest_path, bottom_up_tree, stack)


def credit(nsp, but, st):
    edge_credits = defaultdict(float)
    vertex_credicts = defaultdict(float)
    for i in nsp.keys():
        vertex_credicts[i] = 1
    while not st.empty():
        cur_node = st.get()
        for i in but[cur_node]:
            cur_credit = vertex_credicts[cur_node] * nsp[i] / nsp[cur_node]
            vertex_credicts[i] += cur_credit
            edge_credits[tuple(sorted((cur_node, i)))] += cur_credit
    return edge_credits

