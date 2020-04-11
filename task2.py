from pyspark import SparkContext,SparkConf
import itertools
import os
from queue import Queue,LifoQueue
from collections import defaultdict
import sys
from utils import *
import time

def main(th,inputFile,btFile,coFile):
    scConf = SparkConf() \
        .setAppName('hw4') \
        .setMaster('local[1]')
    sc = SparkContext(conf=scConf)
    sc.setLogLevel("WARN")
    lines = sc.textFile(inputFile)

    user_reviewed = lines.map(lambda x: x.split(",")).map(lambda x: (x[0], x[1])).filter(
            lambda x: x[0] != 'user_id').groupByKey().cache()

    user_review_dict = user_reviewed.collectAsMap()

    def valid_pair(x):
        user_1 = x[0]
        user_2 = x[1]
        corated = len(set(user_review_dict[user_1]).intersection(set(user_review_dict[user_2])))
        if corated >= th:
            return True
        else:
            return False

    valid_pairs = user_reviewed.map(lambda x: (1, x[0])).groupByKey().flatMap(
        lambda x: itertools.combinations(sorted(x[1]), 2)).filter(valid_pair)

    edges=valid_pairs.flatMap(lambda x:[x,(x[1],x[0])])
    graph=edges.groupByKey().map(lambda x:(x[0],set(x[1]))).collectAsMap()
    vertices=list(graph.keys())

    betweenness = sc.parallelize([(graph,x) for x in vertices]).map(lambda x:bfs(x[0],x[1])).map(lambda x:credit(*x)).flatMap(lambda x:list(x.items())).reduceByKey(lambda x,y:x+y).map(lambda x: (x[0],x[1]/2)).sortBy(lambda x: (-x[1],x[0][0]))

    edge_betweeness = betweenness.collect()
    with open(btFile, "w") as f:
        for i in edge_betweeness:
            string_to_write="{}, {}".format(i[0],i[1])
            f.write(string_to_write+"\n")

    result_communities = com_max_mod(graph,vertices,edge_betweeness,sc)

    result=sorted([sorted(x) for x in result_communities],key=lambda x:(len(x),x[0]))


    with open(coFile,"w") as f:
        for i in result:
            f.write(", ".join(list(map(lambda x: "'{}'".format(x), i))) + "\n")

if __name__ == "__main__":
    start_time = time.time()
    threshold = int(sys.argv[1])
    input_file = sys.argv[2]
    betweenness_output_file_path = sys.argv[3]
    community_output_file_path=sys.argv[4]

    main(threshold, input_file, betweenness_output_file_path,community_output_file_path)
    print("Total Running Time:  {}".format(time.time() - start_time))