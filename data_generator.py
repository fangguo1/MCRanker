import os
from collections import defaultdict as ddict
from nltk import sent_tokenize
import copy
from os import listdir
from os.path import isfile, join
import json
import random


class data_generator:
    def __init__(self, dataset_path, verbose=False):
        self.dataset_path = dataset_path
        self.verbose = verbose

    def load_query(self, query_file):
        with open(os.path.join(self.dataset_path, query_file)) as IN:
            for line in IN:
                tmp = line.strip()

                tmp = tmp.split('\t')
                self.query = tmp[0].lower()
                self.trec_qid = tmp[1]

            # self.research_topic=line

    # We assume the input passage file gives the initial retreival ranking result; #The 1 label passage will be given label 0
    ### GF: it is a SUPER UGLY way to pass a dataset name into this funtion!!!

    def load_passages(self, passage_file, dataset):
        self.corpus_file_path = os.path.join(self.dataset_path, passage_file)
        self.all_docs = []
        self.all_docids = []
        self.all_tis = []
        self.all_abs = []
        self.docid2label = {}
        self.label2docids = ddict(list)
        self.docid2doc = ddict(str)
        self.docid2ti = ddict(str)
        self.init_ranking = []
        self.trec_docids = []
        with open(self.corpus_file_path, 'r', errors='ignore') as IN:
            # IN=IN.readlines()
            cnt = 0
            for line in IN:
                line_orig = copy.deepcopy(line)
                line = line.strip()
                if '. . .' in line:
                    line = line.split('. . .', 1)
                    ti = line[0]
                    line[1] = line[1].split('\t')
                    ab = line[1][0]

                    label = int(line[1][1])

                    trec_docid = line[1][2]

                else:
                    line = line.split('\t')
                    ti = line[0]
                    ab = ''
                    label = int(line[1])

                    trec_docid = line[2]

                ab = ab.replace('\"', "'")
                ab = ab.split(' ')[:512]
                ab = ' '.join(ab)
                ti = ti.replace('\"', "'")
                self.all_docs.append(ti + '. . .' + ab)
                self.all_tis.append(ti)
                self.all_abs.append(ab)
                # label2docid[label].append(cnt)
                self.docid2label[cnt] = label
                self.all_docids.append(cnt)
                self.label2docids[label].append(cnt)
                self.docid2doc[cnt] = ti + '. . .' + ab
                self.docid2ti[cnt] = ti
                self.init_ranking.append(cnt)

                self.trec_docids.append(trec_docid)

                cnt += 1

        if self.verbose:
            cnt += 1
            print('TOTAL: ', len(self.docid2label))
            print('VALID 2: ', len(self.label2docids[2]))
            print('VALID 1: ', len(self.label2docids[1]))
            print('VALID 0: ', len(self.label2docids[0]))


if __name__ == '__main__':
    dataset_name = sys.argv[1]