import numpy as np
import json
from tqdm import tqdm
def process(temp):
    temp = temp.split('(')[1]
    temp = temp.split(')')[0]
    _id = temp.split(',')[0]
    _start = temp.split(',')[1]
    _len = temp.split(',')[2]
   # print(_id)
    return int(_id),int(_start),int(_len)

def genDict():
    dic = {}
    file_object = open("entityInfo.dat",'r')
    lines = file_object.readlines()
    print("start generating dictionary")
    for line in tqdm(lines):
        _id, _start, _len = process(line)
        temp = []
        temp.append(_start)
        temp.append(_len)

        dic[_id] = tuple(temp)
    return dic
    
def get_max(dic,entities):
    ans = 1
    for entity in entities:
        try:
            ans = max(ans, dic[entity][1])
        except:
            ans = ans
    return ans

def get_new_facts(facts, entity_vocab, relation_vocab):
    new_facts = []
    print("start generating new facts list")
    for mem in tqdm(facts):
        temp = []
        temp.append(entity_vocab[mem['e1']] if mem['e1'] in entity_vocab else entity_vocab['UNK'])
        temp.append(relation_vocab[mem['r']] if mem['r'] in relation_vocab else relation_vocab['UNK'])
        temp.append(entity_vocab[mem['e2']] if mem['e2'] in entity_vocab else entity_vocab['UNK'])
        new_facts.append(temp)
    return new_facts


def entity2facts(facts, dic, entity_vocab, relation_vocab, max_mem, entities):
    max_num = get_max(dic,entities)

    answer = np.ones([len(entities), max_num*len(entities), 3])
    
    answer[:, :, 0].fill(entity_vocab['DUMMY_MEM'])
    answer[:, :, 1].fill(relation_vocab['DUMMY_MEM'])
    answer[:, :, 2].fill(entity_vocab['DUMMY_MEM'])

    mem_counter = 0
    for counter, entity in enumerate(entities):
        try:
            tu = dic[entity]
        except:
            # print("%d does not exists"%entity)
            continue
        for mem_index in xrange(tu[0], tu[0]+tu[1]):
            # print("memory of %d"%mem_counter)
            mem = facts[mem_index]
            #e1_int = entity_vocab[mem['e1']] if mem['e1'] in entity_vocab else entity_vocab['UNK']
            #e2_int = entity_vocab[mem['e2']] if mem['e2'] in entity_vocab else entity_vocab['UNK']
            #r_int = relation_vocab[mem['r']] if mem['r'] in relation_vocab else relation_vocab['UNK']
            e1_int = mem[0]
            r_int = mem[1]
            e2_int = mem[2]

            answer[counter][mem_counter][0] = e1_int
            answer[counter][mem_counter][1] = r_int
            answer[counter][mem_counter][2] = e2_int

            mem_counter += 1
            if (mem_counter == max_mem):
                print("over!")
                break
    return answer

def read_kb_facts():
    facts = []
    #facts_list = defaultdict(list)
    print('Reading kb file at {}'.format("../kb/freebase.spades.txt"))
    with open("../kb/freebase.spades.txt") as fb:
        for counter, line in tqdm(enumerate(fb)):
            line = line.strip()
            line = line[1:-1]
            e1, r1, r2, e2 = [a.strip('\'') for a in [x.strip() for x in line.split(',')]]
            r = r1 + '_' + r2
            facts.append({'e1': e1, 'r': r, 'e2': e2})
           # facts_list[e1].append(counter)  # just store the fact counter instead of the fact
    return facts

def prepare():
    global entity_object, relation_object, entity_voc, relation_voc, facts, new_facts
    entity_object = open("../vocab/entity_vocab.json")
    relation_object = open("../vocab/relation_vocab.json")
    entity_voc = json.loads(entity_object.read())
    relation_voc = json.loads(relation_object.read())

    facts = read_kb_facts()
    new_facts = get_new_facts(facts, entity_voc, relation_voc)

dic = genDict()
prepare()

def get_memory(entities):
    mem = entity2facts(new_facts, dic, entity_voc, relation_voc, 2147483647, entities)
    return mem

if __name__ == "__main__":
   # file_object = open("entityInfo.txt",'r')
   # lines = file_object.readlines()
   # a,b,c = process(lines[0])
   # print(a)
   # print(b)
   # print(c)
   
   mem = get_memory([824729,824730,407779])
   
   print(mem)

   #print(dic[824729])
   #print(dic[407784])
   


