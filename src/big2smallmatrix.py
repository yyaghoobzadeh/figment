# for debugging: upto = 10000 will only look at the first 10000 lines
# This is a script to summarize entity scores aggregated from all of its context scores
# it should be used on the outputs of test_cm.py script to get gm-like matrices. 
import string, collections, sys
from threading import Thread

from myutils import *
import logging
from _collections import defaultdict
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger('big2small.py')


upto = -1

config = loadConfig(sys.argv[1])

logger.info(config)

numtype = int(config['numtype'])
typefilename = config['typefile']
norm = str_to_bool(config['norm'])
logger.info('** norm is %s', norm)

def big2small(example, big, small):   
    logger.info('Loading the mapping from lines to entities')
    e2i = {}
    e2freq = defaultdict(lambda: 0)
    fbfile = open(example)
    lineno = -1
    fblist = []
    skipped = 0
    for myline in fbfile:
        lineno += 1
        if upto>=0 and lineno>upto: break
        myparts = string.split(myline)
        mytype = myparts[2]
        if '/m/' in myparts[2]:
            subparts = string.split(myparts[2],'(')
            assert len(subparts)==2
            subsubparts = string.split(subparts[0],'/')
            assert len(subsubparts)==3
            mytype = subsubparts[2]
        if mytype not in t2i:
            logger.error('subsub',subsubparts)
            assert 0
        myne = myparts[1].split('/')[2]
        e2freq[myne] += 1
        fblist.append((myne,t2i[mytype]))
        if myne in e2i:
            assert e2i[myne]==t2i[mytype]
        else:
            e2i[myne] = t2i[mytype]
        #print myparts[1]
    logger.info('..loaded')
    logger.info('Loading the big file...')
    biglines = getlines(big)
    logger.info('... loaded')
    logger.info('Loading the big matrix...')
    lineno = -1
    big = []
    for i in range(numtype):
        big.append(collections.defaultdict(lambda: []))
    for myline in biglines:
#         print str(lineno)
        lineno += 1
        if upto>=0 and lineno>upto: break
        if fblist[lineno]==None:
            continue
        myparts = string.split(myline)
        scores = [float(myparts[i]) for i in range(numtype)]
        myne = fblist[lineno][0]
        if norm:
            scores = mynormalize(scores)
        #print len(myparts)
        assert len(myparts)==numtype
        for i in range(numtype):
            big[i][myne].append(scores[i])
            
    logger.info('aggregating scores for entities')
    type2entscores = []
    for i in range(numtype):
        type2entscores.append(collections.defaultdict(lambda: []))
            
    for i in range(numtype):
        for mye in big[i]:
            big[i][mye].sort()
            big[i][mye].reverse()
            for mypercentile in percentiles:
                summaryScore = getpercentile(big[i][mye],mypercentile)
                type2entscores[i][mye].append(summaryScore)
            for myabsolute in absolutes:
                summaryscore = getabsolute(big[i][mye],myabsolute)
                type2entscores[i][mye].append(summaryscore)    
            
    entitythentypefile = open(small, 'w')
    for mye in type2entscores[0]:
        entitythentypefile.write(mye)
        for i in range(numtype):
            entitythentypefile.write(' ')
            scores = type2entscores[i][mye]
            for s in scores:
                entitythentypefile.write(str(s) + ',')
        entitythentypefile.write(' ' + str(e2freq[mye])) #the entity freq in the sampled file
        entitythentypefile.write('\n')
    
    entitythentypefile.close()
    
    logger.info('skipped: %d', skipped)
    logger.info('small matrix saved in: %s', small)

if __name__ == '__main__':
    (t2i,t2f) = fillt2i(typefilename, numtype)
    devsampledfile = config['sampled_dir'] + config['devexample_theta']
    testsampledfile = config['sampled_dir'] + config['testexample_theta']
    big2small(devsampledfile, config['devscores'], config['matrixdev'])
    big2small(testsampledfile, config['testscores'], config['matrixtest'])

