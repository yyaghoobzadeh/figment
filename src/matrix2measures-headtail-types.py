# for debugging: upto = 10000 will only look at the first 10000 lines

#finding thresholds from summarized scores for each entity2type
Etestfreq = [5, 10, 30, 100]
from myutils import * 
import string,collections, sys
upto = -1
config = loadConfig(sys.argv[1])
print 'loading cofing ',config
numtype = int (config['numtype'])
Etestfile = config['Etest'] #/nfs/datm/cluewebwork/nlu/experiments/entity-categorization/allTypes/sbj_datasets/17nov_conf_min0.5/cis_datasets/custom807/Etest
Edevfile = config['Edev'] #'/nfs/datm/cluewebwork/nlu/experiments/entity-categorization/allTypes/sbj_datasets/17nov_conf_min0.5/cis_datasets/custom807/Edev'

matrixdev = config['matrixdev'] #'enttypescores_dev' + str(numtype)
matrixtest = config['matrixtest']#'enttypescores_test' + str(numtype)
typefilename = config['typefile'] #/nfs/datm/cluewebwork/nlu/experiments/entity-categorization/allTypes/experiments/807types/cis/context-datasets/rndTypes_trndevcontextFreq'
donorm = str_to_bool(config['norm'])
onlynt = str_to_bool(config['onlynt'])
print '** norm is ', donorm
topk = 50

# Thresholds are found and put into typethresholdMatrix--- Now we should apply thresholds and calc performance

def calcPrintMeasures(myt2i):  
    overalPrec = [0.0 for x in range(numScorePerType)]
    typeFscoreMatrix = [[0 for x in range(numScorePerType)] for x in range(numtype)] 
    overalfscorestype = [0.0 for x in range(numScorePerType)]
    goods = [0.0 for ind in range(numScorePerType)]; bads = [0.0 for ind in range(numScorePerType)]; totals = [0.0 for ind in range(numScorePerType)];
    for i in myt2i.values():
        thelist_test = []
        for j in range(numScorePerType):
            thelist_test.append([])
            for mye in bigtest[i]:
                gold = 0
                if onlynt == False and i in e2i_test[mye]:
                    gold = 1
                elif onlynt == True and i == e2i_test[mye]:
                    gold = 1
                thelist_test[j].append((bigtest[i][mye][j], gold))
        ind = -1
        for sublist in thelist_test:
            ind += 1
            (good, bad, total) = computeFscore(sublist, typethresholdMatrix[i][ind])
            (p, r, f) = calcPRF(good, bad, total)
            goods[ind] += good; bads[ind] += bad; totals[ind] += total
            typeFscoreMatrix[i][ind] = f
            overalfscorestype[ind] += f
            precAtk = precisionAt(sublist, topk)
            overalPrec[ind] += precAtk
    
    print '**Macro avg F per type: '
    for i in range(len(overalfscorestype)):
        print i, " ", ff.format(overalfscorestype[i] / len(myt2i))
             
    print '**Average Prec@ ', topk, ':'
    for i in range(len(overalPrec)):
        print i, " ", ff.format(overalPrec[i] / len(myt2i))
 
    print '\n**Micro Results'
    for i in range(numScorePerType):
        (pr , re, f ) = calcPRF(goods[i], bads[i], totals[i])
        print 'Prec: ', ff.format(pr), ' Reca: ', ff.format(re), ' F1: ',ff.format(f)
    
etest2f = filltest2freq(Etestfile)

print matrixdev, matrixtest
(t2i,t2f) = fillt2i(typefilename, numtype)


e2i_dev = readdsfile(Edevfile, t2i)
(bigdev, numScorePerType, edev2freq) = loadEnt2ScoresFile(matrixdev, upto, numtype, donorm)

e2i_test = readdsfile(Etestfile, t2i)
(bigtest, numScorePerTypetest, etest2sampledfreq) = loadEnt2ScoresFile(matrixtest, upto, numtype, donorm)
assert numScorePerTypetest == numScorePerType

# assert len(e2i_test) == len(bigtest[0])
print len(bigtest[0]), ' ', len(e2i_test)
typethresholdMatrix = [[0 for x in range(numScorePerType)] for x in range(numtype)] 
firstrun = True
for i in range(numtype):
    thelist = []
#     print 'calc theta for type: ', i
    for j in range(numScorePerType):
        thelist.append([])
        for mye in bigdev[i]:
            gold = 0
            if onlynt == False and i in e2i_dev[mye]:
                gold = 1
            elif onlynt == True and i == e2i_dev[mye]:
                gold = 1
#             if numScorePerType == 1:
#                 thelist[j].append((bigdev[i][mye], gold))
#             else: 
            thelist[j].append((bigdev[i][mye][j], gold))
    ind = -1
    for sublist in thelist:
        ind += 1
        best = findbesttheta(sublist)
        typethresholdMatrix[i][ind] = best[1]



t2i_list = divideTypes(t2f, t2i)
for i in range(len(typefreq)):
    print '-------\nResults for types with freq <=',typefreq[i]    
    myt2i = t2i_list[i]
    calcPrintMeasures(myt2i)
        
i = len(typefreq)
print '-------\nResult for types with freq >', typefreq[i - 1]
myt2i = t2i_list[i]
print 'num of types:', len(myt2i)
calcPrintMeasures(myt2i)

print '-------\nResult for All types'
print 'num of types:', len(t2i)
calcPrintMeasures(t2i)

        


