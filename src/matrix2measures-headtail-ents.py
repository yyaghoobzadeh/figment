# for debugging: upto = 10000 will only look at the first 10000 lines

#finding thresholds from summarized scores for each entity2type

from myutils import * 
import string,collections, sys
config = loadConfig(sys.argv[1])
print 'loading cofing ',config
numtype = int (config['numtype'])
Etestfile = config['Etest'] #/nfs/datm/cluewebwork/nlu/experiments/entity-categorization/allTypes/sbj_datasets/17nov_conf_min0.5/cis_datasets/custom807/Etest
Edevfile = config['Edev'] #'/nfs/datm/cluewebwork/nlu/experiments/entity-categorization/allTypes/sbj_datasets/17nov_conf_min0.5/cis_datasets/custom807/Edev'
Etrainfile = config['Etrain']
matrixdev = config['matrixdev'] #'enttypescores_dev' + str(numtype)
matrixtest = config['matrixtest']#'enttypescores_test' + str(numtype)
typefilename = config['typefile'] #/nfs/datm/cluewebwork/nlu/experiments/entity-categorization/allTypes/experiments/807types/cis/context-datasets/rndTypes_trndevcontextFreq'
freqtype = 'mention'
if 'ent_freq_type' in config:
     freqtype = config['ent_freq_type']

donorm = str_to_bool(config['norm'])
onlynt = str_to_bool(config['onlynt'])
print '** norm is ', donorm
print '* onlynt is', onlynt
allow_missing_entity = False
if 'allow_miss_ent' in config:
    allow_missing_entity = str_to_bool(config['allow_miss_ent'])

upto = -1
print matrixdev, matrixtest
(t2i,t2f) = fillt2i(typefilename, numtype)

e2i_dev = readdsfile(Edevfile, t2i, onlynt)
(smalldev, numScorePerType, edev2freq) = loadEnt2ScoresFile(matrixdev, upto, numtype, donorm)


    
# assert len(e2i_test) == len(bigtest[0])
typethresholdMatrix = [[0 for x in range(numScorePerType)] for x in range(numtype)] 
firstrun = True
for i in range(numtype):
    thelist = []
#     print 'calc theta for type: ', i
    for j in range(numScorePerType):
        thelist.append([])
        for mye in smalldev[i]:
            if mye in edev2freq and edev2freq[mye] < 1:
                continue
            correct = 0
            if onlynt == False and i in e2i_dev[mye]:
                correct = 1
            elif onlynt == True and i == e2i_dev[mye]:
                correct = 1
#             if numScorePerType == 1:
#                 thelist[j].append((smalldev[i][mye], correct))
#             else: 
            thelist[j].append((smalldev[i][mye][j], correct))
    ind = -1
    for sublist in thelist:
        ind += 1
        best = findbesttheta(sublist)
        typethresholdMatrix[i][ind] = best[1]

# Thresholds are found and put into typethresholdMatrix--- Now we should apply thresholds and calc performance
def calcPrintMeasures(myetest, findGoodEnts=False):
    ######
    if len(myetest) == 0:
        print 'no test set with these conditions'
        return 
    correctEntititesP1 = []
    
    entityFscoreMatrix = [[0 for x in range(numScorePerType)] for x in range(len(myetest))] 
    fLooseMacro = [0.0 for x in range(numScorePerType)]
    precLooseMacro = [0.0 for x in range(numScorePerType)]
    recLooseMacro = [0.0 for x in range(numScorePerType)]
    goods = [0.0 for i in range(numScorePerType)]; bads = [0.0 for ind in range(numScorePerType)]; totals = [0.0 for ind in range(numScorePerType)];
    goodsAt1 = [0.0 for i in range(numScorePerType)]; f_nnlb = [0.0 for ind in range(numScorePerType)]; 
    strictGoods = [0.0 for i in range(numScorePerType)]
    
    for mye in myetest:
        e2tscores = []
        etypes = myetest[mye]
        for j in range(numScorePerType):
            e2tscores.append([])
            for i in range(numtype):
                correct = 0
                if onlynt == False and i in etypes:
                    correct = 1
                elif onlynt == True and i == etypes:
                    correct = 1
                if mye not in bigtest[i]:
                    e2tscores[j].append((0.0, correct))
                else:
                    e2tscores[j].append((bigtest[i][mye][j], correct))
        ind = -1
        for sublist in e2tscores:
            ind += 1
            (good, bad, total) = computeFscore(sublist, [typethresholdMatrix[i][ind] for i in range(numtype)])
#             (good, bad, total) = computeFscore(sublist, [0.5 for i in range(numtype)])
    #         print mye, 'good: ', good, 'bad: ', bad, 'total: ', total
            (p,r,f) = calcPRF(good, bad, total)
            
            goods[ind] += good; bads[ind] += bad; totals[ind] += total
            precLooseMacro[ind] += p; 
            recLooseMacro[ind] += r
            fLooseMacro[ind] += f                             
            if good == total and bad == 0: 
                strictGoods[ind] += 1
            (goodAt1, fnn, topscore) = calcNNLBmeasures(sublist)
            goodsAt1[ind] += goodAt1; f_nnlb[ind] += fnn
            
            if goodAt1 > 1:
                correctEntititesP1.append(mye)
    
    
    print '**Prec strict (per entity)---'
    for i in range(numScorePerType):
        print i, ff.format(strictGoods[i] / len(myetest))
    
    print '**Macro (per entity) --- '
    for i in range(numScorePerType):
        recLooseMacro[i] /= len(myetest)
        precLooseMacro[i] /= len(myetest)
        fLooseMacro[i] /= len(myetest)
        print i, " Prec: ", ff.format(precLooseMacro[i]), ' Reca: ', ff.format(recLooseMacro[i]), ' F1: ', ff.format(fLooseMacro[i])  
    
    print '**Micro (per entity or per type) --- '
    for i in range(numScorePerType):
        (pr , re, f ) = calcPRF(goods[i], bads[i], totals[i])
        print i, 'Prec: ', ff.format(pr), ' Reca: ', ff.format(re), ' F1: ',ff.format(f) 
    
    print '**measures based on NNLB: '
    for i in range(numScorePerType):
        print i, 'Prec@1: ', ff.format(goodsAt1[i] / len(myetest)), 'Avg F1: ', ff.format(f_nnlb[i] / len(myetest))
        
#     if findGoodEnts == True:
#         writeGoodEnts(correctEntititesP1)

        
def calcPrintBaseline(bigBaseline, mye2i, numtype, onlynt):
    if len(mye2i) == 0:
        return
    print len(mye2i)
    (prec1, f) = calcMeasuresBaseline(bigBaseline, mye2i, numtype, onlynt)
    print '**Most Frequent types baseline (based on NNLB): '
    print 'Prec@1: ', ff.format(prec1), 'Avg F1: ', ff.format(f)

e2i_test = readdsfile(Etestfile, t2i, onlynt)
(bigtest, numScorePerTypetest, alakifreq) = loadEnt2ScoresFile(matrixtest, upto, numtype, donorm)
print 'num of entities in bigtest: ', len(bigtest[0]), 'and number of test entities: ', len(e2i_test)

if 'rel' in freqtype:
    etest2f = filltest2relfreq(config['ent2relfreq'])
    e2i_test_list = divideEtestByFreq(e2i_test, etest2f, EtestRelFreq, bigtest[0], allow_missing_entity)
else:    
    etest2f = filltest2freq(Etestfile)
    e2i_test_list = divideEtestByFreq(e2i_test, etest2f, Etestfreq, bigtest[0], allow_missing_entity)
assert len(e2i_test_list) == len(Etestfreq) + 1

bigBaseline = fillEnt2scoresBaseline(e2i_test, upto, Etrainfile, t2i)

for i in range(len(Etestfreq)):
    print '-------\nresult for Etest with freq <=',Etestfreq[i]
    myetest = e2i_test_list[i]
    print 'num of entities:', len(myetest)
    calcPrintMeasures(myetest)
    calcPrintBaseline(bigBaseline, myetest, numtype, onlynt)

i = len(Etestfreq)
print '-------\nresult for Etest with freq >', Etestfreq[i - 1]
myetest = e2i_test_list[i]
print 'num of entities:', len(myetest)
if len(myetest) > 0:
    calcPrintBaseline(bigBaseline, myetest, numtype, onlynt)
    calcPrintMeasures(myetest)

print '-------\nresult for All Etest entities'
print 'num of entities:', len(e2i_test)
calcPrintBaseline(bigBaseline, e2i_test, numtype, onlynt)
calcPrintMeasures(e2i_test, findGoodEnts=True)

