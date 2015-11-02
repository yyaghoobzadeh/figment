'''
Created on Nov 1, 2015

@author: yy1
'''
import numpy, os, sys, logging, string, random
from _collections import defaultdict
import gzip, bz2
import multiprocessing
from multiprocessing.pool import Pool
import theano
import math
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger('myutils')

random.seed(100000)

def str_to_bool(s):
    if s == 'True':
        return True
    elif s == 'False':
        return False
    else:
        raise ValueError # evil ValueError that doesn't tell you what the wrong value was
    
class NoDaemonProcess(multiprocessing.Process):
    def _get_daemon(self):
        return False
    def _set_daemon(self, value):
        pass
    daemon = property(_get_daemon, _set_daemon)

class MyPool(Pool):
    Process = NoDaemonProcess

def myopen(filename, mode='r'):
    """
    Open file. Use gzip or bzip2 if appropriate.
    """
    if filename.endswith('.gz'):
        return gzip.open(filename, mode)

    if filename.endswith('.bz2'):
        return bz2.BZ2File(filename, mode)

    return open(filename, mode)

def loadConfig(configFile):
    config = {}
    f = open(configFile, 'r')
    for line in f:
        if "#" == line[0]:
            continue  # skip commentars
        line = line.strip()
        if line == '': continue
        parts = line.split('=')
        name = parts[0]
        value = parts[1]
        config[name] = value
    f.close()
    return config

def openloud(myname):
    myfile = open(myname, 'r')
    print 'now open:', myname
    return myfile
def readall(filename, loud=True):
    if loud:
        myfile = openloud(filename)
    else:
        myfile = open(filename, 'r')
    mystring = myfile.read()
    myfile.close()
    return mystring

def getlines(filename):
    fcontents = readall(filename)
    lines = string.split(fcontents, '\n')
    return lines

def getlinesm1(filename, loud=True):
    fcontents = readall(filename, loud)
    lines = string.split(fcontents, '\n')
    if len(string.strip(lines[-1])) == 0:
        lines = lines[:-1]
    return lines

def getentparts(ent_token):
    ent_parts = ent_token.split('##')
    parts = ent_parts[0].split('/')
    if len(parts) < 4:
        return (ent_parts[0], 'unk', ent_parts[1])
    mid = '/m/' + parts[2]
    tokens = parts[3].split('_')
    if len(ent_parts) < 2:
        print ent_token
    notabletype = ent_parts[1]
    return (mid, tokens, notabletype)

def convertTargetsToBinVec(other_types_ind, n_out):
    outvec = numpy.zeros(n_out, numpy.int32)
#     outvec[nt_ind] = 1
    for ind in other_types_ind:
        outvec[ind] = 1
    return outvec

def parsethesampleline(myline, type2ind):
    parts = myline.split('\t')
    myent = parts[1]
    notable_type = parts[2]
    if '(' in notable_type:
        notable_type = notable_type[0:notable_type.index('(')].strip()
    if notable_type not in type2ind:
        print 'type not found: ' + notable_type  
        assert 1
    ntindex = type2ind[notable_type]
    other_types = parts[3].split(',')
    types_ind = []
    for i in range(0,len(other_types)):
        if other_types[i] not in type2ind:
            continue
        types_ind.append(type2ind[other_types[i]])
    types_ind.append(ntindex)
    return (myent, ntindex, types_ind, parts[4].split(' '))
    
def getwindowcontext(tokens, myent, maxwindow):
    onewaysize = maxwindow / 2
    c = 0; entind = 0
    newtokens = []
    for word in tokens:
        if "/m/" in word:
            (mid, ent_words, notabletype) = getentparts(word)
            if mid == myent and entind == 0:
                entind = c
            newtokens.append(notabletype)
        else:
            newtokens.append(word)
        c += 1
    assert len(newtokens) == len(tokens) 
    thiscontext = ''
    for i in range(entind - onewaysize, entind + onewaysize + 1):
        if i < 0 or i >= len(newtokens):
            thiscontext += '<PAD>'
        else:
            thiscontext += newtokens[i]
        thiscontext += ' '
    return thiscontext.strip()  

def read_lines_data(myname, type2ind, maxwindow=10, targetlabel='nt', upto=-1):
    res_vec = []; res_vec_all = []
    contextlist = []
    tmp = getlinesm1(myname)
    print 'lines loaded from', myname
    print 'now building the context windows'
    mycount = -1
    for myline in tmp:
        mycount += 1
        if upto >= 0 and mycount >= upto: break
        (myent, nt, alltypes, tokens) = parsethesampleline(myline.strip(), type2ind)
        res_vec.append(nt)
        res_vec_all.append(convertTargetsToBinVec(alltypes, len(type2ind)))
        thiscontext = getwindowcontext(tokens, myent, maxwindow)
        contextlist.append(thiscontext.strip())
    return(contextlist, res_vec, res_vec_all)

def loadTargets(filename):
    # Filling type indexes from a type list file
    targetIndMap = dict()
    typefreq_traindev = dict()
    c = 0
    f = open(filename, 'r')
    for line in f:
        t = line.split('\t')[0].strip()
        targetIndMap[t] = c
        if len(line.split('\t')) < 3:
            typefreq = 0
        else:
            typefreq = line.split('\t')[2].strip()
        if typefreq == 'null':
            typefreq_traindev[t] = 0
        else:
            typefreq_traindev[t] = int(typefreq)
        c += 1
    return (targetIndMap, len(targetIndMap), typefreq_traindev) 

def yyreadwordvectors(filename, upto=-1):
  # reading word vectors
  print 'loading word vectors'
  tmp = getlinesm1(filename)
  wordvectors = defaultdict(list)
  vectorsize = 0
  count = 0
  vectorsize = len(tmp[1].split()) - 1
  wordvectors['<UNK>'] = [0.001 for _ in range(vectorsize)]
  for line in tmp:
      if count == 0:
        count += 1  
        continue
      count += 1
      if upto != -1 and count > upto:
          break;
      line = line.strip()
      parts = line.split()
      word = parts[0].strip()
      parts.pop(0)
      wordvectors[word] = map(float, parts)
  print "vector size is: " + str(vectorsize)
  print len(wordvectors)
  return (wordvectors, vectorsize)

def buildtypevectorsmatrix(type2ind, wordvecs, vectorsize):
    m = numpy.zeros(shape=(len(type2ind), vectorsize))
    for t in type2ind:
        ind = type2ind[t]
        m[ind] = wordvecs[t]
    return m
    
def calcInsertMeanInputs(matrix, slotPosition, meanwindow):
    l = meanwindow / 2
    meanvec = numpy.mean(matrix[:,slotPosition-l:slotPosition+l+1], axis=1)
    matrix[:,slotPosition] = meanvec
#     for i in range(0, matrix.shape[0]):
#         matrix[i][slotPosition] = meanofrows[i] 
    return matrix, meanvec

def adeltheanomatrix_flexible(slotposition, vectorsize, contextlist, wordvectors, leftsize, rightsize, sum_window=10, insertsum=True):
    numvec = len(contextlist)
    sample = 0
    desired_window_size = leftsize + rightsize + 1
    myMatrix = numpy.zeros(shape=(numvec, vectorsize * desired_window_size))
    while True:
        if sample >= len(contextlist): break
        thiscontext = contextlist[sample]
        contextWords = thiscontext.split()
        assert len(contextWords) >= desired_window_size
        matrix = numpy.zeros(shape=(vectorsize, len(contextWords)))
        for i in range(0, len(contextWords)):
            if i == slotposition: continue
            word = contextWords[i]
            if word == '<PAD>': continue
            if word in wordvectors:
                curvec = wordvectors[word]
            else:
                curvec = wordvectors['<UNK>']
            matrix[:,i] = curvec
        if insertsum:
            (matrix, meanvec) = calcInsertMeanInputs(matrix, slotposition, sum_window)
        newlimitedmatrix = matrix[:,slotposition-leftsize:slotposition+rightsize+1]
        newlimitedmatrix = numpy.reshape(newlimitedmatrix, vectorsize * desired_window_size)
        myMatrix[sample, :] = newlimitedmatrix
        sample += 1
    return myMatrix

def load_features(typematrix, slotposition, vectorsize, contextlist, wordvectors, leftsize, rightsize, sum_window=10):
    numvec = len(contextlist)
    sample = 0
    desired_window_size = leftsize + rightsize + 1
    myMatrix = numpy.zeros(shape=(numvec, vectorsize * desired_window_size + len(typematrix))) # +1 for the cosine sim vector
    while True:
        if sample >= len(contextlist): break
        thiscontext = contextlist[sample]
        contextWords = thiscontext.split()
        assert len(contextWords) >= desired_window_size
        matrix = numpy.zeros(shape=(vectorsize, len(contextWords)))
        for i in range(0, len(contextWords)):
            if i == slotposition: continue
            word = contextWords[i]
            if word == '<PAD>': continue
            if word in wordvectors:
                curvec = wordvectors[word]
            else:
                curvec = wordvectors['<UNK>']
            matrix[:,i] = curvec
        
        (matrix, meanvec) = calcInsertMeanInputs(matrix, slotposition, sum_window)
        
        newlimitedmatrix = matrix[:,slotposition-leftsize:slotposition+rightsize+1]
        newlimitedmatrix = numpy.reshape(newlimitedmatrix, vectorsize * desired_window_size)
        
        sim_vec = cosine_similarity(meanvec, typematrix)
        featurevec = numpy.concatenate((newlimitedmatrix, sim_vec[0]))
        
        myMatrix[sample, :] = featurevec
        sample += 1
    return myMatrix

def loadTypesAndVectors(targetTypesFile, vectorFile, upto=-1):
    (type2ind, n_targets, typefreq_traindev) = loadTargets(targetTypesFile)
    (wordvectors, vectorsize) = yyreadwordvectors(vectorFile, upto)
    return (type2ind, n_targets, wordvectors, vectorsize, typefreq_traindev)

def getNumberOftypeInset(resultVectorDev, tt):
    c = 0
    for t in resultVectorDev:
        if t == tt:
            c += 1
    return c

def getRandomY(resultAllVec, n, target_labels):
    new_list = list(target_labels)
    for ind in numpy.nonzero(resultAllVec)[0]:
        new_list.remove(ind)
    return random.sample(new_list, n)

def fillOnlyEntityData(myname, vectorsize, wordvectors, type2ind, n_targets, upto=-1, ds='train', binoutvec=False, convbinf=convertTargetsToBinVec):
    resultVector = []
    allTypesResultVector = []
    binntvec = []
    inputEntities = []
    tmp = getlinesm1(myname)
    numinputs = len(tmp)
    myMatrix = numpy.empty(shape=(numinputs, vectorsize))
    c = 0
    for myline in tmp:
        myline = myline.strip()
        parts = myline.split('\t')
        ent = parts[0]
        target = parts[1]
        if target not in type2ind and ds != 'test':
            print 'type not found: ' + target  
            continue;
        types_ind = []
        if len(parts) >= 3:
            other_types = parts[2].split(' ')
            for i in range(0, len(other_types)):
                if other_types[i] not in type2ind:
                    continue
                types_ind.append(type2ind[other_types[i]])    
        if ent not in wordvectors: 
            print ent + ' not in vectors'
            if ds == 'test':
                myMatrix[c, :] = numpy.zeros(shape=vectorsize, dtype=theano.config.floatX)
            else: 
                continue
        else:    
            myMatrix[c, :] = wordvectors[ent]
        inputEntities.append(ent)
        if target in type2ind:
            resultVector.append(type2ind[target])
            types_ind.append(type2ind[target])
        else:
            resultVector.append(0)
        binvec = convbinf(types_ind, n_targets)
        if binoutvec == True:
            allTypesResultVector.append(binvec)
        else:
            allTypesResultVector.append(types_ind)
        binntvec.append(convbinf([], n_targets)) #TODO: buggg    
        c += 1
        if c == upto and upto != -1:
            break
#     targetMatrix = numpy.empty(shape=(c, vectorsize))
#     for i in xrange(0, c):
#         targetMatrix[i] = myMatrix[i] 
#     print 'length targetMatrix: ' + str(len(targetMatrix))
    return(resultVector, myMatrix, inputEntities, allTypesResultVector, binntvec)

def debug_print(var, name, PRINT_VARS=True):
    """Wrap the given Theano variable into a Print node for debugging.

    If the variable is wrapped into a Print node depends on the state of the
    PRINT_VARS variable above. If it is false, this method just returns the
    original Theano variable.
    The given variable is printed to console whenever it is used in the graph.

    Parameters
    ----------
    var : Theano variable
        variable to be wrapped
    name : str
        name of the variable in the console output

    Returns
    -------
    Theano variable
        wrapped Theano variable

    Example
    -------
    import theano.tensor as T
    d = T.dot(W, x) + b
    d = debug_print(d, 'dot_product')
    """

    if PRINT_VARS is False:
        return var

    return theano.printing.Print(name)(var)

def buildcosinematrix(matrix1, matrix2):
    """
    Calculating pairwise cosine distance using matrix1 matrix2 multiplication.
    """
    logger.info('Calculating pairwise cosine distance using matrix1 multiplication.')
    dotted = matrix1.dot(matrix2.T)
    matrix_norms = numpy.matrix(numpy.linalg.norm(matrix1, axis=1))
    matrix2_norms = numpy.matrix(numpy.linalg.norm(matrix2, axis=1))
    norms = numpy.multiply(matrix_norms.T, matrix2_norms)
    sim_matrix = numpy.divide(dotted, norms)
    
    return sim_matrix


def buildtypevecmatrix(t2ind, allvectors, vectorsize):
    typevecmatrix = numpy.zeros(shape=(len(t2ind), vectorsize), dtype=numpy.float64)
    for myt in t2ind:
        if myt not in allvectors: continue
        i = t2ind[myt]
        typevecmatrix[i] = allvectors[myt]
    return typevecmatrix    
    
def extend_in_matrix(initialmatrix, newmatrix):
    """
    The two matrix should have the same row numbers. 
    """
    num_new_col = newmatrix.shape[1]
    num_old_col = initialmatrix.shape[1]
    biggermatrixtrn = numpy.zeros(shape=(len(initialmatrix), num_old_col + num_new_col))
    biggermatrixtrn[:,0:num_old_col] = initialmatrix
    biggermatrixtrn[:,num_old_col:] = newmatrix
    return biggermatrixtrn


percentiles = [-1]
absolutes = []
Etestfreq = [5, 100]
# Etestfreq = [10]
EtestRelFreq = [10]
typefreq = [200, 3000]
ff = "{:10.3f}"
logistic = False
softmaxnorm = True

def softmax(w, t=1.0):
    """Calculate the softmax of a list of numbers w.
    @param w: list of numbers
    @return a list of the same length as w of non-negative numbers
    >>> softmax([0.1, 0.2])
    array([ 0.47502081,  0.52497919])
    >>> softmax([-0.1, 0.2])
    array([ 0.42555748,  0.57444252])
    >>> softmax([0.9, -10])
    array([  9.99981542e-01,   1.84578933e-05])
    >>> softmax([0, 10])
    array([  4.53978687e-05,   9.99954602e-01])
    """
    e = numpy.exp(numpy.array(w) / t)
    dist = e / numpy.sum(e)
    return dist

def divideTypes(t2f, t2i):
    t2i_list = []
    for i in range(len(typefreq) + 1):
        t2i_list.append(defaultdict(lambda: []))
    for t in t2i.keys():
        ind = 0
        for f in typefreq:
            if t2f[t][0] > f:
                ind += 1
        for i in range(ind, len(typefreq)):
            t2i_list[i][t]=t2i[t]
        if ind == len(typefreq):
            t2i_list[ind][t]=t2i[t]
    return t2i_list
    

def divideEtestByFreq(e2i, e2f, freqlist, predents, allow_miss_ent=False):
    e2i_list = []
    for i in range(len(freqlist) + 1):
        e2i_list.append(defaultdict(lambda: []))
    for e in e2i:
        if allow_miss_ent == True and e not in predents:
            continue
        ind = 0
        for f in freqlist:
            if e2f[e] > f:
                ind += 1
        for i in range(ind, len(freqlist)):
            e2i_list[i][e]=e2i[e]
        if ind == len(freqlist):
            e2i_list[ind][e]=e2i[e]
    return e2i_list
        
        

def getpercentile(mylist, mypercentile):
    if mypercentile == -1:
        return sum(mylist) / len(mylist)
    myindex = int(len(mylist) * mypercentile)
    assert myindex >= 0
    assert myindex < len(mylist)
    return mylist[myindex]

def getabsolute(mylist, myabsolute):
    assert myabsolute >= 0
    if myabsolute >= len(mylist):
        myabsolute = len(mylist) - 1
    return mylist[myabsolute]

def mynormalize(scores):
    normscores = [0.0 for i in range(len(scores))]
    mymax = max(scores)
    mymin = min(scores)
    if mymax == 0:
        return normscores
    if logistic:
        normscores = [sigmoid(scores[i]) for i in range(len(scores))] 
    elif softmaxnorm:
        normscores = softmax(scores)
    else:
        normscores = [(scores[i] - mymin) / (mymax - mymin) for i in range(len(scores))]
    return normscores
            
        
def getscores(myparts, numtype, donorm=False):

    numScorePerType = 1
    emscore = True
    if ',' in myparts[0]:
        numScorePerType = len(percentiles) + len(absolutes)
        emscore = False
    scores = [[0.0 for x in range(numtype)] for y in range(numScorePerType)]
    for i in range(numtype):
        if emscore == True:
            scores[0][i] = float(myparts[i])
            continue 
        subparts = myparts[i].split(',')
        for ind in range(numScorePerType):
            scores[ind][i] = float(subparts[ind])
    
    if donorm and emscore == True: #emscore == True
        for ind in range(numScorePerType):
            scores[ind] = mynormalize(scores[ind])
    return (scores, numScorePerType, emscore)

def findbesttheta(unsortedlist):
    mylist = sorted(unsortedlist, key=lambda tuple: tuple[0], reverse=True)
    total = 0
    for mypair in mylist:
        total += mypair[1]
    flist = []
    good = 0.0
    bad = 0.0
    for i in range(len(mylist)):
        if mylist[i][1] == 0:
            bad += 1
        elif mylist[i][1] == 1:
            good += 1
        else:
            assert 0
        prec = good / (good + bad)
        if good == 0 or total == 0:
            f = 0
        else:
            reca = good / total
            f = 2 / (1 / prec + 1 / reca)
        flist.append(f)
    mymax = max(flist)
    for i in range(len(mylist)):
        if flist[i] == mymax:
            return (mymax, mylist[i][0])
    assert 0

def computeFscore(unsortedlist, thetas):
    mylist = unsortedlist#sorted(unsortedlist, key=lambda tuple: tuple[0], reverse=True)
    total = 0
    for mypair in mylist:
        total += mypair[1]
    good = 0.0; bad = 0.0; fn = 0.0; tn = 0.0
    for i in range(len(mylist)):
        if not isinstance(thetas, list):
            theta = thetas
        else:
            theta = thetas[i]
        if mylist[i][0] >= theta:
            if mylist[i][1] == 1:
                good += 1.0
            else:
                bad += 1.0
        
    return (good, bad, total);

# return number of test_lines for each testset entities
def filltest2freq(Etestfile):
    etestfile = open(Etestfile)
    e2f = {}
    for myline in etestfile:
        myparts = myline.split('\t')
        mye = myparts[0].split('/')[2]
        if len(myparts) < 4:
            e2f[mye] = 100
        else:
            e2f[mye] = int(myparts[3])
    return e2f

def filltest2relfreq(etest2relnumfile):
    e2ffile = open(etest2relnumfile)
    e2f = {}
    for myline in e2ffile:
        myparts = myline.split(' ')
        mye = myparts[0].split('/')[2]
        e2f[mye] = int(myparts[1])
    return e2f

def readdsfile(dsfile, t2i, onlynt=False):
    e2i = {}
    etestfile = open(dsfile)
    for myline in etestfile:
        parts = myline.split('\t')
        othertypes = []
        if '/m/' not in parts[1]:
            mytype = parts[1]
        else:
            subparts = string.split(parts[1],'/')
            assert len(subparts)==3
            mytype = subparts[2]

        if len(parts) > 2:
            othertypes = parts[2].split()
        types = []
        for onet in othertypes:
            if '/m/' not in onet:
                t = onet
            else:
                t = onet.split('/')[2]
            if t in t2i and t != mytype:
                types.append(t2i[t])
        if mytype not in t2i:
            print mytype
            print 'mytype',mytype,'mytype'
            continue
        types.append(t2i[mytype])
        myne = parts[0].split('/')[2]
        if onlynt:
            e2i[myne] = t2i[mytype]
        else:
            e2i[myne] = types
    return e2i

def loadEnt2ScoresFile(matrix, upto, numtype, donorm=False):  
    print 'loading ent2type scores from', matrix  
    mtfile = open(matrix)
    lineno = -1
    t2e_scores = []
    numScorePerType = 1
    e2freq = {}
    for i in range(numtype):
        t2e_scores.append(defaultdict(lambda: []))
    for myline in mtfile:
        lineno += 1
        if upto>=0 and lineno>upto: break
        myparts = string.split(myline)
        myne = myparts[0]
        if '/m/' in myne:
            myne = myne.split('/')[2]
        (scores, numScorePerType, emscore) = getscores(myparts[1:], numtype, donorm)
        for i in range(numtype):
#             if emscore == True:
#                 t2e_scores[i][myne].append(float(scores[0][i])) # this is for embedding ent2type file
#                 continue
            for j in range(numScorePerType):
                t2e_scores[i][myne].append(float(scores[j][i]))
        if len(myparts[1:]) > numtype:
            e2freq[myne] = int(myparts[numtype + 1])
        
    return (t2e_scores, numScorePerType, e2freq)

def fillEnt2scoresBaseline(e2i_test, upto, Etrainfile, t2i):
    e2tTrain = readdsfile(Etrainfile, t2i, True)
    typefreq = [0 for i in range(len(t2i))]
    big = []
    for i in range(len(t2i)):
        big.append(defaultdict(lambda: []))
    for e in e2tTrain.keys():
#         for t in e2tTrain[e]:
        t = e2tTrain[e]
        typefreq[t] += 1
    for mye in e2i_test:
        for i in range(len(t2i)):
            big[i][mye] = typefreq[i]
    return big 
    
def fillt2i(typefilename,numtype):
    t2i = {}
    t2f = {}
    typefile = open(typefilename)
    i = -1
    for myline in typefile:
        i += 1
        if '/m/' in myline:
            myparts = string.split(myline)
            subparts = string.split(myparts[0],'/')
            assert len(subparts)==3
            assert subparts[1]=='m'
            t2i[subparts[2]] = i
        else:
            myparts = myline.split('\t')
            assert len(myparts) == 3
            t2i[myparts[0]] = i
            etrn_freq = int(myparts[1])
            contextfreq = int(myparts[2])
            t2f[myparts[0]] = (etrn_freq, contextfreq)
    return (t2i,t2f)
def calcPRF(good, bad, total):
    if (good + bad) == 0.0:
        prec = 0
    else:
        prec = good / (good + bad)
    if good == 0.0 or total == 0.0:
        f = 0.0
        reca = 0.0
    else:
        reca = good / total
        f = 2.0 / (1 / prec + 1 / reca)
    return (prec, reca, f);

def calcNNLBmeasures(unsortedlist, mintopscore=0.0):
    sortedlist = sorted(unsortedlist, key=lambda tuple: tuple[0], reverse=True)
    goodAt1 = sortedlist[0][1]
    topscore = sortedlist[0][0]
    best = findbesttheta(sortedlist)
    return (goodAt1, best[0], topscore)

def calcMeasuresBaseline(bigBaseline, e2i, numtype, onlynt='False'):
    goodsbase= 0.0; fbase = 0.0
    for mye in e2i.keys():
        e2tscores = []
        etypes = e2i[mye]
        for i in range(numtype):
                correct = 0
                if onlynt == 'False' and i in etypes:
                    correct = 1 
                elif onlynt == 'True' and i == etypes:
                    correct = 1
                e2tscores.append((bigBaseline[i][mye], correct))
        (goodAt1, fnn, topscore) = calcNNLBmeasures(e2tscores)
        goodsbase += goodAt1; 
        fbase += fnn
    prec1 = goodsbase / len(e2i)
    f = fbase / len(e2i)    
    return (prec1, f)

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def precisionAt(unsortedlist, topnum=20):
    mylist = sorted(unsortedlist, key=lambda tuple: tuple[0], reverse=True)
    total = 0
    for mypair in mylist:
        total += mypair[1] #mypair[1] is one for labeled entity and type
    if total < topnum:
        topnum = total
    good = 0.0; bad = 0.0; fn = 0.0; tn = 0.0

    for i in range(0, topnum):
        if mylist[i][1] == 1:
            good += 1
    if topnum == 0:
        return 0.0
    return good / topnum

def minimal_of_list(list_of_ele):
    if len(list_of_ele) ==0:
        return 1e10
    else:
        return list_of_ele[0]
    