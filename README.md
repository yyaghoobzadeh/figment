# FIGMENT
Fine-grained embedding-based entity typing

<ul>
     <li><a href="http://cistern.cis.lmu.de/figment/entitydatasets.tar.gz">Entity Dataset</a></li>
     <li><a href="http://cistern.cis.lmu.de/figment/embeddings.txt">Entity Embeddings</a></li>
</ul>


To run the experiment, you have to download the two files.

Then, you can run the script:


```
sh gm.sh
```

It will train an MLP on train entities, apply on the test & dev entities, and finally output the 
measurements (micro F1, macro F1). 

Currently, only the Global Model (GM) code is available here. 


For more information, you can this paper:

*Corpus-level Fine-grained Entity Typing Using Contextual Information*.
Yadollah Yaghoobzadeh, Hinrich schütze. (EMNLP2015)
https://aclweb.org/anthology/D/D15/D15-1083.pdf

