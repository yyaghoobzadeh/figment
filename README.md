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


More information about the models is in this paper:

<a href="https://github.com/yyaghoobzadeh/figment/blob/master/EMNLP2015figment.pdf
">*Corpus-level Fine-grained Entity Typing Using Contextual Information*</a>,
Yadollah Yaghoobzadeh, Hinrich schütze. (EMNLP2015). 


LICENSE

<a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-sa/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/">Creative Commons Attribution-ShareAlike 4.0 International License</a>.
