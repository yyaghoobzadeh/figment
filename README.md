!(https://i.pinimg.com/originals/59/88/de/5988de958adf325cabbfa7e6e233ceb2.gif)

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


If you use any data or code, please cite the following paper:

```
@inproceedings{yaghoobzadeh2015,
  author    = {Yadollah Yaghoobzadeh and
               Hinrich Sch{\"{u}}tze},
  title     = {Corpus-level Fine-grained Entity Typing Using Contextual Information},
  booktitle = {Proceedings of the 2015 Conference on Empirical Methods in Natural
               Language Processing, {EMNLP} 2015, Lisbon, Portugal, September 17-21,
               2015},
  pages     = {715--725},
  year      = {2015},
  url       = {http://aclweb.org/anthology/D/D15/D15-1083.pdf}
}
```



## LICENSE

<a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-sa/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/">Creative Commons Attribution-ShareAlike 4.0 International License</a>.
