# NLP (Natural Language Processing)
**Objective**: This repository is created to capture pointers, guidance for fundamentals around NLP (Natural Language Processing) from learning perspective, innovation / research areas etc. It also throws light into recommended subject areas, content relating to accelerating in the journey of learning in this field.

**Target Audience**: Data Science and AI Practitioners with already having fundamental, working knowledge and familiarity of Machine Learning concepts, Python/R/SQL programming background.

## Contents:
- [Research Focus and Trends](https://github.com/kkm24132/DataScience_NLP/blob/main/README.md#research-focus-and-trends)
- [Intro and Learning Content](https://github.com/kkm24132/DataScience_NLP/blob/main/README.md#intro-and-learning-content)
- [Techniques](https://github.com/kkm24132/DataScience_NLP/blob/main/README.md#techniques)
- [Libraries / Packages](https://github.com/kkm24132/DataScience_NLP/blob/main/README.md#libraries--packages)
- [Services](https://github.com/kkm24132/DataScience_NLP/blob/main/README.md#services)
- [Datasets](https://github.com/kkm24132/DataScience_NLP/blob/main/README.md#datasets)
- [Video and Online Content References](https://github.com/kkm24132/DataScience_NLP/blob/main/README.md#video-and-online-content-references)


## Research Focus and Trends
- Please keep referring to NLP related research papers from AAAI, NeurIPS, ACL, ICLR and similar conferences for latest research focus areas. Most of these may be captured in the arXiv.org site as well.
- Few latest and key research papers for reading are as follows: (Please note this keeps changing and may not be dated)
  - [WinoGrande: An Adversarial Winograd Schema Challenge at Scale](https://arxiv.org/abs/1907.10641) - the [GitHub](https://github.com/allenai/winogrande) page
  - [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/abs/1910.10683) - the [GitHub](https://github.com/google-research/text-to-text-transfer-transformer) page with pretrained models along with the dataset and code
  - [Reformer: The Efficient Transformer](https://arxiv.org/abs/2001.04451) - the [GitHub page with official code implementation from Google](https://github.com/google/trax/tree/master/trax/models/reformer) and the [GitHub page with PyTorch implementation of Reformer](https://github.com/lucidrains/reformer-pytorch)
  - [Longformer: The Long-Document Transformer](https://arxiv.org/abs/2004.05150) - the [GitHub page](https://github.com/allenai/longformer)
- [NLP-Progress](https://nlpprogress.com/) tracks the progress in Natural Language Processing, including the datasets and the current state-of-the-art for the most common NLP tasks.
- [NLP-Overview](https://nlpoverview.com/) is an up-to-date overview of deep learning techniques applied to NLP, including theory, implementations, applications, and state-of-the-art results. This is a great Deep NLP Introduction for researchers.
- [Detect Radiology related entities with Spark NLP](https://www.johnsnowlabs.com/detect-radiology-related-entities-with-spark-nlp/)
- [NLP's ImageNet moment](https://thegradient.pub/nlp-imagenet/)
- [ACL 2018 Highlights: Understanding Representations and Evaluation in More Challenging Settings](https://ruder.io/acl-2018-highlights/)
- [Four deep learning trends from ACL 2017 - Part 1](https://www.abigailsee.com/2017/08/30/four-deep-learning-trends-from-acl-2017-part-1.html) - Linguistic Structure and Word Embeddings
- [Four deep learning trends from ACL 2017 - Part 2](https://www.abigailsee.com/2017/08/30/four-deep-learning-trends-from-acl-2017-part-2.html) - Interpretability and Attention
- [Deep Learning for NLP: Advancements & Trends](https://tryolabs.com/blog/2017/12/12/deep-learning-for-nlp-advancements-and-trends-in-2017/?utm_campaign=Revue%20newsletter&utm_medium=Newsletter&utm_source=The%20Wild%20Week%20in%20AI)
- [Deep Learning for NLP : without Magic](https://www.socher.org/index.php/DeepLearningTutorial/DeepLearningTutorial)
- [Stanford NLP](https://nlp.stanford.edu/teaching/)
- [BERT, ELMo and GPT2](http://ai.stanford.edu/blog/contextual/) How contextual are Contexualized Word Representations? - from Stanford AI Lab
- [The Illustrated BERT, ELMo and others](http://jalammar.github.io/illustrated-bert/) NLP and transfer learning context
- [ALBERT: A Lite BERT for Self-supervised Learning of Language Representations](https://openreview.net/forum?id=H1eA7AEtvS), Related [Code](https://github.com/google-research/ALBERT)
- [A Mutual Information Maximization Perspective of Language Representation Learning](https://openreview.net/forum?id=Syx79eBKwr)
- [DeFINE: Deep Factorized Input Token Embeddings for Neural Sequence Modeling](https://openreview.net/forum?id=rJeXS04FPH)


[Back to Contents](https://github.com/kkm24132/DataScience_NLP/blob/main/README.md#contents)

## Research Focus Sub-Segments
- Lexical Semantics, Semantic Processing
- POS Tagging
- Discourse
- Paraphrasing / Entailment / Generation
- Machine Translation
- Information Retrieval
- Text Mining
- Information Extraction
- Question Answering
- Dialog Systems
- Spoken Language Processing
- Speech Recognition & Synthesis
- Computational Linguistics and NLP
- Chunking / Shallow Parsing
- Parsing / Grammatical Formalisms etc.

## Intro and Learning Content

Area           |Description                                     |Target Timeline |
:--            |:--                                             |      :--:      |
Pre-Requisites |<ul> <li>Familiarity with Python Programming - [Some Ref](https://github.com/kkm24132/Mentoring_Enablement/tree/master/Python)</li> <li> [Descriptive Stats](https://www.khanacademy.org/math/engageny-alg-1/alg1-2) by Khan Academy </li> <li> The Elements of Statistical Learning - [ISLR Book Reference by Hasti,Tishirani et al](https://web.stanford.edu/~hastie/Papers/ESLII.pdf)</li> <li> Machine Learning Fundamentals [Andrew Ng's course around ML](https://www.coursera.org/learn/machine-learning) </li> <li> Familiarity with Data Science processes and frameworks [CRISP-DM](https://en.wikipedia.org/wiki/Cross-industry_standard_process_for_data_mining) </li></ul> | Week 0 |
Handling Text Processing |<ul> <li>Text pre-processing techniques (Familiarity with [spaCy](https://spacy.io/usage) library, familiarity with [NLTK](https://www.nltk.org/) library, [Tokenization using spaCy library](https://medium.com/@makcedward/nlp-pipeline-word-tokenization-part-1-4b2b547e6a3), [Stopword removal and text normalization](https://www.analyticsvidhya.com/blog/2019/08/how-to-remove-stopwords-text-normalization-nltk-spacy-gensim-python/?utm_source=blog&utm_medium=learning-path-nlp-2020) )</li> <li> Regular expressions </li> <li> [Exploratory Analysis](https://towardsdatascience.com/a-complete-exploratory-data-analysis-and-visualization-for-text-data-29fb1b96fb6a) with Text data </li>  <li> [Extract Meta Features from text](https://towardsdatascience.com/understanding-feature-engineering-part-3-traditional-methods-for-text-data-f6f7d70acd41) </li> <li> Build a text classification model [Practice Problem - Identify Sentiments](https://datahack.analyticsvidhya.com/contest/linguipedia-codefest-natural-language-processing-1/?utm_source=blog&utm_medium=learning-path-nlp-2020#LeaderBoard) ..can be any such equivalent problem for experience </li></ul> | Week 1-4 |
Language Modeling & Sentiment Classification with DL, Translation with RNNs | <ul> <li>Language Model</li> <li>Transfer Learning</li>  <li>Sentiment Classification</li> <li> [Predicting English word version of numbers using an RNN](https://github.com/fastai/course-nlp/blob/master/6-rnn-english-numbers.ipynb) </li> <li> [Transfer Learning for Natural Language Modeling using imdb](https://github.com/fastai/course-nlp/blob/master/5-nn-imdb.ipynb) </li> </ul> | Week 5-8 |
Reading and handling Text from Images | <ul> <li>OpenCV - [Ref](https://opencv.org/about/)</li> <li>PyTesseract - [Tesseract software Wiki Ref](https://en.wikipedia.org/wiki/Tesseract_(software))</li> <li> [Here is an example to read text from images](https://towardsdatascience.com/read-text-from-image-with-one-line-of-python-code-c22ede074cac)</li> | Week 9-12 |

[fast.ai NLP Course Cool links](https://github.com/fastai/course-nlp)

[Back to Contents](https://github.com/kkm24132/DataScience_NLP/blob/main/README.md#contents)

## Techniques
- Text Embeddings
  - Word Embeddings
    - Thumb Rule: fastText >> GloVe > word2vec
    - Implementation from Facebook Research - [fastText](https://github.com/facebookresearch/fastText)
    - [gloVe](https://nlp.stanford.edu/pubs/glove.pdf) : Global Vectors for Word Representation - [Explainer Blog](https://blog.acolyer.org/2016/04/22/glove-global-vectors-for-word-representation/)
    - [word2vec](https://papers.nips.cc/paper/2013/file/9aa42b31882ec039965f3c4923ce901b-Paper.pdf) - [Implementation](https://code.google.com/archive/p/word2vec/) - [Explainer Blog](http://colah.github.io/posts/2014-07-NLP-RNNs-Representations/)
  - Sentence and Language Model Based Word Embeddings
    - ElMo : Embeddings from Language Models : [Basics](https://www.analyticsvidhya.com/blog/2019/03/learn-to-use-elmo-to-extract-features-from-text/) , [Deep contextualized word representations](https://arxiv.org/abs/1802.05365)
      - [PyTorch Implementation](https://github.com/allenai/allennlp) from [AllenAI/AllenNLP](https://allennlp.org/elmo)
      - [TF Implementation](https://github.com/allenai/bilm-tf) from AllenAI
    - ULMFiT : Universal Language Model Fine-tuning for Text Classification by Jeremy Howard and Sebastian Ruder - [Paper Ref](https://arxiv.org/abs/1801.06146)
    - InferSent - Supervised Learning of Universal Sentence Representations from Natural Language Inference Data by facebook - [Paper Ref](https://arxiv.org/abs/1705.02364)
- Question Answering and Knowledge Extraction
  - [DrQA](https://github.com/facebookresearch/DrQA) - Open Domain Question Answering work by Facebook Research on Wikipedia data
  - [Document-QA](https://github.com/allenai/document-qa) - Simple and Effective Multi-Paragraph Reading Comprehension by AllenAI
  - [Privee](https://www.sebastianzimmeck.de/zimmeckAndBellovin2014Privee.pdf) - An Architecture for Automatically Analyzing Web Privacy Policies
  - [Template-Based Information Extraction without the Templates](https://www.usna.edu/Users/cs/nchamber/pubs/acl2011-chambers-templates.pdf)

[Back to Contents](https://github.com/kkm24132/DataScience_NLP/blob/main/README.md#contents)

## Libraries / Packages
- R NLP Libraries
  - [text2vec](https://github.com/dselivanov/text2vec) - Fast vectorization, topic modeling, distances and GloVe word embeddings in R
  - [wordVectors](https://github.com/bmschmidt/wordVectors) - An R package for creating and exploring word2vec and other word embedding models
  - [RMallet](https://github.com/mimno/RMallet) - R package to interface with the Java machine learning tool MALLET
  - [dfr-browser](https://github.com/agoldst/dfr-browser) - Creates d3 visualizations for browsing topic models of text in a web browser.
  - [dfrtopics](https://github.com/agoldst/dfrtopics) - R package for exploring topic models of text.
  - [sentiment_classifier](https://github.com/kevincobain2000/sentiment_classifier) - Sentiment Classification using Word Sense Disambiguation and WordNet Reader
- Python NLP Libraries
  - [NLTK](https://www.nltk.org/) - Natural Language ToolKit
  - [TextBlob](https://textblob.readthedocs.io/en/dev/) - Simplified text processing. Providing a consistent API for diving into common natural language processing (NLP) tasks. Stands on the giant shoulders of Natural Language Toolkit (NLTK) and Pattern, and plays nicely with both
  - [spaCy](https://github.com/explosion/spaCy) - Industrial strength NLP with Python and Cython
  - [gensim](https://radimrehurek.com/gensim/index.html) - Python library to conduct unsupervised semantic modelling from plain text
  - [scattertext](https://github.com/JasonKessler/scattertext) - Python library to produce d3 visualizations of how language differs between corpora
  - [GluonNLP](https://github.com/dmlc/gluon-nlp) - A deep learning toolkit for NLP, built on MXNet/Gluon, for research prototyping and industrial deployment of state-of-the-art models on a wide range of NLP tasks.
  - [AllenNLP](https://github.com/allenai/allennlp) - An NLP research library, built on PyTorch, for developing state-of-the-art deep learning models on a wide variety of linguistic tasks.
  - [PyTorch-NLP](https://github.com/PetrochukM/PyTorch-NLP) - NLP research toolkit designed to support rapid prototyping with better data loaders, word vector loaders, neural network layer representations, common NLP metrics such as BLEU
  - [Rosetta](https://github.com/columbia-applied-data-science/rosetta) - Text processing tools and wrappers (e.g. Vowpal Wabbit)

[Back to Contents](https://github.com/kkm24132/DataScience_NLP/blob/main/README.md#contents)

## Services
- [Amazon Comprehend](https://aws.amazon.com/comprehend/) - NLP and ML suite covers most common tasks like NER (Named Entity Recognition), tagging, and sentiment analysis
- [Google Cloud Natural Language API](https://cloud.google.com/natural-language/) - Syntax Analysis, NER, Sentiment Analysis, and Content tagging in atleast 9 languages include English and Others
- [Microsoft Cognitive Service: Text Analytics](https://azure.microsoft.com/en-us/services/cognitive-services/text-analytics/)
- [IBM Watson's Natural Language Understanding](https://github.com/watson-developer-cloud/natural-language-understanding-nodejs) - API and Github demo
- [Cloudmersive](https://cloudmersive.com/nlp-api) - Unified and free NLP APIs that perform actions such as speech tagging, text rephrasing, language translation/detection, and sentence parsing
- [ParallelDots](https://www.paralleldots.com/text-analysis-apis) - High level Text Analysis API Service ranging from Sentiment Analysis to Intent Analysis
- [Wit.ai](https://github.com/wit-ai/wit) - Natural Language Interface for apps and devices
- [Rosette](https://www.rosette.com/) - An adaptable platform for text analytics and discovery
- [TextRazor](https://www.textrazor.com/) - Extract meaning from your text
- [Textalytic](https://www.textalytic.com/) - Natural Language Processing in the Browser with sentiment analysis, named entity extraction, POS tagging, word frequencies, topic modeling, word clouds, and more

[Back to Contents](https://github.com/kkm24132/DataScience_NLP/blob/main/README.md#contents)

## Datasets
- [NLP-datasets](https://github.com/niderhoff/nlp-datasets) - Great collection of NLP datasets for use
- [gensim-datasets](https://github.com/RaRe-Technologies/gensim-data) - Data repository for pretrained NLP models and NLP corpora

[Back to Contents](https://github.com/kkm24132/DataScience_NLP/blob/main/README.md#contents)

## Video and Online Content References
- [Stanford Deep Learning for Natural Language Processing (cs224-n)](https://web.stanford.edu/class/cs224n/) - Richard Socher and Christopher Manning's Stanford Course
- [Deep Natural Language Processing](https://github.com/oxford-cs-deepnlp-2017/lectures) - Lectures series from Oxford
- [Neural Networks for NLP - Carnegie Mellon](http://phontron.com/class/nn4nlp2017/) Language Technology Institute there
- [fast.ai Code-First Intro to Natural Language Processing](https://www.fast.ai/2019/07/08/fastai-nlp/) - This covers a blend of traditional NLP topics (including regex, SVD, naive bayes, tokenization) and recent neural network approaches (including RNNs, seq2seq, GRUs, and the Transformer), as well as addressing urgent ethical issues, such as bias and disinformation. Find the Jupyter Notebooks [here](https://github.com/fastai/course-nlp)
- [Deep NLP Course by Yandex Data School](https://github.com/yandexdataschool/nlp_course), covering important ideas from text embedding to machine translation including sequence modeling, language models and so on
- [Machine Learning University](https://www.youtube.com/playlist?list=PL8P_Z6C4GcuWfAq8Pt6PBYlck4OprHXsw) - Accelerated Natural Language Processing - Lectures go from introduction to NLP and text processing to Recurrent Neural Networks and Transformers. Material can be found [here](https://github.com/aws-samples/aws-machine-learning-university-accelerated-nlp)
- [Knowledge Graphs in Natural Language Processing @ ACL 2020](https://towardsdatascience.com/knowledge-graphs-in-natural-language-processing-acl-2020-ebb1f0a6e0b1)
- [NLP at Scale - MLOps aspects for customer success](https://docs.google.com/presentation/d/1tAeZXGSJDxhLrZcH_55h7mLH-bCWhZmBLdq-ybk9gb8/edit#slide=id.gc5f743980a_0_33)
- [MLM with BERT](https://towardsdatascience.com/masked-language-modelling-with-bert-7d49793e5d2c)

[Back to Contents](https://github.com/kkm24132/DataScience_NLP/blob/main/README.md#contents)



```
End of Contents
```


**Disclaimer:** Information represented here is based on my own experiences, learnings, readings and no way represent any firm's opinion, strategy etc or any individual's opinion or not intended for anything else other than learning and/or research/innovation in the field. Content here and on this repository is non-exhaustive and continuous improvement / continuous learning focus is needed to learn more. 
Recommendation - Keep Learning and keep improving.

