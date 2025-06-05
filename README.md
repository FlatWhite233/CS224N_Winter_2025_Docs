# CS224N

https://web.stanford.edu/class/cs224n/

# Week 1

## Word Vectors

[[slides](https://web.stanford.edu/class/cs224n/slides_w25/cs224n-2025-lecture01-wordvecs1.pdf)] [[notes](https://web.stanford.edu/class/cs224n/readings/cs224n_winter2023_lecture1_notes_draft.pdf)]

Suggested Readings:

1. [Efficient Estimation of Word Representations in Vector Space](http://arxiv.org/pdf/1301.3781.pdf) (original word2vec paper)
2. [Distributed Representations of Words and Phrases and their Compositionality](http://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf) (negative sampling paper)

## Word Vectors and Language Models

[[slides](https://web.stanford.edu/class/cs224n/slides_w25/cs224n-2025-lecture02-wordvecs2.pdf)] [[notes](https://web.stanford.edu/class/cs224n/readings/cs224n-2019-notes02-wordvecs2.pdf)] [[code](https://web.stanford.edu/class/cs224n/materials/gensim_2024.zip)]

Suggested Readings:

1. [GloVe: Global Vectors for Word Representation](http://nlp.stanford.edu/pubs/glove.pdf) (original GloVe paper)
2. [Improving Distributional Similarity with Lessons Learned from Word Embeddings](http://www.aclweb.org/anthology/Q15-1016)
3. [Evaluation methods for unsupervised word embeddings](http://www.aclweb.org/anthology/D15-1036)

Additional Readings:

1. [A Latent Variable Model Approach to PMI-based Word Embeddings](http://aclweb.org/anthology/Q16-1028)
2. [Linear Algebraic Structure of Word Senses, with Applications to Polysemy](https://transacl.org/ojs/index.php/tacl/article/viewFile/1346/320)
3. [On the Dimensionality of Word Embedding](https://papers.nips.cc/paper/7368-on-the-dimensionality-of-word-embedding.pdf)

## Python Review Session

[[slides](https://web.stanford.edu/class/cs224n/slides_w25/2024%20CS224N%20Python%20Review%20Session%20Slides.pptx.pdf)] [[colab](https://colab.research.google.com/drive/1hxWtr98jXqRDs_rZLZcEmX_hUcpDLq6e?usp=sharing)]

# Week 2

## Backpropagation and Neural Network Basics

[[slides](https://web.stanford.edu/class/cs224n/slides_w25/cs224n-2025-lecture03-neuralnets.pdf)] [[notes](https://web.stanford.edu/class/cs224n/readings/cs224n-2019-notes03-neuralnets.pdf)]

Suggested Readings:

1. [matrix calculus notes](https://web.stanford.edu/class/cs224n/readings/gradient-notes.pdf)
2. [Review of differential calculus](https://web.stanford.edu/class/cs224n/readings/review-differential-calculus.pdf)
3. [CS231n notes on network architectures](http://cs231n.github.io/neural-networks-1/)
4. [CS231n notes on backprop](http://cs231n.github.io/optimization-2/)
5. [Derivatives, Backpropagation, and Vectorization](http://cs231n.stanford.edu/handouts/derivatives.pdf)
6. [Learning Representations by Backpropagating Errors](http://www.iro.umontreal.ca/~vincentp/ift3395/lectures/backprop_old.pdf) (seminal Rumelhart et al. backpropagation paper)

Additional Readings:

1. [Yes you should understand backprop](https://medium.com/@karpathy/yes-you-should-understand-backprop-e2f06eab496b)
2. [Natural Language Processing (Almost) from Scratch](http://www.jmlr.org/papers/volume12/collobert11a/collobert11a.pdf)

## Dependency Parsing

[[slides](https://web.stanford.edu/class/cs224n/slides_w25/cs224n-2025-lecture04-dep-parsing.pdf)] [[notes](https://web.stanford.edu/class/cs224n/readings/cs224n-2019-notes04-dependencyparsing.pdf)]

Suggested Readings:

1. [Incrementality in Deterministic Dependency Parsing](https://www.aclweb.org/anthology/W/W04/W04-0308.pdf)
2. [A Fast and Accurate Dependency Parser using Neural Networks](https://www.emnlp2014.org/papers/pdf/EMNLP2014082.pdf)
3. [Dependency Parsing](https://link.springer.com/book/10.1007/978-3-031-02131-2)
4. [Globally Normalized Transition-Based Neural Networks](https://arxiv.org/pdf/1603.06042.pdf)
5. [Universal Stanford Dependencies: A cross-linguistic typology](http://nlp.stanford.edu/~manning/papers/USD_LREC14_UD_revision.pdf)
6. [Universal Dependencies website](http://universaldependencies.org/)
7. [Jurafsky & Martin Chapter 19](https://web.stanford.edu/~jurafsky/slp3/19.pdf)

## PyTorch Tutorial Session

[[colab](https://colab.research.google.com/drive/1Pz8b_h-W9zIBk1p2e6v-YFYThG1NkYeS?usp=sharing)]

# Week 3

## Basic Sequence Models to RNNs

[[slides](https://web.stanford.edu/class/cs224n/slides_w25/cs224n-2025-lecture05-rnnlm.pdf)] [[notes (lectures 5 and 6)](https://web.stanford.edu/class/cs224n/readings/cs224n-2019-notes05-LM_RNN.pdf)]

Suggested Readings:

1. [N-gram Language Models](https://web.stanford.edu/~jurafsky/slp3/3.pdf) (textbook chapter)
2. [The Unreasonable Effectiveness of Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/) (blog post overview)
3. [Sequence Modeling: Recurrent and Recursive Neural Nets](http://www.deeplearningbook.org/contents/rnn.html) (Sections 10.1 and 10.2)
4. [On Chomsky and the Two Cultures of Statistical Learning](http://norvig.com/chomsky.html)

## Advanced Variants of RNNs, Attention

[[slides](https://web.stanford.edu/class/cs224n/slides_w25/cs224n-2025-lecture06-fancy-rnn.pdf)] [[notes (lectures 5 and 6)](https://web.stanford.edu/class/cs224n/readings/cs224n-2019-notes05-LM_RNN.pdf)]

Suggested Readings:

1. [Learning long-term dependencies with gradient descent is difficult](https://ieeexplore.ieee.org/document/279181) (one of the original vanishing gradient papers)
2. [On the difficulty of training Recurrent Neural Networks](https://arxiv.org/pdf/1211.5063.pdf) (proof of vanishing gradient problem)
3. [Vanishing Gradients Jupyter Notebook](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1174/lectures/vanishing_grad_example.html) (demo for feedforward networks)
4. [Attention Is All You Need](https://arxiv.org/abs/1706.03762.pdf)

# Week 4

## Final Projects: Custom and Default; Practical Tips

[[slides](https://web.stanford.edu/class/cs224n/slides_w25/cs224n-2025-lecture07-final-project.pdf)]

Suggested Readings:

1. [Practical Methodology](https://www.deeplearningbook.org/contents/guidelines.html) (*Deep Learning* book chapter)

## Transformers

[[slides](https://web.stanford.edu/class/cs224n/slides_w25/cs224n-2025-lecture08-transformers.pdf)] [[Custom project tips](https://web.stanford.edu/class/cs224n/project/custom-final-project-tips.pdf)] [[notes](https://web.stanford.edu/class/cs224n/readings/cs224n-self-attention-transformers-2023_draft.pdf)]

Suggested Readings:

1. [Attention Is All You Need](https://arxiv.org/abs/1706.03762.pdf)
2. [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
3. [Transformer (Google AI blog post)](https://ai.googleblog.com/2017/08/transformer-novel-neural-network.html)
4. [Layer Normalization](https://arxiv.org/pdf/1607.06450.pdf)
5. [Image Transformer](https://arxiv.org/pdf/1802.05751.pdf)
6. [Music Transformer: Generating music with long-term structure](https://arxiv.org/pdf/1809.04281.pdf)
7. [Jurafsky and Martin Chapter 9 (The Transformer)](https://web.stanford.edu/~jurafsky/slp3/9.pdf)

# Week 5

## Pretraining

[[slides](https://web.stanford.edu/class/cs224n/slides_w25/cs224n-2025-lecture09-pretraining.pdf)]

Suggested Readings:

1. [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/pdf/1810.04805.pdf)
2. [Contextual Word Representations: A Contextual Introduction](https://arxiv.org/abs/1902.06006.pdf)
3. [The Illustrated BERT, ELMo, and co.](http://jalammar.github.io/illustrated-bert/)
4. [Jurafsky and Martin Chapter 11 (Masked Language Models)](https://web.stanford.edu/~jurafsky/slp3/11.pdf)

## Post-training (RLHF, SFT, DPO)

[[slides](https://web.stanford.edu/class/cs224n/slides_w25/cs224n-2025-lecture10-instruction-tunining-rlhf.pdf)]

Suggested Readings:

1. [Aligning language models to follow instructions](https://openai.com/research/instruction-following)
2. [Scaling Instruction-Finetuned Language Models](https://arxiv.org/abs/2210.11416)
3. [AlpacaFarm: A Simulation Framework for Methods that Learn from Human Feedback](https://arxiv.org/abs/2305.14387)
4. [How Far Can Camels Go? Exploring the State of Instruction Tuning on Open Resources](https://arxiv.org/abs/2306.04751)
5. [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290)

##  Hugging Face Transformers Tutorial Session

[[colab](https://colab.research.google.com/drive/13r94i6Fh4oYf-eJRSi7S_y_cen5NYkBm#scrollTo=OTsW-Wwi-X81)]

# Week 6

## Efficient Adaptation (Prompting + PEFT)

[[slides](https://web.stanford.edu/class/cs224n/slides_w25/cs224n-2025-lecture11-adapatation.pdf)]

Suggested Readings:

1. [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165)
2. [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](https://arxiv.org/abs/2201.11903)
3. [The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks](https://arxiv.org/abs/1803.03635)
4. [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
5. [Parameter-Efficient Transfer Learning for NLP](https://arxiv.org/abs/1902.00751)

## Benchmarking and Evaluation

[[slides](https://web.stanford.edu/class/cs224n/slides_w25/cs224n-2025-lecture12-evaluation-final.pdf)]

Suggested Readings:

1. [Challenges and Opportunities in NLP Benchmarking](https://www.ruder.io/nlp-benchmarking/)
2. [Measuring Massive Multitask Language Understanding](https://arxiv.org/abs/2009.03300)
3. [Holistic Evaluation of Language Models](https://arxiv.org/abs/2211.09110)
4. [AlpacaEval](https://tatsu-lab.github.io/alpaca_eval/)

# Week 7

## Question Answering and Knowledge

[[slides](https://web.stanford.edu/class/cs224n/slides_w25/cs224n-2025-lecture13-QA.pdf)]

Suggested readings:

1. [SQuAD: 100,000+ Questions for Machine Comprehension of Text](https://arxiv.org/abs/1606.05250)
2. [Dense Passage Retrieval for Open-Domain Question Answering](https://arxiv.org/abs/2004.04906)
3. [Bidirectional Attention Flow for Machine Comprehension](https://arxiv.org/abs/1611.01603)
4. [Reading Wikipedia to Answer Open-Domain Questions](https://arxiv.org/abs/1704.00051)
5. [REALM: Retrieval-Augmented Language Model Pre-Training](https://arxiv.org/abs/2002.08909)
6. [Lost in the Middle: How Language Models Use Long Contexts](https://arxiv.org/abs/2307.03172)