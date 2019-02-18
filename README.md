# BERT_Chinese_Classification
本实验，是用BERT进行中文情感分类，记录了详细操作及完整程序

本文参考[奇点机智的文章](https://www.jianshu.com/p/aa2eff7ec5c1)，记录自己在运行BERT中的一些操作。

BERT的代码同论文里描述的一致，主要分为两个部分。一个是**训练语言模型（language model）的预训练（pretrain）部分**。另一个是**训练具体任务(task)的fine-tune部分**。

在开源的代码中，预训练的入口是在run_pretraining.py而fine-tune的入口针对不同的任务分别在run_classifier.py和run_squad.py。

其中run_classifier.py适用的任务为分类任务。如CoLA、MRPC、MultiNLI这些数据集。而run_squad.py适用的是阅读理解(MRC)任务，如squad2.0和squad1.1。

因此如果要在自己的数据集上fine-tune跑代码，需要编写类似run_classifier.py的具体任务文件。

本实验，是用BERT进行中文情感分类，以下介绍具体操作步骤。

对于中文而言，google公布了一个参数较小的BERT预训练模型。具体参数数值如下所示：

```
Chinese Simplified and Traditional, 12-layer, 768-hidden, 12-heads, 110M parameters
```
## 下载预训练模型
模型的[下载链接](https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip)可以在github上google的开源代码里找到。对下载的压缩文件进行解压，可以看到文件里有五个文件，
- bert_model.ckpt开头的文件是负责模型变量载入的
- vocab.txt是训练时中文文本采用的字典
- bert_config.json是BERT在训练时，可选调整的一些参数

## 修改 processor

任何模型的训练、预测都是需要有一个明确的输入，而BERT代码中processor就是负责对模型的输入进行处理。我们以分类任务的为例，介绍如何修改processor来运行自己数据集上的fine-tune。在run_classsifier.py文件中我们可以看到，google对于一些公开数据集已经写了一些processor，如XnliProcessor,MnliProcessor,MrpcProcessor和ColaProcessor。这给我们提供了一个很好的示例，指导我们如何针对自己的数据集来写processor。

对于一个需要执行训练、交叉验证和测试完整过程的模型而言，自定义的processor里需要继承DataProcessor，并重载获取label的get_labels和获取单个输入的get_train_examples,get_dev_examples和get_test_examples函数。其分别会在main函数的FLAGS.do_train、FLAGS.do_eval和FLAGS.do_predict阶段被调用。
这三个函数的内容是相差无几的，区别只在于需要指定各自读入文件的地址。

以get_train_examples为例，函数需要返回一个由InputExample类组成的list。InputExample类是一个很简单的类，只有初始化函数，需要传入的参数中guid是用来区分每个example的，可以按照train-%d'%(i)的方式进行定义。text_a是一串字符串，text_b则是另一串字符串。在进行后续输入处理后(BERT代码中已包含，不需要自己完成) text_a和text_b将组合成[CLS] text_a [SEP] text_b [SEP]的形式传入模型。最后一个参数label也是字符串的形式，label的内容需要保证出现在get_labels函数返回的list里。

现在，我们想要处理一个能够分类文本的模型，现在在data的路径下有一个名为train_sentiment.txt的输入文件，如果我们现在输入文件的格式如下txt形式：

```
1	最大的优点也就是价钱比较实惠，另外有免费停车场如果住在古镇里面，白天是不允许把车开进去的。这个宾馆的服务员都说自己的餐厅做的当地小吃好吃，实在不敢苟同，份量少，味道也不地道，价格却不低，建议重视当地美食的朋友不要在宾馆的餐厅就餐，会对山西的小吃产生错误	2
2	华为回应CFO孟晚舟在加拿大被捕不实报道	2
```

那么我们可以写一个如下的get_train_examples的函数。如果文件为csv格式，对于csv的处理，可以使用诸如csv.reader的形式进行读入。

本实验中具体的Processor代码为:

```
# read txt
    #返回InputExample类组成的list
    #text_a是一串字符串，text_b则是另一串字符串。在进行后续输入处理后(BERT代码中已包含，不需要自己完成)
    # text_a和text_b将组合成[CLS] text_a [SEP] text_b [SEP]的形式传入模型
    def get_train_examples(self, data_dir):
        file_path = os.path.join(data_dir, 'train_sentiment.txt')
        f = open(file_path, 'r', encoding='utf-8')
        train_data = []
        index = 0
        for line in f.readlines():
            guid = 'train-%d' % index#参数guid是用来区分每个example的
            line = line.replace("\n", "").split("\t")
            text_a = tokenization.convert_to_unicode(str(line[1]))#要分类的文本
            label = str(line[2])#文本对应的情感类别
            train_data.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))#加入到InputExample列表中
            index += 1
        return train_data
```

同时对应文本分类这个任务(有3种情感标签)，get_labels函数可以写成如下的形式：

```
    def get_labels(self):
        return ['0', '1', '2']
```

在对get_dev_examples和get_test_examples函数做类似get_train_examples的操作后，便完成了对processor的修改。其中get_test_examples可以传入一个随意的label数值，因为在模型的预测（prediction）中label将不会参与计算。


## 修改 processor 字典

修改完成processor后，需要在在原本main函数的processor字典里，加入修改后的processor类，即可在运行参数里指定调用该processor。

```
    processors = {
        "cola": ColaProcessor,
        "mnli": MnliProcessor,
        "mrpc": MrpcProcessor,
        "xnli": XnliProcessor,
        "sim": SimProcessor,
    }
```

## 运行 fine-tune

之后就可以直接运行run_classsifier.py进行模型的训练。在运行时需要制定一些参数，一个较为完整的运行参数如下所示：


```
python3 run_classifier.py \
  --data_dir=data \
  --task_name=sim \
  --vocab_file=chinese_L-12_H-768_A-12/vocab.txt \
  --bert_config_file=chinese_L-12_H-768_A-12/bert_config.json \
  --output_dir=sim_model \
  --do_train=true \
  --do_eval=true \
  --init_checkpoint=chinese_L-12_H-768_A-12/bert_model.ckpt \
  --max_seq_length=70 \
  --train_batch_size=32 \
  --learning_rate=5e-5 \
  --num_train_epochs=3.0
```

如果要执行验证，则参数变为:

```
python3.6 run_classifier.py   \
--task_name=sim   \
--do_eval=true   \
--data_dir=data   \
--vocab_file=chinese_L-12_H-768_A-12/vocab.txt   --bert_config_file=chinese_L-12_H-768_A-12/bert_config.json   --init_checkpoint=sim_model   \
--max_seq_length=70   \
--output_dir=output
```


## BERT 源代码里还有什么
在开始训练我们自己fine-tune的BERT后，我们可以再来看看BERT代码里除了processor之外的一些部分。

我们可以发现，process在得到字符串形式的输入后，在file_based_convert_examples_to_features里先是对字符串长度，加入[CLS]和[SEP]等一些处理后，将其写入成TFrecord的形式。这是为了能在estimator里有一个更为高效和简易的读入。

我们还可以发现，在create_model的函数里，除了从modeling.py获取模型主干输出之外，还有进行fine-tune时候的loss计算。因此，如果对于fine-tune的结构有自定义的要求，可以在这部分对代码进行修改。如进行NER任务的时候，可以按照BERT论文里的方式，不只读第一位的logits，而是将每一位logits进行读取。

BERT这次开源的代码，由于是考虑在google自己的TPU上高效地运行，因此采用的estimator是tf.contrib.tpu.TPUEstimator,虽然TPU的estimator同样可以在gpu和cpu上运行，但若想在gpu上更高效地做一些提升，可以考虑将其换成tf.estimator.Estimator,于此同时model_fn里一些tf.contrib.tpu.TPUEstimatorSpec也需要修改成tf.estimator.EstimatorSpec的形式，以及相关调用参数也需要做一些调整。在转换成较普通的estimator后便可以使用常用的方式对estimator进行处理，如生成用于部署的.pb文件等。

