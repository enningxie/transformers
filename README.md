# transformers
From huggingface/transformers.

Garbage In, Garbage Out.

train.tsv 238767

chinese-rbt3
在原有`train.tsv`数据上训练得到

在业务相关的badcase上的表现：
--> precision: 0.3.
--> recall: 0.39622641509433965.
--> f1 score: 0.34146341463414637.
--> accuracy score: 0.3025830258302583.

在原有`test.tsv`上的表现：
--> precision: 0.94144.
--> recall: 0.7995651583095529.
--> f1 score: 0.8647218752296275.
--> accuracy score: 0.85272.

chinese-rbt3-01
在训练集中加入了业务相关的badcase进行训练得到

训练完成后对badcase的表现：
--> precision: 0.9938775510204082.
--> recall: 0.8790613718411552.
--> f1 score: 0.9329501915708812.
--> accuracy score: 0.9138991389913899.

训练完成后在原有`test.tsv`上的表现：
--> precision: 0.97776.
--> recall: 0.7000801924619086.
--> f1 score: 0.8159423192469457.
--> accuracy score: 0.77944.

chinese-rbt3-02
在训练集中加入了业务相关的badcase进行训练得到，不同于一般模型微调epoch为5，该模型微调epoch为10

训练完成后对badcase的表现：
--> precision: 1.0.
--> recall: 0.9514563106796117.
--> f1 score: 0.9751243781094527.
--> accuracy score: 0.9692496924969249.

训练完成后在原有`test.tsv`上的表现：
--> precision: 0.968.
--> recall: 0.7365473581689798.
--> f1 score: 0.8365597345132744.
--> accuracy score: 0.81088.

chinese-rbt3-03
training 15 epoch with training data shuffled.

训练完成后对badcase的表现：
--> precision: 0.9877551020408163.
--> recall: 0.9918032786885246.
--> f1 score: 0.9897750511247443.
--> accuracy score: 0.9876998769987699.

训练完成后在原有`test.tsv`上的表现：
--> precision: 0.93392.
--> recall: 0.8093455352190793.
--> f1 score: 0.8671816966275442.
--> accuracy score: 0.85696.