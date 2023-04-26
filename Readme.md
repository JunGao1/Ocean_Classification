## 程序结构
   run.py 主程序，包括网络训练、验证和测试
   python run.py --phase ['train']['test'] --flag ['human'] --save_path ['checkpoint/human'] \
   --gpu_id ['-1']['0'] ...

   dataset.py 数据集预处理
   model.py 模型文件
   checkpoint/human 目录下存放权重文件,其中
