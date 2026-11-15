---
name: shu-mi-original-session-preparation
description: 将同一被试五份作者已切段SHU-MI源session MAT按原DWDaNet/DWTNet数据流程整理为可追溯训练/测试集；不当成连续raw二次切段，不执行训练或分类。
---

输入必须为同一被试session1–5各一份、命名明确的源MAT；data为有限浮点[N,32,1000]，labels为1/2。官方sidecar证明250Hz、32通道、微伏uV及M1单极参考，MAT本身未保存这些元信息，不能依据形状猜测。

两种有版本的划分：

- `shu-session1to4-vs5-v1`：session1–4顺序拼接为训练，session5为测试。
- `shu-mixed-session-stratified70-v1`：混合同一被试五session，逐类RandomState(2026+subject)分层70/30，与原no_cross_session脚本算法一致；此划分不等于跨session泛化评估。

两个模型family共享该数据准备流程，不虚构DWDaNet/DWTNet在此阶段的数值差异。保持原试次、通道顺序，不滤波、不去均值、不重参考、不重采样、不拟合归一化。train.mat/test.mat保持原float32微伏；仅将指定数量测试试次换算为V登记推理sample。模型归一化统计必须与V单位一致；当前未验证32通道分类权重。

标签只用于划分，不能进入分类输入。产物保存五源哈希、原脚本哈希、源session/trial索引、split/seed、软件版本及实现哈希。每次创建独立目录，不覆盖用户原始source或Cxx文件。上传单份和合计最多128MiB，估计处理峰值最多512MiB，复用服务的单任务锁与内存检查。

官方证据：

- [数据集](https://figshare.com/articles/software/shu_dataset/19228725)
- [250Hz与参考](https://ndownloader.figshare.com/files/34166160)
- [通道顺序与microV](https://ndownloader.figshare.com/files/34166154)
- [1左手、2右手](https://ndownloader.figshare.com/files/34166166)
- [保存4–8秒MI阶段及访问要求](https://ndownloader.figshare.com/files/36729006)
